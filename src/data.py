import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio as ta
import os
from os import path as osp
import pickle
import h5py
from omegaconf import DictConfig
import logging
from sklearn.manifold import TSNE
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.utils import proj_root, half_cpu_cores, get_device
from src.prep import prep_data, skip_prep
from src.models import Cnn14_Encoder
from src.torchlibrosa import stft as tl_stft
from src.torchlibrosa import augmentation as tl_augmentation


class LogMelExtractor:
    def __init__(self, config: DictConfig, train: bool, mean: float, std: float):
        self.config = config
        self.train = train
        self.mean, self.std = mean, std
        self.sr = config.sr
        self.n_fft = config.n_fft
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        self.window = config.window
        self.center = config.center
        self.pad_mode = config.pad_mode
        self.n_mels = config.n_mels
        self.fmin = config.fmin
        self.fmax = config.fmax
        self.ref = config.ref
        self.amin = config.amin
        self.top_db = config.top_db
        if hasattr(config, 'aug') and train:
            self.time_mask_prob = config.aug.time_mask_prob
            self.time_drop_width = config.aug.time_drop_width
            self.time_stripes_num = config.aug.time_stripes_num
            self.freq_mask_prob = config.aug.freq_mask_prob
            self.freq_drop_width = config.aug.freq_drop_width
            self.freq_stripes_num = config.aug.freq_stripes_num
            self.noise_prob = config.aug.noise_prob
            self.noise_coef_min = config.aug.noise_coef_min
            self.noise_coef_max = config.aug.noise_coef_max
        else:
            self.time_mask_prob = 0.
            self.time_drop_width = 0
            self.time_stripes_num = 0
            self.freq_mask_prob = 0.
            self.freq_drop_width = 0
            self.freq_stripes_num = 0
            self.noise_prob = 0.
            self.noise_coef_min = 0.
            self.noise_coef_max = 0.
        self.spec_extractor = tl_stft.Spectrogram(
            n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, window=self.window,
            center=self.center, pad_mode=self.pad_mode, freeze_parameters=True)
        self.logmel_extractor = tl_stft.LogmelFilterBank(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            ref=self.ref, amin=self.amin, top_db=self.top_db, freeze_parameters=True)
        self.time_masker = tl_augmentation.SpecAugmentation(
            time_drop_width=self.time_drop_width, time_stripes_num=self.time_stripes_num,
            freq_drop_width=0, freq_stripes_num=0)
        self.freq_masker = tl_augmentation.SpecAugmentation(
            time_drop_width=0, time_stripes_num=0,
            freq_drop_width=self.freq_drop_width, freq_stripes_num=self.freq_stripes_num)
        self.spec_extractor.eval()
        self.logmel_extractor.eval()
        self.time_masker.eval()
        self.freq_masker.eval()

    def extract(self, x):
        """
        Extract log-mel spectrogram from waveform, normalize it, and apply augmentation if needed
        """
        with torch.no_grad():
            if self.train and torch.rand(1).item() < self.noise_prob:
                noise_coef = (self.noise_coef_max - self.noise_coef_min) * torch.rand(1).item() + self.noise_coef_min
                x += torch.randn_like(x, device=x.device) * torch.std(x) * noise_coef
                x = torch.clamp(x, min=-1., max=1.)
            x = self.spec_extractor(x)
            x = self.logmel_extractor(x)
            # if you want to use batch_norm, dim 1 and 3 should be transposed at first and then transposed back
            x = (x - self.mean) / self.std
            x = self.time_masker(x) if self.train and (torch.rand(1).item() < self.time_mask_prob) else x
            x = self.freq_masker(x) if self.train and (torch.rand(1).item() < self.freq_mask_prob) else x
            return x


def merge_slots(input_slots: np.ndarray):
    """
    Merge overlapping slots to a single slot and return the merged non-overlapping slots
    """
    assert len(input_slots.shape) == 2
    assert input_slots.shape[1] == 2
    input_slots = input_slots[np.argsort(input_slots[:, 0])]
    merged_slots = np.array([input_slots[0]])
    for idx in range(1, len(input_slots)):
        cur_start = input_slots[idx][0]
        cur_end = input_slots[idx][1]
        prev_start = merged_slots[-1][0]
        prev_end = merged_slots[-1][1]
        if cur_start <= prev_end:
            merged_slots[-1][0] = prev_start
            merged_slots[-1][1] = np.maximum(prev_end, cur_end)
        else:
            merged_slots = np.vstack((merged_slots, [cur_start, cur_end]))
    return merged_slots


class DCASE2024Task5SLDataset(Dataset):
    def __init__(self, config: DictConfig, div: str, spl: str):
        assert div in ['al', 'sl'], f"Invalid div {div}, must be either 'al' or 'sl'"
        assert spl in ['train', 'valid', 'test'], f"Invalid division {spl}, must be either 'train', 'valid', or 'test'"
        self.data_home = osp.join(proj_root(), 'data', div.upper(), spl)
        if not skip_prep(osp.join(proj_root(), 'data')):
            prep_data(config=config)
        self.csv_file = osp.join(proj_root(), 'data', div.upper(), f'{spl}_label.csv')
        self.sample_df = pd.read_csv(self.csv_file)
        config = config.al.data if div == 'al' else config.sl.data
        self.config = config
        self.sr = config.sr
        self.clip_tlen = config.clip_tlen
        self.hop_tlen = config.hop_tlen
        self.hop_length = config.hop_length
        self.frames_per_point = Cnn14_Encoder().frames_per_point
        mean_std_pkl = osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl')
        if osp.exists(mean_std_pkl):
            with open(mean_std_pkl, 'rb') as f:
                self.mean, self.std = pickle.load(f)
        else:
            self.mean, self.std = 0., 1.
        if div == 'sl' and spl == 'train' and osp.exists(mean_std_pkl):
            self.train = True
            assert config.aug.rand_cut if config.aug.balanced else True, "rand_cut must be True if balanced is True"
            self.rand_cut = config.aug.rand_cut
            self.balanced = config.aug.balanced
        else:
            self.train = False
            self.rand_cut = False
            self.balanced = False
        self.logmel_extractor = LogMelExtractor(config=config, train=self.train, mean=self.mean, std=self.std)
        self.class_list = list(self.sample_df.columns)[3:]
        self.num_classes = len(self.class_list)
        self.time_slots = self.time_slots()
        self.audiofile_list = np.unique(self.sample_df['Audiofilename'].to_numpy())
        self.total_duration = 0.
        self.start_time_each_audiofile = [0.]
        self.total_num_clips = 0
        self.start_idx_each_audiofile = [0]
        for audiofile in self.audiofile_list:
            tmp_duration = sf.info(osp.join(self.data_home, str(audiofile))).duration
            tmp_num_clips = 1 + max(0, int(np.ceil((tmp_duration - self.clip_tlen) / self.hop_tlen)))
            self.total_duration += tmp_duration
            self.start_time_each_audiofile.append(self.start_time_each_audiofile[-1] + tmp_duration)
            self.total_num_clips += tmp_num_clips
            self.start_idx_each_audiofile.append(self.start_idx_each_audiofile[-1] + tmp_num_clips)
        pos_rows_df = self.sample_df[(self.sample_df.iloc[:, 3:] == 1).any(axis=1)]
        self.class_audiofile_slot_dict = {class_iter: {} for class_iter in (self.class_list + ['Negative'])}
        self.duration_each_class_each_file = np.zeros((len(self.class_list) + 1, len(self.audiofile_list)))
        for i, class_iter in enumerate(self.class_list):
            class_group = pos_rows_df[pos_rows_df[class_iter] == 1]
            for j, audiofile in enumerate(self.audiofile_list):
                tmp_duration = sf.info(osp.join(self.data_home, str(audiofile))).duration
                file_group = class_group[class_group['Audiofilename'] == audiofile]
                pos_slots = file_group[['Starttime', 'Endtime']].to_numpy()
                if len(pos_slots) == 0:
                    continue
                pos_slots[:, 0] = np.maximum(0., pos_slots[:, 0] - self.clip_tlen)
                merged_pos_slots = np.clip(merge_slots(pos_slots), a_min=0., a_max=tmp_duration)
                self.class_audiofile_slot_dict[class_iter][audiofile] = merged_pos_slots
                tmp_merged_pos_duration = np.sum(merged_pos_slots[:, 1] - merged_pos_slots[:, 0])
                self.duration_each_class_each_file[i, j] = tmp_merged_pos_duration
        for i, audiofile in enumerate(self.audiofile_list):
            tmp_duration = sf.info(osp.join(self.data_home, str(audiofile))).duration
            tmp_pos_slots = np.concatenate(
                [self.class_audiofile_slot_dict[class_iter][audiofile] if audiofile in self.class_audiofile_slot_dict[
                    class_iter].keys() else np.empty((0, 2)) for class_iter in self.class_list], axis=0)
            merged_pos_slots = merge_slots(tmp_pos_slots)
            tmp_neg_slots = [[0., merged_pos_slots[0][0]]] if merged_pos_slots[0][0] > 0. else []
            for j in range(1, merged_pos_slots.shape[0]):
                tmp_neg_slots.append([merged_pos_slots[j - 1][1], merged_pos_slots[j][0]])
            if merged_pos_slots[-1][1] < tmp_duration:
                tmp_neg_slots.append([merged_pos_slots[-1][1], tmp_duration])
            tmp_neg_slots = np.array(tmp_neg_slots)
            self.class_audiofile_slot_dict['Negative'][audiofile] = tmp_neg_slots
            self.duration_each_class_each_file[-1, i] = np.sum(tmp_neg_slots[:, 1] - tmp_neg_slots[:, 0])
        each_class_prob = 1. / (np.sum(self.duration_each_class_each_file, axis=1) + np.finfo(float).eps)
        self.each_class_prob = each_class_prob / np.sum(each_class_prob)
        self.each_class_each_file_prob = self.duration_each_class_each_file / (np.sum(
            self.duration_each_class_each_file, axis=1, keepdims=True) + np.finfo(float).eps)

    def time_slots(self):
        num_frames = int(np.floor(self.clip_tlen * self.sr / self.hop_length) + 1)
        frame_slice = np.arange(0, num_frames, self.frames_per_point)
        sample_point_slice = np.clip(frame_slice * self.hop_length - self.hop_length // 2, a_min=0,
                                     a_max=self.clip_tlen * self.sr)
        time_slice = sample_point_slice / self.sr
        time_slice = np.append(time_slice, self.clip_tlen)
        return time_slice

    def __len__(self):
        return self.total_num_clips

    def __getitem__(self, idx):
        if not self.rand_cut and not self.balanced:
            audiofile_idx = np.searchsorted(self.start_idx_each_audiofile, idx, side='right') - 1
            audiofile = self.audiofile_list[audiofile_idx]
            audiofile_path = osp.join(self.data_home, str(audiofile))
            start_time = self.hop_tlen * (idx - self.start_idx_each_audiofile[audiofile_idx])
        elif self.rand_cut and not self.balanced:
            tmp_time = np.random.rand() * self.total_duration
            audiofile_idx = np.searchsorted(self.start_time_each_audiofile, tmp_time, side='right') - 1
            audiofile = self.audiofile_list[audiofile_idx]
            audiofile_path = osp.join(self.data_home, str(audiofile))
            tmp_duration = sf.info(audiofile_path).duration
            if tmp_duration <= self.clip_tlen:
                start_time = 0.
            else:
                start_time = np.random.rand() * (tmp_duration - self.hop_tlen)
        elif self.rand_cut and self.balanced:
            class_idx = np.random.choice(len(self.class_list) + 1, p=self.each_class_prob)
            tmp_class = self.class_list[class_idx] if class_idx < len(self.class_list) else 'Negative'
            audiofile_idx = np.random.choice(len(self.audiofile_list), p=self.each_class_each_file_prob[class_idx, :])
            audiofile = self.audiofile_list[audiofile_idx]
            audiofile_path = osp.join(self.data_home, str(audiofile))
            tmp_slots = self.class_audiofile_slot_dict[tmp_class][audiofile]
            cumsum_duration = np.cumsum(tmp_slots[:, 1] - tmp_slots[:, 0])
            cumsum_duration = np.append(0., cumsum_duration)
            tmp_time = np.random.rand() * cumsum_duration[-1]
            start_idx = np.searchsorted(cumsum_duration, tmp_time, side='right') - 1
            start_time = tmp_slots[start_idx, 0] + tmp_time - cumsum_duration[start_idx]
        else:
            raise ValueError("Invalid combination of rand_cut and balanced")
        sig, _ = ta.load(audiofile_path, frame_offset=int(np.round(start_time * self.sr)),
                         num_frames=int(self.clip_tlen * self.sr) if start_time + self.clip_tlen <= sf.info(
                             audiofile_path).duration else -1)
        if sig.shape[-1] < self.clip_tlen * self.sr:
            sig = torch.nn.functional.pad(sig, (0, int(self.clip_tlen * self.sr - sig.shape[-1])), "constant", 0)
        tmp_df = self.sample_df[self.sample_df['Audiofilename'] == audiofile]
        tmp_slots = self.time_slots + start_time
        label = np.stack(
            [(tmp_df[(tmp_df['Starttime'] <= tmp_slots[i + 1]) & (tmp_df['Endtime'] >= tmp_slots[i])].iloc[:,
              3:] == 1).any(axis=0).to_numpy().astype(np.float32) for i in range(len(tmp_slots) - 1)])
        return self.logmel_extractor.extract(sig).squeeze(0), torch.Tensor(label)


def store_mean_std(config: DictConfig, div: str):
    assert div in ['al', 'sl'], f"Invalid div {div}, must be either 'al' or 'sl'"
    logging.info(
        f"No mean and std found in {osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl')}, start to calculate...")
    data_set = DCASE2024Task5SLDataset(config=config, div=div, spl='train')
    data_loader = DataLoader(data_set, batch_size=config.hyper.batch_size, shuffle=False, drop_last=False,
                             num_workers=half_cpu_cores(), pin_memory=False)
    sum_x = 0.
    sum_x2 = 0.
    num_elements = 0
    report_interval = max(len(data_loader) // 10, 1)
    with torch.no_grad():
        for i, (spec, _) in enumerate(data_loader):
            sum_x += spec.sum().item()
            sum_x2 += (spec ** 2).sum().item()
            num_elements += spec.numel()
            if (i + 1) % report_interval == 0 or (i + 1) == len(data_loader):
                logging.info(f"[{i + 1}/{len(data_loader)}] of the dataset has been processed.")
    mean = sum_x / num_elements
    std = np.sqrt((sum_x2 - (sum_x ** 2) / num_elements) / (num_elements - 1))
    with open(osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl'), 'wb') as f:
        pickle.dump((mean, std), f)
    logging.info(
        f"[Mean: {mean:.3f}] and [std: {std:.3f}] have been stored in {osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl')}")


def generate_dataset(config: DictConfig, div: str, spl: str):
    if not osp.exists(osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl')):
        store_mean_std(config=config, div=div)
    dataset = DCASE2024Task5SLDataset(config=config, div=div, spl=spl)
    return dataset


def generate_dataloader(config: DictConfig, div: str, spl: str, shuffle: bool = None):
    assert div in ['al', 'sl'], f"Invalid div {div}, must be either 'al' or 'sl'"
    assert spl in ['train', 'valid', 'test'], f"Invalid division {spl}, must be either 'train', 'valid', or 'test'"
    dataset = generate_dataset(config=config, div=div, spl=spl)
    batch_size = config.sl.trainer.batch_size if div == 'sl' else config.al.trainer.batch_size
    shuffle = dataset.train if shuffle is None else shuffle
    pin_memory = osp.exists(osp.join(proj_root(), 'data', div.upper(), 'mean_std.pkl')) and div == 'sl'
    num_workers = half_cpu_cores()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


def cos_sim(a: np.ndarray, b: np.ndarray, block_size: int = 10000):
    assert len(a.shape) == 2 and len(b.shape) == 2, 'The input must be 2D matrix'
    assert a.shape[1] == b.shape[1], 'The dimensions of two matrices must be the same'
    n_a, d = a.shape
    n_b, _ = b.shape
    similarity_matrix = np.empty((n_a, n_b))
    a_blocks = int(np.ceil(n_a / block_size))
    b_blocks = int(np.ceil(n_b / block_size))
    for i in range(a_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, n_a)
        a_iter = a[start_i:end_i]
        a_iter_norm = np.linalg.norm(a_iter, axis=1, keepdims=True)
        normed_a_iter = a_iter / (a_iter_norm + np.finfo(a_iter_norm.dtype).eps)
        # normed_a_iter = a_iter / a_iter_norm
        for j in range(b_blocks):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, n_b)
            b_iter = b[start_j:end_j]
            b_iter_norm = np.linalg.norm(b_iter, axis=1, keepdims=True)
            normed_b_iter = b_iter / (b_iter_norm + np.finfo(b_iter_norm.dtype).eps)
            # normed_b_iter = b_iter / b_iter_norm
            cos_sim_sub_mtx = np.dot(normed_a_iter, np.transpose(normed_b_iter, (1, 0)))
            similarity_matrix[start_i:end_i, start_j:end_j] = cos_sim_sub_mtx
    np.fill_diagonal(similarity_matrix, 1.)
    return np.clip(similarity_matrix, -1., 1.)


def encode_al_h5(config: DictConfig):
    def encode_spl(data_loader: DataLoader, spl: str):
        assert spl in ['train', 'valid', 'test'], f"Invalid division {spl}, must be either 'train', 'valid', or 'test'"
        h5file = osp.join(proj_root(), 'data', 'AL', f'{spl}_set.h5')
        encoder = Cnn14_Encoder()
        encoder.load_from_pth(osp.join(proj_root(), 'models', 'sl', 'best_encoder.pth'))
        device = get_device()
        encoder.to(device)
        encoder.eval()
        batch_size = data_loader.batch_size
        log_interval = max(len(data_loader) // 10, 1)
        with h5py.File(h5file, 'w') as h5_writer:
            h5_writer.create_dataset(name='class_list',
                                     data=data_loader.dataset.class_list,
                                     dtype='S100')
            h5_writer.create_dataset(name='embeddings',
                                     shape=(len(data_loader.dataset), encoder.frames_per_point, 2048),
                                     dtype='f')
            h5_writer.create_dataset(name='labels',
                                     shape=(len(data_loader.dataset), encoder.frames_per_point,
                                            len(data_loader.dataset.class_list)),
                                     dtype='f')
            with torch.no_grad():
                for i, (data, label) in enumerate(data_loader):
                    cur_batch_size = data.size(0)
                    data, label = data.to(device), label.to(device)
                    embeddings = encoder(data)
                    h5_writer['embeddings'][i * batch_size: i * batch_size +
                                            cur_batch_size, :, :] = embeddings.cpu().numpy()
                    h5_writer['labels'][i * batch_size: i * batch_size + cur_batch_size, :, :] = label.cpu().numpy()
                    if (i + 1) % log_interval == 0 or (i + 1) == len(data_loader):
                        logging.info(f"[{i + 1}/{len(data_loader)}] batches of al_{spl} set have been encoded.")
        with h5py.File(h5file, 'r+') as h5_rp:
            embeddings = np.array(h5_rp['embeddings'][:])
            pooled_embeddings = np.max(embeddings, axis=-2)
            h5_rp.create_dataset(name='dist_mtx', data=1. - cos_sim(pooled_embeddings, pooled_embeddings), dtype='f')
            logging.info(f'Distance matrix in {h5file} has been processed.')
        logging.info(f'{h5file} have been processed.')

    os.makedirs(osp.join(proj_root(), 'data', 'AL'), exist_ok=True)
    encode_spl(data_loader=generate_dataloader(config=config, div='al', spl='train'), spl='train')
    encode_spl(data_loader=generate_dataloader(config=config, div='al', spl='valid'), spl='valid')
    encode_spl(data_loader=generate_dataloader(config=config, div='al', spl='test'), spl='test')
    logging.info("AL sets have been encoded and stored.")
