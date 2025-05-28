from torch import optim, nn
import torch
import numpy as np
import logging
import h5py
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from shutil import rmtree
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryF1Score, MultilabelF1Score
import os
from os import path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from src.utils import proj_root, get_device, get_optimal_threshold_f1, half_cpu_cores, get_cmap
from src.models import MLPClassifier
from src.data import encode_al_h5


class ALTrainer:
    def __init__(self,
                 config: DictConfig,
                 save_folder: str = None,
                 cuda_id: int = 0):
        self.device = get_device(cuda_id)
        save_folder = 'al' if save_folder is None else save_folder
        self.save_home = osp.join(proj_root(), 'models', osp.normpath(save_folder))
        os.makedirs(self.save_home, exist_ok=True)
        self.al_data_home = osp.join(proj_root(), 'data', 'AL')
        if not osp.exists(osp.join(self.al_data_home, 'test_set.h5')):
            encode_al_h5(config)
        self.batch_size = config.al.trainer.batch_size
        self.num_workers = half_cpu_cores()
        self.select_mode = config.al.trainer.select_mode
        self.init_pos_num_per_class = config.al.trainer.init_pos_num_per_class

        self.train_file = osp.join(self.al_data_home, 'train_set.h5')
        self.valid_file = osp.join(self.al_data_home, 'valid_set.h5')
        self.test_file = osp.join(self.al_data_home, 'test_set.h5')
        self.num_anno_samples_list = config.al.trainer.num_anno_samples_list

        with h5py.File(self.train_file, 'r') as train_reader:
            self.class_list = [item.decode() for item in list(train_reader['class_list'][:])]
            self.total_train_samples = train_reader['labels'].shape[0]
            assert self.num_anno_samples_list[-1] <= self.total_train_samples
        self.info_dict = {'anno_idx': np.full([self.total_train_samples], False, dtype=bool)}
        target_class = self.class_list
        self.target_class = [item for item in self.class_list if item in target_class]
        self.target_class_idx = [self.class_list.index(item) for item in self.target_class]
        self.trainer_config = config.al.trainer

        if osp.exists(self.save_path(check=True)):
            logging.info(f'Already finished {osp.dirname(self.save_path(check=True))}')
            self.skip_cur_exp = True
        else:
            self.skip_cur_exp = False
            tmp_home = osp.dirname(self.save_path(check=True))
            rmtree(tmp_home, ignore_errors=True)
            os.makedirs(tmp_home, exist_ok=True)
            logging.info('=' * 80)
            logging.info(f'Starting {tmp_home}')
            self.classifier = MLPClassifier(output_dim=len(self.target_class))
            """
                will be saved in {num_anno}_vf1_{valid_f1}.pth
                [model: state_dict, anno_idx: one_hot_vec(self.num),
                thres: float, valid_f1(C): np.ndarray, test_f1(C): np.ndarray,
                valid_cmap(C): np.ndarray, test_cmap(C): np.ndarray]
            """

    def classifier_inference(self, embeddings: np.ndarray | torch.Tensor):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        else:
            embeddings = embeddings.float()
        dataset = torch.utils.data.TensorDataset(embeddings)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False)
        self.classifier.eval()
        self.classifier.to(self.device)
        all_pred = []
        with torch.no_grad():
            for batch in dataloader:
                # note: TensorDataset.__getitem__ returns a tuple, so batch[0] is the input tensor
                inputs = batch[0].to(self.device)
                pred = torch.sigmoid(self.classifier(inputs))
                all_pred.append(pred.cpu())
        return torch.cat(all_pred, dim=0).numpy()

    def f1_thres(self,
                 pred: np.ndarray | torch.Tensor,
                 label: np.ndarray | torch.Tensor,
                 thres: float = None):
        assert pred.shape == label.shape
        assert pred.shape[-1] == len(self.target_class)
        num_classes = len(self.target_class)
        pred = torch.Tensor(pred).to(self.device)
        label = torch.Tensor(label).to(self.device)
        if pred.ndim == 3:
            pred = pred.flatten(0, 1)
            label = label.flatten(0, 1)
        elif pred.ndim != 2:
            raise ValueError(f"Invalid dim of pred and label: {pred.ndim}D")
        if thres is None:
            thres, f1 = get_optimal_threshold_f1(pred, label, deci_num=self.trainer_config.deci_num)
            thres = thres.item()
        else:
            if num_classes == 1:
                f1 = BinaryF1Score(threshold=thres).to(self.device)(pred.squeeze(), label.squeeze())
                f1 = torch.Tensor([f1.item()])
            else:
                f1 = MultilabelF1Score(
                    threshold=thres, num_labels=num_classes, average='none').to(self.device)(pred, label)
        return f1, thres

    def teach_classifier(self):
        self.classifier = MLPClassifier(output_dim=len(self.target_class),
                                        pretrained=True)  # to ensure same initialization
        self.classifier.to(self.device)
        best_classifier_state_dict = deepcopy(self.classifier.state_dict())
        optimizer = optim.Adam(self.classifier.parameters(),
                               lr=self.trainer_config.optimizer.lr,
                               weight_decay=self.trainer_config.optimizer.weight_decay,
                               betas=self.trainer_config.optimizer.betas,
                               eps=self.trainer_config.optimizer.eps)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.trainer_config.scheduler.mode,
                                                         factor=self.trainer_config.scheduler.factor,
                                                         patience=self.trainer_config.scheduler.patience,
                                                         threshold=self.trainer_config.scheduler.threshold,
                                                         threshold_mode=self.trainer_config.scheduler.threshold_mode,
                                                         cooldown=self.trainer_config.scheduler.cooldown)
        with h5py.File(self.train_file, 'r') as train_reader:
            train_embeddings = np.array(train_reader['embeddings'][:])[self.info_dict['anno_idx'], ...]
            train_labels = np.array(train_reader['labels'][:])[self.info_dict['anno_idx'], ...][
                ..., self.target_class_idx]
        with h5py.File(self.valid_file, 'r') as valid_reader:
            valid_embeddings = np.array(valid_reader['embeddings'][:])
            valid_labels = np.array(valid_reader['labels'][:])[..., self.target_class_idx]
        with h5py.File(self.test_file, 'r') as test_reader:
            test_embeddings = np.array(test_reader['embeddings'][:])
            test_labels = np.array(test_reader['labels'][:])[..., self.target_class_idx]
        train_set = TensorDataset(torch.from_numpy(train_embeddings).float(),
                                  torch.from_numpy(train_labels).float())
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  drop_last=False)
        best_valid_cmap = torch.zeros(len(self.target_class_idx))
        epoch_idx = None
        for epoch_idx in range(self.trainer_config.num_epochs):
            self.classifier.train()
            for _, (embedding, label) in enumerate(train_loader):
                embedding, label = embedding.to(self.device), label.to(self.device)
                label = torch.flatten(label, start_dim=0, end_dim=1)
                pos_weight = (label.shape[0] - torch.sum(label, dim=0)) / (torch.sum(label, dim=0) + 1e-5)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
                optimizer.zero_grad()
                output = self.classifier(embedding).flatten(start_dim=0, end_dim=1)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            self.classifier.eval()
            with torch.no_grad():
                valid_pred = self.classifier_inference(valid_embeddings)
                valid_cmap = get_cmap(torch.from_numpy(valid_pred), torch.from_numpy(valid_labels).int())
                if valid_cmap.mean().item() > best_valid_cmap.mean().item():
                    logging.debug(f'Using {int(np.sum(self.info_dict["anno_idx"]))} samples, '
                                  f'Epoch {epoch_idx + 1}, '
                                  f'valid cmap: {float(valid_cmap.mean().item()):.4f}')
                    best_classifier_state_dict = deepcopy(self.classifier.state_dict())
                    best_valid_cmap = valid_cmap
            scheduler.step(valid_cmap.mean().item())
            if optimizer.param_groups[0]['lr'] < self.trainer_config.early_stop_lr:
                logging.debug(f'Current lr {optimizer.param_groups[0]["lr"]} is less than '
                              f'early stop lr {self.trainer_config.early_stop_lr}, '
                              f'stop training at epoch {epoch_idx + 1}')
                break
        self.classifier.load_state_dict(best_classifier_state_dict, strict=True)
        self.info_dict['model'] = best_classifier_state_dict
        self.info_dict['valid_cmap'] = best_valid_cmap
        valid_pred = self.classifier_inference(valid_embeddings)
        self.info_dict['valid_f1'], self.info_dict['thres'] = self.f1_thres(pred=valid_pred, label=valid_labels)
        test_pred = self.classifier_inference(test_embeddings)
        self.info_dict['test_f1'], _ = self.f1_thres(pred=test_pred, label=test_labels, thres=self.info_dict['thres'])
        self.info_dict['test_cmap'] = get_cmap(torch.from_numpy(test_pred), torch.from_numpy(test_labels).int())
        logging.info(f'Using {int(np.sum(self.info_dict["anno_idx"]))} samples, '
                     f'valid f1: {float(torch.mean(self.info_dict["valid_f1"]).item()):.4f}, '
                     f'valid cmap: {float(torch.mean(self.info_dict['valid_cmap']).item()):.4f}, '
                     f'test f1: {float(torch.mean(self.info_dict["test_f1"]).item()):.4f}, '
                     f'test cmap: {float(torch.mean(self.info_dict['test_cmap']).item()):.4f}, '
                     f'experienced {epoch_idx + 1} epochs')
        logging.debug('-' * 80)

    def cur_class_sample_count(self):
        with h5py.File(self.train_file, 'r') as train_reader:
            cur_cls_label = np.max(np.array(train_reader['labels'][:])[
                                   self.info_dict['anno_idx'], ...][..., self.target_class_idx], axis=-2)
        cls_cnt_dict = {self.class_list[self.target_class_idx[i]]: int(
            np.sum(cur_cls_label[..., i])) for i in range(len(self.target_class_idx))}
        is_pos = np.max(cur_cls_label, axis=-1).astype(bool)
        neg_cnt = np.sum(~is_pos, axis=0).astype(int).tolist()
        cls_cnt_dict['Negative'] = neg_cnt
        logging.debug(f'{cls_cnt_dict}')

    def init_info_dict(self):
        if self.init_pos_num_per_class > 0:
            with h5py.File(self.train_file, 'r') as train_reader:
                for class_idx in self.target_class_idx:
                    cur_cls_label = np.array(train_reader['labels'][:])[..., class_idx]  # [B, T, C] -> [B, T]
                    cur_cls_pos_idx = np.where(np.max(cur_cls_label, axis=-1) == 1)[0]  # [B, T] -> [B] -> [pos_num]
                    if len(cur_cls_pos_idx) < self.init_pos_num_per_class:
                        logging.warning(f'Not enough positive samples for class '
                                        f'{self.class_list[class_idx]} in initial stage')
                    sel_cur_cls_pos_idx = self.rand_select(
                        to_select_num=min(self.init_pos_num_per_class, len(cur_cls_pos_idx)),
                        candidate_idx=cur_cls_pos_idx)
                    self.info_dict['anno_idx'][sel_cur_cls_pos_idx] = True
            num_anno = int(np.sum(self.info_dict['anno_idx']))
            if num_anno < self.num_anno_samples_list[0]:
                self.select(to_select_num=self.num_anno_samples_list[0] - num_anno)
        else:
            self.info_dict['anno_idx'][int(np.random.randint(0, self.total_train_samples))] = True
            self.select(to_select_num=self.num_anno_samples_list[0] - 1)

    def rand_select(self, to_select_num: int, candidate_idx: np.ndarray = None) -> np.ndarray:
        candidate_idx = np.where(~self.info_dict['anno_idx'])[0] if candidate_idx is None else candidate_idx
        return np.random.choice(candidate_idx, to_select_num, replace=False)

    def ft_select(self, to_select_num: int, base_idx: np.ndarray = None, candidate_idx: np.ndarray = None) -> np.ndarray:
        base_idx = np.where(self.info_dict['anno_idx'])[0] if base_idx is None else base_idx
        candidate_idx = np.where(~self.info_dict['anno_idx'])[0] if candidate_idx is None else candidate_idx
        assert len(candidate_idx) >= to_select_num, 'Not enough unannotated samples'
        with h5py.File(self.train_file, 'r') as train_reader:
            tmp_dist = np.array(train_reader['dist_mtx'][:])
        candidate_dist_vec = np.min(tmp_dist[np.ix_(candidate_idx, base_idx)], axis=-1)
        selected_idx = np.empty(to_select_num, dtype=int)
        for i in range(to_select_num):
            farthest_idx = candidate_idx[np.argmax(candidate_dist_vec)]
            selected_idx[i] = farthest_idx
            if i < to_select_num - 1:
                candidate_dist_vec = np.minimum(candidate_dist_vec, tmp_dist[candidate_idx, farthest_idx])
        return selected_idx

    def group_idx_by_mismatch(self, base_idx: np.ndarray = None, candidate_idx: np.ndarray = None) -> np.ndarray:
        base_idx = np.where(self.info_dict['anno_idx'])[0] if base_idx is None else base_idx
        candidate_idx = np.where(~self.info_dict['anno_idx'])[0] if candidate_idx is None else candidate_idx
        with h5py.File(self.train_file, 'r') as train_reader:
            tmp_dist = np.array(train_reader['dist_mtx'][:])
            base_labels = np.array(train_reader['labels'][:])[base_idx, ...][..., self.target_class_idx]
            candidate_embeddings = np.array(train_reader['embeddings'][:])[candidate_idx, ...]

        with h5py.File(self.valid_file, 'r') as valid_reader:
            valid_labels = np.array(valid_reader['labels'][:])[..., self.target_class_idx]
            valid_embeddings = np.array(valid_reader['embeddings'][:])
            valid_classifier_pred = self.classifier_inference(valid_embeddings)
            _, thres = self.f1_thres(pred=valid_classifier_pred, label=valid_labels)

        candidate_classifier_pred = self.classifier_inference(candidate_embeddings)
        candidate_classifier_pred = np.max(candidate_classifier_pred, axis=-2)
        candidate_classifier_pred[candidate_classifier_pred >= thres] = 1
        candidate_classifier_pred[candidate_classifier_pred < thres] = 0

        nearest_idx = np.argmin(tmp_dist[np.ix_(candidate_idx, base_idx)], axis=-1)
        candidate_cluster_pred = np.max(base_labels, axis=-2)[nearest_idx]

        candidate_mismatch_rates = np.sum(np.abs(candidate_classifier_pred - candidate_cluster_pred), axis=-1)

        sorted_idx = np.argsort(candidate_mismatch_rates)[::-1]
        sorted_candidate_idx = candidate_idx[sorted_idx]
        sorted_candidate_mismatch_rates = candidate_mismatch_rates[sorted_idx]
        unique_values, group_count = np.unique(sorted_candidate_mismatch_rates, return_counts=True)
        unique_values = unique_values[::-1]
        group_count = group_count[::-1]
        return sorted_candidate_idx, unique_values, np.cumsum(group_count)

    def mp_select(self, to_select_num: int, base_idx: np.ndarray = None, candidate_idx: np.ndarray = None) -> np.ndarray:
        selected_idx = []
        base_idx = np.where(self.info_dict['anno_idx'])[0] if base_idx is None else base_idx
        candidate_idx = np.where(~self.info_dict['anno_idx'])[0] if candidate_idx is None else candidate_idx
        sorted_candidate_idx, unique_values, cumsum_group_count = self.group_idx_by_mismatch(base_idx, candidate_idx)
        cumsum_group_count = np.insert(cumsum_group_count, 0, 0)
        for i in range(len(cumsum_group_count) - 1):
            if to_select_num <= 0:
                break
            group_size = cumsum_group_count[i + 1] - cumsum_group_count[i]
            group_samples = sorted_candidate_idx[cumsum_group_count[i]:cumsum_group_count[i + 1]]
            if group_size <= to_select_num:
                selected_idx.extend(group_samples)
                to_select_num -= group_size
                logging.debug(f'mismatch rate: {unique_values[i]}, num samples: {group_size}, use: {group_size}')
            else:
                selected_idx.extend(self.rand_select(to_select_num=to_select_num, candidate_idx=group_samples))
                logging.debug(f'mismatch rate: {unique_values[i]}, num samples: {group_size}, use: {to_select_num}')
                break
        return selected_idx

    def mfft_select(self, to_select_num: int, base_idx: np.ndarray = None, candidate_idx: np.ndarray = None) -> np.ndarray:
        selected_idx = []
        base_idx = np.where(self.info_dict['anno_idx'])[0] if base_idx is None else base_idx
        candidate_idx = np.where(~self.info_dict['anno_idx'])[0] if candidate_idx is None else candidate_idx
        sorted_candidate_idx, unique_values, cumsum_group_count = self.group_idx_by_mismatch(base_idx, candidate_idx)
        cumsum_group_count = np.insert(cumsum_group_count, 0, 0)
        for i in range(len(cumsum_group_count) - 1):
            if to_select_num <= 0:
                break
            group_size = cumsum_group_count[i + 1] - cumsum_group_count[i]
            group_samples = sorted_candidate_idx[cumsum_group_count[i]:cumsum_group_count[i + 1]]
            if group_size <= to_select_num:
                selected_idx.extend(group_samples)
                to_select_num -= group_size
                logging.debug(f'mismatch rate: {unique_values[i]}, num samples: {group_size}, use: {group_size}')
            else:
                base_idx = np.concatenate([base_idx, np.array(selected_idx).astype(int)],
                                          axis=0) if len(selected_idx) != 0 else base_idx
                selected_idx.extend(self.ft_select(
                    to_select_num=to_select_num,
                    base_idx=base_idx,
                    candidate_idx=group_samples))
                logging.debug(f'mismatch rate: {unique_values[i]}, num samples: {group_size}, use: {to_select_num}')
                break
        return selected_idx

    def select(self, to_select_num: int):
        num_anno = int(np.sum(self.info_dict['anno_idx']))
        is_init = True if num_anno < self.num_anno_samples_list[0] else False
        if self.select_mode == 'rand':
            to_anno_idx = self.rand_select(to_select_num=to_select_num)
        elif self.select_mode == 'ft':
            to_anno_idx = self.ft_select(to_select_num=to_select_num)
        elif self.select_mode == 'mp':
            to_anno_idx = self.rand_select(to_select_num=to_select_num) if is_init else self.mp_select(
                to_select_num=to_select_num)
        elif self.select_mode == 'mfft':
            to_anno_idx = self.ft_select(to_select_num=to_select_num) if is_init else self.mfft_select(
                to_select_num=to_select_num)
        else:
            raise ValueError(f'Invalid select_mode: {self.select_mode}')
        self.info_dict['anno_idx'][to_anno_idx] = True
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            self.cur_class_sample_count()

    def step(self):
        num_anno = int(np.sum(self.info_dict['anno_idx']))
        if num_anno == 0:
            self.init_info_dict()
            self.teach_classifier()
            torch.save(self.info_dict, self.save_path())
            return True
        else:
            to_anno_num = 0
            for num_iter in self.num_anno_samples_list:
                if num_iter > num_anno:
                    to_anno_num = num_iter - num_anno
                    break
            if to_anno_num == 0:
                logging.info(f'Finished all iterations in {osp.dirname(self.save_path(check=True))}')
                return False
            else:
                self.select(to_select_num=to_anno_num)
                self.teach_classifier()
                torch.save(self.info_dict, self.save_path())
                return True

    def save_path(self, check: bool = False):
        if check:
            return osp.join(
                self.save_home, f'{self.select_mode}_with_{self.init_pos_num_per_class}_pos_init',
                f'{int(self.num_anno_samples_list[-1])}.pth')
        else:
            return osp.join(
                self.save_home, f'{self.select_mode}_with_{self.init_pos_num_per_class}_pos_init',
                f'{int(np.sum(self.info_dict['anno_idx']))}.pth')
