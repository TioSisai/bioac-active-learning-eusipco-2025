import os
from omegaconf import DictConfig, OmegaConf
from os import path as osp
import logging
import torch
import platform
import numpy as np
import torchmetrics as tm


def proj_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_conf() -> DictConfig:
    conf_path = osp.join(proj_root(), 'configs', 'config.yaml')
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Config file not found: {conf_path}")
    return OmegaConf.load(conf_path)


def half_cpu_cores() -> int:
    return min(os.cpu_count() // 2, 8)


def download(url: str, save_filename: str):
    save_filename = osp.normpath(save_filename)
    if not osp.exists(save_filename):
        logging.info(f'Downloading {save_filename} from {url}...')
        os.makedirs(osp.dirname(save_filename), exist_ok=True)
        if platform.system() == 'Windows':
            os.system(f'curl -L -o {save_filename} {url}')
        elif platform.system() == 'Linux':
            os.system(f'wget -O {save_filename} {url}')
        elif platform.system() == 'Darwin':
            os.system(f'curl -L -o {save_filename} {url}')
        else:
            raise NotImplementedError(f'Unsupported platform: {platform.system()}')
        logging.info('Download complete.')
    else:
        logging.info(f'{osp.basename(save_filename)} already exists. Skipping download.')


def get_device(cuda_id: int = 0) -> torch.device:
    config = load_conf()
    try:
        device = torch.device(config.device)
        return device
    except AttributeError:
        pass
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if cuda_id == 0:
                return torch.device('cuda:0')
            elif cuda_id < num_gpus:
                return torch.device(f'cuda:{cuda_id}')
            else:
                logging.warning(f'cuda_id={cuda_id} is out of range. Using cpu instead.')
                return torch.device('cpu')
        else:
            logging.warning('No GPU detected. Using cpu instead.')
            return torch.device('cpu')
    elif platform.system() == 'Darwin':
        return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    else:
        raise NotImplementedError(f'Unsupported platform: {platform.system()}')


def valid_pred_label(pred: torch.Tensor, label: torch.Tensor):
    assert pred.ndim == 2 or pred.ndim == 3
    assert pred.shape == label.shape
    assert pred.device == label.device
    pred = torch.flatten(pred, start_dim=0, end_dim=1) if pred.ndim == 3 else pred
    label = torch.flatten(label, start_dim=0, end_dim=1) if label.ndim == 3 else label
    return pred, label


def get_optimal_threshold_f1(pred: torch.Tensor, label: torch.Tensor, deci_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    pred, label = valid_pred_label(pred, label)
    num_classes = pred.shape[-1]
    cur_f1_tensor = None
    cur_thres = 0.5
    for deci_place in range(1, deci_num + 1):
        interval = 0.1 ** deci_place
        if deci_place == 1:
            start = cur_thres - 4 * interval
            stop = cur_thres + 4 * interval
            num_intervals = 9
        else:
            start = cur_thres - 9 * interval
            stop = cur_thres + 9 * interval
            num_intervals = 19
        tmp_thres_list = np.linspace(start=start, stop=stop, num=num_intervals).tolist()
        if num_classes == 1:
            tmp_f1_tensor_list = [
                tm.classification.BinaryF1Score(
                    threshold=threshold).to(pred.device)(pred, label).unsqueeze(-1) for threshold in tmp_thres_list]
            tmp_f1_list = [tmp_f1_tensor_list[i].item() for i in range(len(tmp_f1_tensor_list))]
        else:
            tmp_f1_tensor_list = [
                tm.classification.MultilabelF1Score(
                    num_labels=num_classes,
                    threshold=threshold,
                    average='none').to(pred.device)(pred, label) for threshold in tmp_thres_list]
            tmp_f1_list = [tmp_f1_tensor_list[i].mean().item() for i in range(len(tmp_f1_tensor_list))]

        cur_thres = tmp_thres_list[np.argmax(tmp_f1_list).item()]
        cur_f1_tensor = tmp_f1_tensor_list[np.argmax(tmp_f1_list)].detach().cpu()
    return torch.Tensor([cur_thres]), cur_f1_tensor


def get_cmap(pred: torch.Tensor, label: torch.Tensor, thresholds: int = None) -> torch.Tensor:
    pred, label = valid_pred_label(pred, label)
    num_classes = pred.shape[-1]
    if num_classes == 1:
        map_func = tm.classification.BinaryAveragePrecision(thresholds=thresholds).to(pred.device)
    else:
        map_func = tm.classification.MultilabelAveragePrecision(
            num_labels=num_classes, average='none', thresholds=thresholds).to(pred.device)
    map_tensor = map_func(pred, label) if num_classes > 1 else map_func(pred, label).unsqueeze(-1)
    map_tensor = map_tensor.detach().cpu()
    map_tensor = torch.nan_to_num(map_tensor, nan=0.0)
    return map_tensor


def set_logger(log_path: str = None, log_level=logging.INFO, write_mode='a'):
    if log_path is not None:
        os.makedirs(osp.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_path,
            filemode=write_mode,
            encoding='utf-8')
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
