from pathlib import Path
import h5py
import torch
import pandas as pd
import numpy as np
import sys
import os
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from src.utils import proj_root


# 'class_list', 'dist_mtx', 'embeddings', 'labels' in the h5 file
# 'Mosquito', 'Red_Deer', 'Meerkat_alarm_call', 'Meerkat_move_call', 'Song_thrush_call', 'Blackbird_call', 'Pilot_whale_foraging_buzzes' in the class_list
# 'anno_idx', 'model', 'valid_cmap', 'valid_f1', 'thres', 'test_f1', 'test_cmap' in the pth file


def pool_labels(labels: np.ndarray):
    return np.max(labels, axis=1)


def cls_count(labels: np.ndarray):
    return np.sum(pool_labels(labels), axis=0)


project_root = Path(proj_root())

with h5py.File(project_root / "data" / "AL" / "train_set.h5", 'r') as f:
    classes: list[str] = [item.decode() for item in f['class_list']]
    train_labels: np.ndarray = np.array(f['labels'][:])

with h5py.File(project_root / "data" / "AL" / "valid_set.h5", 'r') as f:
    valid_labels: np.ndarray = np.array(f['labels'][:])

with h5py.File(project_root / "data" / "AL" / "test_set.h5", 'r') as f:
    test_labels: np.ndarray = np.array(f['labels'][:])

full_home = Path(project_root) / "models" / "al_train_full"
full_pths = list(full_home.iterdir())
full_valid_f1s = []
full_valid_cmaps = []
full_test_f1s = []
full_test_cmaps = []
for full_pth in full_pths:
    state_dict = torch.load(full_pth, map_location='cpu', weights_only=False)
    full_valid_f1s.append(state_dict['valid_f1'].mean())
    full_valid_cmaps.append(state_dict['valid_cmap'].mean())
    full_test_f1s.append(state_dict['test_f1'].mean())
    full_test_cmaps.append(state_dict['test_cmap'].mean())
print("Full test cmap mean: ", np.mean(full_test_cmaps))
print("Full test cmap std: ", np.std(full_test_cmaps))

results: list[dict] = []
total_cls_count = cls_count(train_labels)
total_infos: dict = {
    "total_num_samples": int(train_labels.shape[0])
}
for tmp_cls in classes:
    total_infos['Total_' + tmp_cls] = int(total_cls_count[classes.index(tmp_cls)])
total_infos['Total_Negative'] = int(total_infos['total_num_samples'] - np.sum(total_cls_count))
total_infos['full_valid_f1_mean'] = round(float(np.mean(full_valid_f1s) * 100), 2)
total_infos['full_valid_f1_std'] = round(float(np.std(full_valid_f1s) * 100), 2)
total_infos['full_valid_cmap_mean'] = round(float(np.mean(full_valid_cmaps) * 100), 2)
total_infos['full_valid_cmap_std'] = round(float(np.std(full_valid_cmaps) * 100), 2)
total_infos['full_test_f1_mean'] = round(float(np.mean(full_test_f1s) * 100), 2)
total_infos['full_test_f1_std'] = round(float(np.std(full_test_f1s) * 100), 2)
total_infos['full_test_cmap_mean'] = round(float(np.mean(full_test_cmaps) * 100), 2)
total_infos['full_test_cmap_std'] = round(float(np.std(full_test_cmaps) * 100), 2)


def extract_corr_result(al_pth: str | Path):
    if isinstance(al_pth, str):
        al_pth = Path(al_pth)
    tmp_info: dict = {
        "exp_id": int(al_pth.parents[1].name.split('_')[1]),
        "sampling_method": al_pth.parents[0].name.split('_')[0],
        "pos_init_num": int(al_pth.parents[0].name.split('_')[2]),
        "num_annotated_samples": int(al_pth.stem)
    }
    state_dict = torch.load(al_pth, map_location='cpu', weights_only=False)
    anno_idx = state_dict['anno_idx']
    anno_cls_count = cls_count(train_labels[anno_idx])
    for class_iter in classes:
        tmp_info[class_iter] = int(anno_cls_count[classes.index(class_iter)])
        tmp_info['Total_' + class_iter] = int(total_infos['Total_' + class_iter])

    tmp_info['Negative'] = int(tmp_info['num_annotated_samples'] - np.sum(anno_cls_count))
    tmp_info['Total_Negative'] = int(total_infos['Total_Negative'])
    tmp_info['valid_f1'] = round(float(state_dict['valid_f1'].mean() * 100), 2)
    tmp_info['valid_cmap'] = round(float(state_dict['valid_cmap'].mean() * 100), 2)
    tmp_info['test_f1'] = round(float(state_dict['test_f1'].mean() * 100), 2)
    tmp_info['test_cmap'] = round(float(state_dict['test_cmap'].mean() * 100), 2)
    tmp_info['full_valid_f1_mean'] = total_infos['full_valid_f1_mean']
    tmp_info['full_valid_f1_std'] = total_infos['full_valid_f1_std']
    tmp_info['full_valid_cmap_mean'] = total_infos['full_valid_cmap_mean']
    tmp_info['full_valid_cmap_std'] = total_infos['full_valid_cmap_std']
    tmp_info['full_test_f1_mean'] = total_infos['full_test_f1_mean']
    tmp_info['full_test_f1_std'] = total_infos['full_test_f1_std']
    tmp_info['full_test_cmap_mean'] = total_infos['full_test_cmap_mean']
    tmp_info['full_test_cmap_std'] = total_infos['full_test_cmap_std']
    results.append(tmp_info)


def process_one_al_round(to_processed_sub_exp_path: str | Path):
    to_processed_al_file_iters: list[Path] = list(to_processed_sub_exp_path.iterdir())
    for al_file_iter in to_processed_al_file_iters:
        extract_corr_result(al_file_iter)


def process_one_sub_exp(sub_exp_path: str | Path):
    to_processed_sub_exp_path = sub_exp_path
    process_one_al_round(to_processed_sub_exp_path)


def process_one_exp(exp_path: str | Path):
    sub_exp_paths: list[Path] = list(exp_path.iterdir())
    for sub_exp_path in sub_exp_paths:
        process_one_sub_exp(sub_exp_path)


def al_eval():
    model_home = Path(project_root) / "models" / "al_train"
    for exp_path in model_home.iterdir():
        process_one_exp(exp_path)
    df = pd.DataFrame(results)
    df = df.groupby(
        ['sampling_method', 'pos_init_num', 'num_annotated_samples']).agg(
            Mosquito_mean=('Mosquito', 'mean'), Mosquito_std=('Mosquito', 'std'), Total_Mosquito=('Total_Mosquito', 'first'), Red_Deer_mean=('Red_Deer', 'mean'), Red_Deer_std=('Red_Deer', 'std'), Total_Red_Deer=('Total_Red_Deer', 'first'), Meerkat_alarm_call_mean=('Meerkat_alarm_call', 'mean'), Meerkat_alarm_call_std=('Meerkat_alarm_call', 'std'), Total_Meerkat_alarm_call=('Total_Meerkat_alarm_call', 'first'), Meerkat_move_call_mean=('Meerkat_move_call', 'mean'), Meerkat_move_call_std=('Meerkat_move_call', 'std'), Total_Meerkat_move_call=('Total_Meerkat_move_call', 'first'), Song_thrush_call_mean=('Song_thrush_call', 'mean'), Song_thrush_call_std=('Song_thrush_call', 'std'), Total_Song_thrush_call=('Total_Song_thrush_call', 'first'), Blackbird_call_mean=('Blackbird_call', 'mean'), Blackbird_call_std=('Blackbird_call', 'std'), Total_Blackbird_call=('Total_Blackbird_call', 'first'), Pilot_whale_foraging_buzzes_mean=('Pilot_whale_foraging_buzzes', 'mean'), Pilot_whale_foraging_buzzes_std=('Pilot_whale_foraging_buzzes', 'std'), Total_Pilot_whale_foraging_buzzes=('Total_Pilot_whale_foraging_buzzes', 'first'), Negative_mean=('Negative', 'mean'), Negative_std=('Negative', 'std'), Total_Negative=('Total_Negative', 'first'), valid_f1_mean=('valid_f1', 'mean'), valid_f1_std=('valid_f1', 'std'), valid_cmap_mean=('valid_cmap', 'mean'), valid_cmap_std=('valid_cmap', 'std'), test_f1_mean=('test_f1', 'mean'), test_f1_std=('test_f1', 'std'), test_cmap_mean=('test_cmap', 'mean'), test_cmap_std=('test_cmap', 'std'), full_valid_f1_mean=('full_valid_f1_mean', 'first'), full_valid_f1_std=('full_valid_f1_std', 'first'), full_valid_cmap_mean=('full_valid_cmap_mean', 'first'), full_valid_cmap_std=('full_valid_cmap_std', 'first'), full_test_f1_mean=('full_test_f1_mean', 'first'), full_test_f1_std=('full_test_f1_std', 'first'), full_test_cmap_mean=('full_test_cmap_mean', 'first'), full_test_cmap_std=('full_test_cmap_std', 'first')).reset_index()
    df = df.round({'pos_init_num': 2, 'num_annotated_samples': 2, 'Mosquito_mean': 2, 'Mosquito_std': 2, 'Total_Mosquito': 2, 'Red_Deer_mean': 2, 'Red_Deer_std': 2, 'Total_Red_Deer': 2, 'Meerkat_alarm_call_mean': 2, 'Meerkat_alarm_call_std': 2, 'Total_Meerkat_alarm_call': 2, 'Meerkat_move_call_mean': 2, 'Meerkat_move_call_std': 2, 'Total_Meerkat_move_call': 2, 'Song_thrush_call_mean': 2, 'Song_thrush_call_std': 2, 'Total_Song_thrush_call': 2, 'Blackbird_call_mean': 2, 'Blackbird_call_std': 2, 'Total_Blackbird_call': 2, 'Pilot_whale_foraging_buzzes_mean': 2,
                  'Pilot_whale_foraging_buzzes_std': 2, 'Total_Pilot_whale_foraging_buzzes': 2, 'Negative_mean': 2, 'Negative_std': 2, 'Total_Negative': 2, 'valid_f1_mean': 2, 'valid_f1_std': 2, 'valid_cmap_mean': 2, 'valid_cmap_std': 2, 'test_f1_mean': 2, 'test_f1_std': 2, 'test_cmap_mean': 2, 'test_cmap_std': 2, 'full_valid_f1_mean': 2, 'full_valid_f1_std': 2, 'full_valid_cmap_mean': 2, 'full_valid_cmap_std': 2, 'full_test_f1_mean': 2, 'full_test_f1_std': 2, 'full_test_cmap_mean': 2, 'full_test_cmap_std': 2})
    df = df.sort_values(['pos_init_num', 'sampling_method', 'num_annotated_samples'], ascending=[True, False, True])
    os.makedirs(project_root / "results", exist_ok=True)
    df.to_csv(project_root / "results" / "al_results.csv", index=False)
    print("Results CSV generated successfully.")
