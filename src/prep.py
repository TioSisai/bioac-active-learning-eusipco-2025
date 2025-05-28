import numpy as np
from omegaconf import DictConfig
from shutil import move, rmtree
import zipfile
import logging
from glob import glob
import pandas as pd
import soundfile as sf
import librosa
import os
from os import path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.utils import proj_root, download


def unzip_data(zip_file: str, save_dir: str):
    """
    Unzip the source zip file to the save_dir
    """
    logging.info('Unzipping the data...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if osp.exists(osp.join(save_dir, file)) or osp.exists(osp.join(save_dir, file.replace(' ', '_').replace('-', '_'))):
                continue
            else:
                zip_ref.extract(file, save_dir)
    logging.info('Data unzipped.')


def remove_redundant_files(data_home: str):
    """
    Remove the redundant files
    """
    try:
        os.remove(osp.join(data_home, 'Development_Set', 'Validation_Set', 'PB', 'BUK5_20180921_015906a.csv'))
        os.remove(osp.join(data_home, 'Development_Set', 'Validation_Set', 'PB', 'BUK5_20180921_015906a.wav'))
    except FileNotFoundError:
        pass
    logging.info('Redundant files removed.')


def rename_illegal_files(data_home: str):
    """
    Rename the illegal files which include special characters
    """
    for root, _, _ in os.walk(data_home):
        for wav_file in glob(osp.join(root, '*.wav')):
            if ' ' in wav_file or '-' in wav_file:
                new_wav_file = wav_file.replace(' ', '_').replace('-', '_')
                if osp.exists(new_wav_file):
                    os.remove(new_wav_file)
                    logging.debug(f'{osp.basename(new_wav_file)} has been removed.')
                os.rename(wav_file, new_wav_file)
                csv_file = wav_file.replace('.wav', '.csv')
                new_csv_file = csv_file.replace(' ', '_').replace('-', '_')
                if osp.exists(new_csv_file):
                    os.remove(new_csv_file)
                os.rename(csv_file, new_csv_file)
                logging.debug(f'{osp.basename(wav_file)} has been renamed to {osp.basename(new_wav_file)}.')
    logging.info('Illegal files renamed.')


def sort_df_columns(df: pd.DataFrame, div: str):
    assert div in ['sl', 'al'], 'div should be either "sl" or "al".'
    if div == 'sl':
        sort_list = ['Audiofilename', 'Starttime', 'Endtime', 'Meerkat_short_note', 'Meerkat_close_call', 'Meerkat_aggression_call', 'Meerkat_social_call', 'Hyena_groan_and_moo', 'Hyena_giggle', 'Hyena_squitter', 'Hyena_whoop', 'Hyena_rumble', 'Jackdaw_call', 'Black_throated_Blue_Warbler', 'Chipping_Sparrow', 'Common_Yellowthroat', 'Gray_cheeked_Thrush', 'Ovenbird', 'Rose_breasted_Grosbeak', 'Savannah_Sparrow', "Swainson's_Thrush", 'White_throated_Sparrow', 'American_Redstart', 'Bay_breasted_Warbler', 'Acrocephalus_scirpaceus_song', 'Circus_aeruginosus_call', 'Himantopus_himantopus_call',
                     'Tachybaptus_ruficollis_song', 'Alcedo_atthis_call', 'Alcedo_atthis_song', 'Motacilla_flava_song', 'Ixobrychus_minutus_song', 'Ixobrychus_minutus_cal', 'Ardea_purpurea_call', 'Fulica_atra_call', 'Acrocephalus_melanopogon_song', 'Acrocephalus_arundinaceus_song', 'Botaurus_stellaris_song', 'Botaurus_stellaris_call', 'Ciconia_ciconia_clapping', 'Gallinula_chloropus_call', 'Coracias_garrulus_call', 'Anas_strepera_song', 'Dendrocopos_minor_drumming', 'Charadrius_alexandrinus_call', 'Charadrius_alexandrinus_song', 'Porphyrio_porphyrio_call', 'Porphyrio_porphyrio_song']
    else:
        sort_list = ['Audiofilename', 'Starttime', 'Endtime', 'Mosquito', 'Red_Deer', 'Meerkat_alarm_call',
                     'Meerkat_move_call', 'Song_thrush_call', 'Blackbird_call', 'Pilot_whale_foraging_buzzes']
    return df.reindex(columns=sort_list)


def combine_all_sl_fold_labels(data_home: str):
    """
    Combine all the label files in SL folder into one file
    """
    os.makedirs(osp.join(data_home, 'SL'), exist_ok=True)
    if not osp.exists(osp.join(data_home, 'sl_label.csv')):
        merged_group = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'])
        for root, _, _ in os.walk(osp.join(data_home, 'Development_Set', 'Training_Set')):
            for file in glob(osp.join(root, '*.csv')):
                group = pd.read_csv(file)
                group.iloc[:, 0] = group.iloc[:, 0].str.replace('.csv', '.wav')
                group.iloc[:, 0] = group.iloc[:, 0].str.replace('[ -]', '_', regex=True)
                merged_group = pd.merge(merged_group, group, how='outer')
        merged_group = merged_group.fillna(0)
        merged_group = merged_group.map(lambda x: 1 if x == 'POS' else (0 if x == 'NEG' else (-1 if x == 'UNK' else x)))
        train_class_code_name_group = pd.read_csv(osp.join(data_home, 'DCASE2024_task5_training_set_classes.csv'))
        train_class_code_name_group['class_name'] = train_class_code_name_group['class_name'].str.replace('[ -]', '_',
                                                                                                          regex=True)
        train_class_code_name_dict = dict(zip(train_class_code_name_group['class_code'],
                                              train_class_code_name_group['class_name']))
        orig_columns = merged_group.columns
        new_columns = [train_class_code_name_dict[col] if col in train_class_code_name_dict else col for col in
                       orig_columns]
        merged_group = merged_group.rename(mapper=dict(zip(orig_columns, new_columns)), axis=1)
        merged_group.drop('Anas_platyrhynchos_song', axis=1, inplace=True)
        cur_columns = list(merged_group.columns)
        tachy_idx = cur_columns.index('Tachybaptus_ruficollis_song')
        cur_columns[tachy_idx + 1] = 'Tachybaptus_ruficollis_song_tmp'
        merged_group.columns = cur_columns
        tmp_1 = merged_group.iloc[:, tachy_idx].to_numpy()
        tmp_2 = merged_group.iloc[:, tachy_idx + 1].to_numpy()
        tmp = [1 if ((tmp_1[i] == 1) or (tmp_2[i] == 1)) else (-1 if ((tmp_1[i] == -1) or (tmp_2[i] == -1)) else 0) for
               i in range(len(tmp_1))]
        merged_group.iloc[:, tachy_idx] = pd.Series(tmp)
        merged_group = merged_group.drop('Tachybaptus_ruficollis_song_tmp', axis=1)
        for root, _, _ in os.walk(osp.join(data_home, 'Development_Set', 'Training_Set')):
            for csv_file in glob(osp.join(root, '*.csv')):
                wav_file = csv_file.replace('.csv', '.wav')
                move(wav_file, osp.join(data_home, 'SL', osp.basename(wav_file)))
                os.remove(csv_file)
        rmtree(osp.join(data_home, 'Development_Set', 'Training_Set'))
        merged_group = sort_df_columns(merged_group, 'sl')
        merged_group.to_csv(osp.join(data_home, 'sl_label.csv'), sep=',', index=False)
        logging.info('All SL fold labels combined.')
    else:
        logging.info('All SL fold labels have already been combined.')


def combine_all_al_fold_labels(data_home: str):
    """
    Combine all the label files in SL folder into one file
    """
    os.makedirs(osp.join(data_home, 'AL'), exist_ok=True)
    if not osp.exists(osp.join(data_home, 'al_label.csv')):
        valid_class_code_name_group = pd.read_csv(osp.join(data_home, 'DCASE2024_task5_validation_set_classes.csv'))
        valid_class_code_name_group['recording'] = valid_class_code_name_group['recording'].str.replace('[ -]', '_',
                                                                                                        regex=True)
        valid_class_code_name_group['class_name'] = valid_class_code_name_group['class_name'].str.replace('[ -]', '_',
                                                                                                          regex=True)
        file_class_name_dict = dict(zip(valid_class_code_name_group['recording'] + '.csv',
                                        valid_class_code_name_group['class_name']))
        merged_group = pd.DataFrame(columns=['Audiofilename', 'Starttime', 'Endtime'])
        for root, _, _ in os.walk(osp.join(data_home, 'Development_Set', 'Validation_Set')):
            for csv_file in glob(osp.join(root, '*.csv')):
                basename = osp.basename(csv_file)
                group = pd.read_csv(csv_file)
                group.iloc[:, 0] = basename.replace('.csv', '.wav')
                group.columns = ['Audiofilename', 'Starttime', 'Endtime', file_class_name_dict[basename]]
                merged_group = pd.merge(merged_group, group, how='outer')
        merged_group = merged_group.fillna(0)
        merged_group = merged_group.map(lambda x: 1 if x == 'POS' else (0 if x == 'NEG' else (-1 if x == 'UNK' else x)))
        merged_group = sort_df_columns(merged_group, 'al')
        merged_group.to_csv(osp.join(data_home, 'al_label.csv'), sep=',', index=False)
        for root, _, _ in os.walk(osp.join(data_home, 'Development_Set', 'Validation_Set')):
            for csv_file in glob(osp.join(root, '*.csv')):
                wav_file = csv_file.replace('.csv', '.wav')
                move(wav_file, osp.join(data_home, 'AL', osp.basename(wav_file)))
                os.remove(csv_file)
        rmtree(osp.join(data_home, 'Development_Set', 'Validation_Set'))
        logging.info('All AL fold labels combined.')
    else:
        logging.info('All AL fold labels have already been combined.')


def split_dataset(data_home: str, mode: str, train_ratio: float, valid_ratio: float = 0.):
    """
    Split the dataset into training and validation set
    """
    if mode == 'sl':
        csv_file = osp.join(data_home, 'sl_label.csv')
    elif mode == 'al':
        csv_file = osp.join(data_home, 'al_label.csv')
    else:
        logging.error('Mode in function "split_dataset" not supported.')
        raise ValueError('Mode not supported.')
    df = pd.read_csv(csv_file)
    groups = df.groupby('Audiofilename')
    split_file_time_dict = {}
    for name, group in groups:
        train_split_list = []
        valid_split_list = []
        total_time_each_class = np.array(
            [(group[group.iloc[:, i] == 1].iloc[:, 2] - group[group.iloc[:, i] == 1].iloc[:, 1]).sum() for i in
             range(3, group.shape[1])])
        train_time_gap_each_class = total_time_each_class * train_ratio
        valid_time_gap_each_class = total_time_each_class * valid_ratio + train_time_gap_each_class
        focused_class_idx = np.where(total_time_each_class != 0)[0]
        for col_idx in focused_class_idx:
            idx_df = group.iloc[:, col_idx + 3] == 1
            tmp_df = group[idx_df].iloc[:, 2] - group[idx_df].iloc[:, 1]
            tmp_cumsum_df = tmp_df.cumsum()
            train_split_list.append(find_the_time_slot(group, tmp_cumsum_df, train_time_gap_each_class[col_idx]))
            if valid_ratio != 0:
                valid_split_list.append(find_the_time_slot(group, tmp_cumsum_df, valid_time_gap_each_class[col_idx]))
        train_split_ndarray = np.array(train_split_list)
        if np.sum(train_split_ndarray[:, 0] > train_split_ndarray[:, 1]) > 0:
            logging.warning(f'Overlaps of Positive Segment are found (End(n-1) > '
                            f'Start(n)) in {name} in {mode} train set')
        latest_train_start_time = train_split_ndarray[:, 0].max()
        earliest_train_end_time = train_split_ndarray[:, 1].min()
        if latest_train_start_time > earliest_train_end_time:
            logging.warning(
                f'Train split time slots overlap. Audiofilename: {name}, train_split_ndarray: {train_split_ndarray}')
            split_file_time_dict[name] = {'train_split': float(np.sort(train_split_ndarray)[-2:].mean())}
            tmp_train_split = float(np.max(np.mean(train_split_ndarray, axis=-1)))
        else:
            tmp_train_split = (latest_train_start_time + earliest_train_end_time) / 2
        split_file_time_dict[name] = {'train_split': tmp_train_split}
        if len(valid_split_list) != 0:
            valid_split_ndarray = np.array(valid_split_list)
            if np.sum(valid_split_ndarray[:, 0] > valid_split_ndarray[:, 1]) > 0:
                logging.warning(f'Overlaps of Positive Segment are found (End(n-1) > '
                                f'Start(n)) in {name} in {mode} valid set')
            latest_valid_start_time = valid_split_ndarray[:, 0].max()
            earliest_valid_end_time = valid_split_ndarray[:, 1].min()
            if latest_valid_start_time > earliest_valid_end_time:
                logging.warning(
                    f'Valid split time slots overlap. Audiofilename: {name}, '
                    f'valid_split_ndarray: {valid_split_ndarray}')
                tmp_valid_split = float(np.max(np.mean(valid_split_ndarray, axis=-1)))
            else:
                tmp_valid_split = (latest_valid_start_time + earliest_valid_end_time) / 2
            if latest_valid_start_time < earliest_train_end_time:
                logging.warning(
                    f'Valid split time slots overlap with train split time slots. Audiofilename: {name}, '
                    f'valid_split_ndarray: {valid_split_ndarray}, train_split_ndarray: {train_split_ndarray}')
                raise ValueError('Valid split time slots overlap with train split time slots.')
            split_file_time_dict[name]['valid_split'] = tmp_valid_split
    return split_file_time_dict


def find_the_time_slot(father_df: pd.DataFrame, time_cumsum_df: pd.Series, time_gap: float):
    idx_df = time_cumsum_df < time_gap
    try:
        return [father_df.loc[idx_df[idx_df].index[-2], 'Endtime'],
                father_df.loc[idx_df[idx_df].index[-1], 'Starttime']]
    except IndexError:
        try:
            return [father_df.loc[idx_df.index[-2], 'Endtime'],
                    father_df.loc[idx_df.index[-1], 'Starttime']]
        except IndexError:
            logging.warning(father_df.iloc[0, 0] + ' has no enough number of positive time slots for splitting.')
            return [father_df.loc[idx_df.index[-1], 'Starttime'] + time_gap,
                    father_df.loc[idx_df.index[-1], 'Starttime'] + time_gap]


def store_splitting_wav_files(fold: str, split_file_time_dict: dict, target_sr: int):
    """
    Store the wav files
    """
    if not osp.exists(osp.join(fold, 'valid', list(split_file_time_dict.keys())[-1])):
        is_test_set_available = ('valid_split' in list(split_file_time_dict.values())[0].keys())
        os.makedirs(osp.join(fold, 'train'), exist_ok=True)
        os.makedirs(osp.join(fold, 'valid'), exist_ok=True)
        if is_test_set_available:
            os.makedirs(osp.join(fold, 'test'), exist_ok=True)
        total_num = len(split_file_time_dict)
        for i, (filename, time_dict) in enumerate(split_file_time_dict.items()):
            signal, sr = sf.read(osp.join(fold, filename))
            if len(signal.shape) == 2:
                signal = signal[:, 0]
            train_split_idx = round(time_dict['train_split'] * sr)
            len_sig = signal.shape[0]
            sf.write(osp.join(fold, 'train', filename),
                     data=librosa.resample(signal[0: train_split_idx], orig_sr=sr, target_sr=target_sr),
                     samplerate=target_sr)
            if not is_test_set_available:
                sf.write(osp.join(fold, 'valid', filename),
                         data=librosa.resample(signal[train_split_idx: len_sig], orig_sr=sr, target_sr=target_sr),
                         samplerate=target_sr)
            else:
                valid_split_idx = round(time_dict['valid_split'] * sr)
                sf.write(osp.join(fold, 'valid', filename),
                         data=librosa.resample(
                             signal[train_split_idx: valid_split_idx], orig_sr=sr, target_sr=target_sr),
                         samplerate=target_sr)
                sf.write(osp.join(fold, 'test', filename),
                         data=librosa.resample(signal[valid_split_idx: len_sig], orig_sr=sr, target_sr=target_sr),
                         samplerate=target_sr)
            logging.info(f'{i + 1}/{total_num} {filename} has been split and stored.')
            os.remove(osp.join(fold, filename))
        logging.info('All wav files have been split and stored.')
    else:
        logging.info('Wav files have already been split and stored.')


def regenerate_label_csv(orig_csv: str, fold: str, split_file_time_dict: dict):
    """
    Regenerate the label csv file using the split_file_time_dict
    """
    df = pd.read_csv(orig_csv)
    train_label_df = pd.DataFrame(columns=df.columns)
    valid_label_df = pd.DataFrame(columns=df.columns)
    test_label_df = pd.DataFrame(columns=df.columns)
    is_test_set_available = ('valid_split' in list(split_file_time_dict.values())[0].keys())
    if not osp.exists(osp.join(fold, 'train_label.csv')):
        groups = df.groupby('Audiofilename')
        for name, group in groups:
            tmp_train_df = group[group['Starttime'] <= split_file_time_dict[name]['train_split']]
            if tmp_train_df.iloc[-1, 2] > split_file_time_dict[name]['train_split']:
                tmp_train_df.iloc[-1, 2] = split_file_time_dict[name]['train_split']
            tmp_train_df.iloc[:, 1:3] = tmp_train_df.iloc[:, 1:3].clip(
                lower=0., upper=split_file_time_dict[name]['train_split'])
            train_label_df = pd.merge(train_label_df, tmp_train_df, how='outer')

            if not is_test_set_available:
                tmp_valid_df = group[(group['Endtime'] >= split_file_time_dict[name]['train_split'])]
                if tmp_valid_df.iloc[0, 1] < split_file_time_dict[name]['train_split']:
                    tmp_valid_df.iloc[0, 1] = split_file_time_dict[name]['train_split']
                tmp_valid_df.iloc[:, 1:3] -= split_file_time_dict[name]['train_split']
                tmp_valid_df.iloc[:, 1:3] = tmp_valid_df.iloc[:, 1:3].clip(lower=0.)
                valid_label_df = pd.merge(valid_label_df, tmp_valid_df, how='outer')
            else:
                tmp_valid_df = group[(group['Endtime'] >= split_file_time_dict[name]['train_split']) & (
                    group['Starttime'] <= split_file_time_dict[name]['valid_split'])]
                tmp_test_df = group[(group['Endtime'] >= split_file_time_dict[name]['valid_split'])]
                if tmp_valid_df.iloc[0, 1] < split_file_time_dict[name]['train_split']:
                    tmp_valid_df.iloc[0, 1] = split_file_time_dict[name]['train_split']
                if tmp_valid_df.iloc[-1, 2] > split_file_time_dict[name]['valid_split']:
                    tmp_valid_df.iloc[-1, 2] = split_file_time_dict[name]['valid_split']
                if tmp_test_df.iloc[0, 1] < split_file_time_dict[name]['valid_split']:
                    tmp_test_df.iloc[0, 1] = split_file_time_dict[name]['valid_split']
                tmp_valid_df.iloc[:, 1:3] -= split_file_time_dict[name]['train_split']
                tmp_test_df.iloc[:, 1:3] -= split_file_time_dict[name]['valid_split']
                tmp_valid_df.iloc[:, 1:3] = tmp_valid_df.iloc[:, 1:3].clip(
                    lower=0.,
                    upper=split_file_time_dict[name]['valid_split'] - split_file_time_dict[name]['train_split'])
                tmp_test_df.iloc[:, 1:3] = tmp_test_df.iloc[:, 1:3].clip(lower=0.)
                valid_label_df = pd.merge(valid_label_df, tmp_valid_df, how='outer')
                test_label_df = pd.merge(test_label_df, tmp_test_df, how='outer')
        if 'SL' in osp.basename(fold):
            div = 'sl'
        elif 'AL' in osp.basename(fold):
            div = 'al'
        else:
            raise ValueError(f'{osp.basename(fold)} Division not supported.')
        train_label_df = sort_df_columns(train_label_df, div)
        train_label_df.to_csv(osp.join(fold, 'train_label.csv'), sep=',', index=False)
        valid_label_df = sort_df_columns(valid_label_df, div)
        valid_label_df.to_csv(osp.join(fold, 'valid_label.csv'), sep=',', index=False)
        if is_test_set_available:
            test_label_df = sort_df_columns(test_label_df, div)
            test_label_df.to_csv(osp.join(fold, 'test_label.csv'), sep=',', index=False)
        logging.info('Label csv files have been regenerated.')
    else:
        logging.info('Label csv files have already been regenerated.')


def remove_unnecessary_files(data_home: str):
    """
    Remove the unnecessary files
    """
    rmtree(osp.join(data_home, '__MACOSX'))
    rmtree(osp.join(data_home, 'Development_Set'))
    os.remove(osp.join(data_home, 'DCASE2024_task5_training_set_classes.csv'))
    os.remove(osp.join(data_home, 'DCASE2024_task5_validation_set_classes.csv'))
    os.remove(osp.join(data_home, 'sl_label.csv'))
    os.remove(osp.join(data_home, 'al_label.csv'))
    os.remove(osp.join(data_home, 'Development_set.zip'))
    os.remove(osp.join(data_home, 'Development_set_annotations.zip'))
    os.remove(osp.join(data_home, 'source.zip'))
    logging.info('Unnecessary files removed.')


def judge_a_csv_file(csv_file: str):
    df = pd.read_csv(csv_file)
    time = [(df[df.iloc[:, class_idx_iter] == 1]['Endtime'] - df[df.iloc[:, class_idx_iter] == 1][
        'Starttime']).sum() for class_idx_iter in range(3, df.shape[1])]
    tmp_col = df.columns[3:]
    return pd.DataFrame(data={'class_name': tmp_col, 'time': time})


def skip_prep(data_home: str):
    if osp.exists(osp.join(data_home, 'AL', 'test_label.csv')):
        return True
    else:
        return False


def prep_data(config: DictConfig):
    data_home = osp.join(proj_root(), 'data')
    if skip_prep(data_home):
        logging.info(judge_a_csv_file(osp.join(data_home, 'SL', 'train_label.csv')))
        logging.info(judge_a_csv_file(osp.join(data_home, 'SL', 'valid_label.csv')))
        logging.info(judge_a_csv_file(osp.join(data_home, 'AL', 'train_label.csv')))
        logging.info(judge_a_csv_file(osp.join(data_home, 'AL', 'valid_label.csv')))
        logging.info(judge_a_csv_file(osp.join(data_home, 'AL', 'test_label.csv')))
        logging.info('Data has already been prepared.')
        return
    else:
        download(url='https://zenodo.org/api/records/10829604/files-archive',
                 save_filename=osp.join(data_home, 'source.zip'))
        unzip_data(zip_file=osp.join(data_home, 'source.zip'), save_dir=data_home)
        unzip_data(zip_file=osp.join(data_home, 'Development_set.zip'), save_dir=data_home)
        remove_redundant_files(data_home=data_home)
        rename_illegal_files(data_home=data_home)
        combine_all_sl_fold_labels(data_home=data_home)
        combine_all_al_fold_labels(data_home=data_home)
        sl_split_file_gap_dict = split_dataset(data_home=data_home, mode='sl', train_ratio=0.8, valid_ratio=0.)
        store_splitting_wav_files(fold=osp.join(data_home, 'SL'), split_file_time_dict=sl_split_file_gap_dict,
                                  target_sr=config.sl.data.sr)
        regenerate_label_csv(orig_csv=osp.join(data_home, 'sl_label.csv'),
                             fold=osp.join(data_home, 'SL'),
                             split_file_time_dict=sl_split_file_gap_dict)
        al_split_file_gap_dict = split_dataset(data_home=data_home, mode='al', train_ratio=0.7, valid_ratio=0.15)
        store_splitting_wav_files(fold=osp.join(data_home, 'AL'), split_file_time_dict=al_split_file_gap_dict,
                                  target_sr=config.sl.data.sr)
        regenerate_label_csv(orig_csv=osp.join(data_home, 'al_label.csv'),
                             fold=osp.join(data_home, 'AL'),
                             split_file_time_dict=al_split_file_gap_dict)
        remove_unnecessary_files(data_home=data_home)
