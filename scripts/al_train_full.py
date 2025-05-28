from os import path as osp
import os
import shutil
import logging
import sys


sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.utils import proj_root
from src.al.train import ALTrainer
from src.utils import proj_root, load_conf, set_logger


if __name__ == '__main__':
    config = load_conf()
    set_logger(log_path=osp.join(proj_root(), 'logs', f'al_train_full.log'), log_level=logging.INFO, write_mode='a')
    for exp_id in range(1, 6):
        config.al.trainer.init_pos_num_per_class = 0
        # no matter what select mode is, it will select all samples in the train set, but rand is the fastest without any unnecessary computation
        config.al.trainer.select_mode = 'rand'
        config.al.trainer.num_anno_samples_list = [21414]  # 21414 is the total number of samples in the train set
        al_trainer = ALTrainer(config=config, save_folder=f'al_train_full_{exp_id}', cuda_id=0)
        if not al_trainer.skip_cur_exp:
            keep_on = True
            while keep_on:
                keep_on = al_trainer.step()
        else:
            pass
        os.makedirs(osp.join(proj_root(), 'models', 'al_train_full'), exist_ok=True)
        shutil.move(osp.join(proj_root(), 'models', f'al_train_full_{exp_id}', 'rand_with_0_pos_init', '21414.pth'),
                    osp.join(proj_root(), 'models', 'al_train_full', f'exp_{exp_id}.pth'))
        shutil.rmtree(osp.join(proj_root(), 'models', f'al_train_full_{exp_id}'))
    logging.info('All experiments are done.')
