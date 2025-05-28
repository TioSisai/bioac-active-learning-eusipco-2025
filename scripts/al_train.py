from os import path as osp
import logging
import sys
from itertools import product

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.al.train import ALTrainer
from src.utils import proj_root, load_conf, set_logger


def main():
    for exp_id in range(1, 6):
        set_logger(log_path=osp.join(proj_root(), 'logs', 'al_train',
                                     f'exp_{exp_id}.log'), log_level=logging.INFO, write_mode='a')
        init_pos_num_per_class_list = [0, 5]
        select_mode_list = ['rand', 'ft', 'mp', 'mfft']
        for init_pos_num_per_class, select_mode in product(init_pos_num_per_class_list, select_mode_list):
            config = load_conf()
            config.al.trainer.init_pos_num_per_class = init_pos_num_per_class
            config.al.trainer.select_mode = select_mode
            al_trainer = ALTrainer(config=config, save_folder=f'./al_train/exp_{exp_id}')
            if not al_trainer.skip_cur_exp:
                keep_on = True
                while keep_on:
                    keep_on = al_trainer.step()
            else:
                pass


if __name__ == '__main__':
    main()
