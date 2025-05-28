import sys
import logging
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.sl.train import SLTrainer
from src.utils import proj_root, load_conf, set_logger


if __name__ == '__main__':
    config = load_conf()
    set_logger(log_path=osp.join(proj_root(), 'logs', 'sl_train.log'), log_level=logging.INFO, write_mode='a')
    trainer = SLTrainer(config=config)
    for epoch_idx in range(config.sl.trainer.num_epochs):
        trainer.step()
