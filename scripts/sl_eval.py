import sys
import logging
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.sl.eval import SLEvaluator
from src.utils import proj_root, load_conf, set_logger


if __name__ == '__main__':
    config = load_conf()
    set_logger(log_path=osp.join(proj_root(), 'logs', 'sl_eval.log'), log_level=logging.INFO, write_mode='a')
    evaluator = SLEvaluator(
        config=config,
        encoder_pth=osp.join(proj_root(), 'models', 'sl', 'best_encoder.pth'),
        classifier_pth=osp.join(proj_root(), 'models', 'sl', 'best_classifier.pth'),
        div='sl',
        spl='valid')
    evaluator.eval()
