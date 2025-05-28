import os
from os import path as osp
import sys
import torch
from omegaconf import DictConfig
import torchmetrics as tm
import logging

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from src.data import generate_dataloader
from src.utils import get_device, get_cmap, get_optimal_threshold_f1
from src.models import PANNs


class SLEvaluator:
    def __init__(self, config: DictConfig, encoder_pth: str, classifier_pth: str, div: str = 'sl', spl: str = 'valid'):
        assert div in ['sl', 'al'], f'Invalid div: {div}, must be one of [sl, al]'
        assert spl in ['valid', 'test'], f'Invalid spl: {spl}, must be one of [valid, test]'
        self.config = config
        self.deci_num = config.sl.trainer.deci_num
        self.device = get_device()
        self.valid_loader = generate_dataloader(config=config, div=div, spl=spl)
        self.num_classes = self.valid_loader.dataset.num_classes
        self.model = PANNs(pretrained=True, num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        encoder_state_dict = torch.load(encoder_pth, map_location=self.device, weights_only=True)
        classifier_state_dict = torch.load(classifier_pth, map_location=self.device, weights_only=True)
        self.model.encoder.load_state_dict(encoder_state_dict)
        try:
            self.model.classifier.load_state_dict(classifier_state_dict)
        except RuntimeError:
            self.model.classifier.to('cpu')
            self.model.classifier.load_from_pth(classifier_pth)
            self.model.classifier.to(self.device)
        self.model.eval()

    def eval(self):
        with torch.no_grad():
            all_label = torch.empty([0, self.num_classes]).to(self.device)
            all_pred = torch.empty([0, self.num_classes]).to(self.device)
            for _, (sig, label) in enumerate(self.valid_loader):
                label = torch.flatten(label, start_dim=0, end_dim=1)
                sig, label = sig.to(self.device), label.to(self.device)
                all_label = torch.cat([all_label, label], dim=0)
                output = self.model(sig).flatten(start_dim=0, end_dim=1)
                all_pred = torch.cat([all_pred, torch.sigmoid(output)], dim=0)
            thres, f1 = get_optimal_threshold_f1(all_pred, all_label, self.deci_num)
            if all_pred.device.type == 'mps':
                all_pred = all_pred.to('cpu')
                all_label = all_label.to('cpu')
                thres = thres.to(device='cpu', dtype=all_pred.dtype)
            f1 = torch.mean(f1).item()
            thres = thres.item()
            all_label = all_label.int()
            precision = tm.classification.MultilabelPrecision(
                num_labels=self.num_classes, threshold=thres).to(all_pred.device)(all_pred, all_label)
            recall = tm.classification.MultilabelRecall(
                num_labels=self.num_classes, threshold=thres).to(all_pred.device)(all_pred, all_label)
            accuracy = tm.classification.MultilabelAccuracy(
                num_labels=self.num_classes, threshold=thres).to(all_pred.device)(all_pred, all_label)
            m_ap = torch.mean(get_cmap(all_pred, all_label)).item()

            logging.info(f'Precision: {precision.item() * 100:.2f}%, '
                         f'Recall: {recall.item() * 100:.2f}%, '
                         f'Accuracy: {accuracy.item() * 100:.2f}%, '
                         f'F1: {f1 * 100:.2f}%, '
                         f'mAP: {m_ap * 100:.2f}%')
