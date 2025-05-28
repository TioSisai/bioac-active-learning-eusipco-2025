from omegaconf import DictConfig
from torch import nn, optim
import torch
import torchmetrics as tm
import numpy as np
import logging
import os
from os import path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from src.utils import get_device, proj_root, get_optimal_threshold_f1, get_cmap
from src.data import generate_dataloader
from src.models import PANNs


class SLTrainer:
    def __init__(self, config: DictConfig, div: str = 'sl'):
        assert div in ['sl', 'al'], f'Invalid div: {div}, must be one of [sl, al]'
        self.config = config
        self.device = get_device()
        self.save_path = osp.join(proj_root(), 'models', div)
        os.makedirs(self.save_path, exist_ok=True)
        self.train_loader = generate_dataloader(config=config, div=div, spl='train', shuffle=True)
        self.valid_loader = generate_dataloader(config=config, div=div, spl='valid')
        self.num_classes = self.train_loader.dataset.num_classes
        self.model = PANNs(pretrained=True, num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.sl.trainer.optimizer.lr,
                                    weight_decay=config.sl.trainer.optimizer.weight_decay,
                                    betas=config.sl.trainer.optimizer.betas,
                                    eps=config.sl.trainer.optimizer.eps)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=config.sl.trainer.scheduler.mode,
            factor=config.sl.trainer.scheduler.factor,
            patience=config.sl.trainer.scheduler.patience,
            threshold=config.sl.trainer.scheduler.threshold,
            threshold_mode=config.sl.trainer.scheduler.threshold_mode,
            cooldown=config.sl.trainer.scheduler.cooldown)
        self.early_stop_lr = config.sl.trainer.early_stop_lr
        self.num_epochs = config.sl.trainer.num_epochs
        self.deci_num = config.sl.trainer.deci_num
        self.log_interval = max(len(self.train_loader) // 10, 1)
        self.cur_epoch = 1
        self.best_m_ap = 0.
        self.best_epoch = -1

    def step(self):
        if self.cur_epoch <= self.num_epochs:
            self.train_one_epoch()
            self.valid_once()
            self.cur_epoch += 1
        else:
            self.stop()

    def train_one_epoch(self):
        self.model.train()
        cur_acc_loss_list = []
        for batch_idx, (spec, label) in enumerate(self.train_loader):
            label = torch.flatten(label, start_dim=0, end_dim=1)  # [B * T, C]
            pos_weight = (label.shape[0] - torch.sum(label, dim=0)) / (torch.sum(label, dim=0) + 1e-5)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
            spec, label = spec.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(spec).flatten(start_dim=0, end_dim=1)
            loss = criterion(output, label)
            cur_acc_loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                logging.info(f'Train: '
                             f'Epoch: [ {self.cur_epoch} | {self.num_epochs} ], '
                             f'Batch: [ {batch_idx + 1} | {len(self.train_loader)} ], '
                             f'Loss: {np.mean(cur_acc_loss_list):.4f}')
                cur_acc_loss_list = []

    def valid_once(self):
        self.model.eval()
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
            if m_ap > self.best_m_ap:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(self.model.encoder.state_dict(), osp.join(
                    self.save_path, f'best_encoder.pth'))
                torch.save(self.model.classifier.state_dict(), osp.join(
                    self.save_path, f'best_classifier.pth'))
                self.best_m_ap = m_ap
                self.best_epoch = self.cur_epoch

            logging.info(f'Valid: '
                         f'Epoch: [ {self.cur_epoch} | {self.num_epochs} ] '
                         f'(Precision: {precision.item() * 100:.2f}%, '
                         f'Recall: {recall.item() * 100:.2f}%, '
                         f'Accuracy: {accuracy.item() * 100:.2f}%, '
                         f'F1: {f1 * 100:.2f}%, '
                         f'mAP: {m_ap * 100:.2f}%), '
                         f'(Best Epoch: {self.best_epoch}, mAP: {self.best_m_ap * 100:.2f}%)')
        self.scheduler.step(m_ap)
        if self.optimizer.param_groups[0]['lr'] < self.early_stop_lr:
            logging.info(f"Learning rate will be {self.optimizer.param_groups[0]['lr']} which is less than minimum "
                         f"learning rate {self.early_stop_lr}. Training is stopped.")
            self.stop()

    def stop(self):
        logging.info(f"Training is stopped. Best mAP: {self.best_m_ap * 100:.2f}%")
        sys.exit(0)
