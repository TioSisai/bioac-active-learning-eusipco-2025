import torch
import torch.nn as nn
import torch.nn.functional as func
from copy import deepcopy
import sys
import os
from os import path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.utils import proj_root, download


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class AvgMaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(AvgMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        return x1 + x2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel, pool_type: str):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel)
        elif pool_type == 'avg_max':
            self.pool = AvgMaxPool2d(kernel_size=pool_kernel)
        else:
            raise ValueError('Incorrect pool_type')
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = func.relu_(self.bn1(self.conv1(x)))
        x = func.relu_(self.bn2(self.conv2(x)))
        if x.shape[-2] % self.pool.kernel_size[0] == 0:
            x = self.pool(x)
        else:
            residual_size = x.shape[-2] % self.pool.kernel_size[0]
            supply_size = self.pool.kernel_size[0] - residual_size
            residual_x_mean = x[..., -residual_size:, :].mean(dim=-2, keepdim=True)
            x = torch.cat([x, residual_x_mean.expand(-1, -1, supply_size, -1)], dim=-2)
            x = self.pool(x)
        return x


class Cnn14_Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(Cnn14_Encoder, self).__init__()
        self.pretrain_pth = osp.join(proj_root(), 'models', 'pretrained', 'encoder.pth')
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64, pool_kernel=(2, 2), pool_type='avg')
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, pool_kernel=(2, 2), pool_type='avg')
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, pool_kernel=(2, 2), pool_type='avg')
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, pool_kernel=(2, 2), pool_type='avg')
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, pool_kernel=(2, 2), pool_type='avg')
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048, pool_kernel=(1, 1), pool_type='avg')
        self.frames_per_point = self.frames_per_point()

        if pretrained:
            if not osp.exists(self.pretrain_pth):
                self.prep_pretrained_models()
            self.load_from_pth(self.pretrain_pth)

    def prep_pretrained_models(self):
        pretrained_path = osp.join(proj_root(), 'models', 'pretrained')
        os.makedirs(pretrained_path, exist_ok=True)
        panns_pth = osp.join(pretrained_path, 'Cnn14_DecisionLevelMax_mAP=0.385.pth')
        if not osp.exists(panns_pth):
            url = 'https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
            download(url, panns_pth)
        state_dict = torch.load(panns_pth, map_location='cpu', weights_only=True)['model']
        tmp_encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('conv_block'):
                tmp_encoder_state_dict[key] = value
        torch.save(tmp_encoder_state_dict, self.pretrain_pth)

    def load_from_pth(self, pth_file: str):
        state_dict = torch.load(pth_file, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=True)

    def frames_per_point(self):
        cnt_list = []
        t_pool_seq = []
        for key, value in self.state_dict().items():
            if key.startswith('conv_block') and int(key.lstrip('conv_block')[0]) not in cnt_list:
                cnt_list.append(int(key.lstrip('conv_block')[0]))
                t_pool_seq.append(getattr(getattr(self, key[:11]), 'pool').kernel_size[0])
        multiplier = 1
        for pool_size in t_pool_seq:
            multiplier *= pool_size
        return multiplier

    def forward(self, x):
        """
        x: (batch_size, 1, time_frames, mel_bins)
        """
        x = self.conv_block1(x)
        x = func.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = func.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = func.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = func.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x)
        x = func.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x)
        x = func.dropout(x, p=0.2, training=self.training)  # [B, D, T, F]
        x = torch.mean(x, dim=3)  # [B, D, T]

        x1 = func.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = func.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = func.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)  # [B, T, D]

        return x  # [B, T, D]


class MLPClassifier(nn.Module):
    def __init__(self, output_dim: int, pretrained: bool = False):
        """
        Args:
            input_dim: the number of dimensions of the input features
            hidden_dim: the number of dimensions of the hidden layer
            output_dim: the number of categories
        """
        super(MLPClassifier, self).__init__()
        self.pretrain_pth = osp.join(proj_root(), 'models', 'pretrained', 'classifier.pth')
        self.fc1 = nn.Linear(2048, 1024)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_dim)
        if pretrained:
            if not osp.exists(self.pretrain_pth):
                self.prep_pretrained_models()
            self.load_from_pth(self.pretrain_pth)

    def prep_pretrained_models(self):
        pretrained_path = osp.join(proj_root(), 'models', 'pretrained')
        os.makedirs(pretrained_path, exist_ok=True)
        panns_pth = osp.join(pretrained_path, 'Cnn14_DecisionLevelMax_mAP=0.385.pth')
        if not osp.exists(panns_pth):
            url = 'https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
            download(url, panns_pth)
        state_dict = torch.load(panns_pth, map_location='cpu', weights_only=True)['model']
        tmp_classifier_state_dict = {
            'fc1.weight': state_dict['fc1.weight'], 'fc1.bias': state_dict['fc1.bias'],
            'fc2.weight': state_dict['fc_audioset.weight'], 'fc2.bias': state_dict['fc_audioset.bias']}
        torch.save(tmp_classifier_state_dict, self.pretrain_pth)

    def load_from_pth(self, pth_file: str):
        state_dict = torch.load(pth_file, map_location='cpu', weights_only=True)
        tmp_self_state_dict = deepcopy(self.state_dict())
        for key in state_dict.keys():
            if tmp_self_state_dict[key].shape == state_dict[key].shape:
                pass
            else:
                if len(tmp_self_state_dict[key].shape) == 2:
                    state_dict[key] = torch.nn.functional.interpolate(
                        state_dict[key].unsqueeze(0).unsqueeze(0), size=tmp_self_state_dict[key].shape,
                        mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
                elif len(tmp_self_state_dict[key].shape) == 1:
                    state_dict[key] = torch.nn.functional.interpolate(
                        state_dict[key].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                        size=[1, tmp_self_state_dict[key].shape[0]],
                        mode='bilinear', align_corners=True).squeeze(0).squeeze(0).squeeze(0)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class PANNs(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super(PANNs, self).__init__()
        self.encoder = Cnn14_Encoder(pretrained=pretrained)
        self.classifier = MLPClassifier(output_dim=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
