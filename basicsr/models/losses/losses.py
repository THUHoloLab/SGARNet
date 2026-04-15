# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.models as models
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG feature extraction.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        pretrained (bool): Whether to use pretrained weights. Default: True.
        layer_weights (dict): Weight for each feature layer.
            Example: {'relu1_1': 0.1, 'relu2_1': 0.1, ...}.
        criterion (str): Criterion for feature loss. Options: 'mse' | 'l1'. Default: 'mse'.
        norm_img (bool): Whether to normalize images to ImageNet stats. Default: True.
    """

    def __init__(self, loss_weight=1.0, pretrained=True, layer_weights=None,
                 criterion='mse', norm_img=True):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.norm_img = norm_img
        # ImageNet mean and std (RGB format)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Load VGG19 (pretrained on ImageNet)
        vgg = models.vgg19(pretrained=pretrained).features.eval()
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        # Default layer weights if not provided
        if layer_weights is None:
            layer_weights = {
                'relu1_1': 1.0,
                'relu2_1': 1.0,
                'relu3_1': 1.0,
                'relu4_1': 1.0,
                'relu5_1': 1.0
            }
        self.layer_weights = layer_weights
        self.criterion = criterion

        # Register hooks to extract features from specified layers
        self.feature_layers = list(layer_weights.keys())
        self.features = {}

        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output

            return hook

        # VGG19 layer mapping (convolutional layers followed by ReLU)
        layer_mapping = {
            'conv1_1': 0, 'relu1_1': 1,
            'conv2_1': 5, 'relu2_1': 6,
            'conv3_1': 10, 'relu3_1': 11,
            'conv4_1': 19, 'relu4_1': 20,
            'conv5_1': 28, 'relu5_1': 29
        }

        for layer_name in self.feature_layers:
            layer_idx = layer_mapping[layer_name]
            self.vgg[layer_idx].register_forward_hook(hook_fn(layer_name))

    def forward(self, pred, target):
        # Normalize images to ImageNet stats (if enabled)
        if self.norm_img:
            pred = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
            target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        # Extract features (hooks will populate self.features)
        self.features = {}
        self.vgg(pred)
        pred_feats = {k: v for k, v in self.features.items()}

        self.features = {}
        with torch.no_grad():
            self.vgg(target)
        target_feats = {k: v for k, v in self.features.items()}

        # Calculate feature loss
        loss_percep = 0.0
        for layer_name, weight in self.layer_weights.items():
            pred_feat = pred_feats[layer_name]
            target_feat = target_feats[layer_name]
            if self.criterion == 'mse':
                loss_percep += weight * F.mse_loss(pred_feat, target_feat)
            elif self.criterion == 'l1':
                loss_percep += weight * F.l1_loss(pred_feat, target_feat)
            else:
                raise ValueError(f"Unsupported criterion: {self.criterion}")

        return self.loss_weight * loss_percep, None