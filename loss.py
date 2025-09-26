import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np
import cv2
global CLASS_NUMBER


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, batch_dice=None, use_sigmoid=True, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def calc_loss(pred, target, bce_weight=0.5, loss_type='mse'):
    if loss_type == 'BCE':
        loss = nn.BCEWithLogitsLoss()(pred.squeeze(1), target)
    if loss_type == 'BCE_HEM':
        batchBase = False
        loss = nn.BCEWithLogitsLoss(reduction='none')(pred.squeeze(1), target)
        #loss = torch.mean(loss)i
        if batchBase:
            loss = torch.mean(loss, axis=(1,2))
            # Select the top 4 losses
            topk_losses, topk_indices = torch.topk(loss, 2)
            mask = torch.zeros_like(loss)
            mask[topk_indices] = 1
            # Apply the mask to losses to keep only the selected ones
            masked_loss = (loss * mask).sum() / mask.sum()  # Mean of selected losses
            
        else:
            loss_f = loss.flatten()
            topk_losses, topk_indices = torch.topk(loss_f, 500)
            mask = torch.zeros_like(loss_f)
            mask[topk_indices] = 1
            # Apply the mask to losses to keep only the selected ones
            masked_loss = (loss_f * mask).sum() / mask.sum()  # Mean of selected losses
        loss = masked_loss
    if loss_type == 'CE':
        loss = nn.CrossEntropyLoss()(pred, target[:].long())
    if loss_type == 'mse':
        loss = nn.MSELoss()(pred.squeeze(1), target)
    if loss_type == 'rmse':
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    if loss_type == 'dice_bce':
        loss_bce = nn.BCEWithLogitsLoss()(pred.squeeze(1), target)
        loss_dice = BinaryDiceLoss()(pred.squeeze(1), target)
        loss = 0.5 * loss_bce + 0.5 * loss_dice
    if loss_type == 'dice_bce_mc':
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

    return loss
