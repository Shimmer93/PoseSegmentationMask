import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS

@MODELS.register_module()
class BodySegTrainLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_target_weight=False):
        super(BodySegTrainLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(self, point_logits, point_labels, target_weight=None):
        # point_logits: B 1 P
        # point_labels: B P
        # target_weight: B C
        reduced_weight = torch.sum(target_weight, dim=1, keepdim=True) / target_weight.size(1)
        loss = self.criterion(point_logits.squeeze(1), point_labels) * reduced_weight
        loss = torch.mean(loss, dim=1)
        return loss * self.loss_weight

@MODELS.register_module()
class JointSegTrainLoss(nn.Module):
    def __init__(self, loss_weight=1.0, neg_weight=0.8, use_target_weight=False):
        super(JointSegTrainLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_weight = loss_weight
        self.neg_weight = neg_weight
        self.use_target_weight = use_target_weight

    def forward(self, point_logits, point_labels, target_weight=None):
        # point_logits: B C P
        # point_labels: B P
        # target_weight: B C
        # print(point_logits.size(), point_labels.size(), target_weight.size())
        # gt_hmaps = gt_hmaps / gt_hmaps.max(dim=(2,3), keepdim=True)[0]
        # print(point_logits.shape, point_labels.shape, target_weight.shape)

        reduced_weight = torch.sum(target_weight, dim=1, keepdim=True) / target_weight.size(1)

        loss = 0
        for i in range(point_logits.size(2)):
            loss_i = self.criterion(point_logits[:, i, i].squeeze(), point_labels[:, i]) * reduced_weight
            loss += torch.mean(loss_i, dim=1)
            # preds = point_logits[:,i,...]
            # pos_mask = (point_labels == i+1)
            # neg_mask = (point_labels != i+1)
            # pos_preds = preds[pos_mask]
            # pos_gt = torch.ones_like(pos_preds)
            # neg_preds = preds[neg_mask]
            # neg_gt = torch.zeros_like(neg_preds)
            
            # loss_i_pos = self.criterion(pos_preds, pos_gt)
            # loss_i_neg = self.criterion(neg_preds, neg_gt)
            # # preds_ = torch.cat((pos_preds, neg_preds), dim=0)
            # # gts_ = torch.cat((pos_gt, neg_gt), dim=0)
            # # loss += self.criterion(preds_, gts_)
            
            # if target_weight is not None and self.use_target_weight:
            #     weights = target_weight[:,i:i+1].repeat(1, point_logits.size(-1))
            #     loss_i_pos = loss_i_pos * weights[pos_mask]
            #     loss_i_neg = loss_i_neg * weights[neg_mask]
                
            # loss += (1 - self.neg_weight) * torch.mean(loss_i_pos) + \
            #     self.neg_weight * torch.mean(loss_i_neg)
            
        # loss += F.mse_loss(pred_hmaps, gt_hmaps)

        return loss * self.loss_weight