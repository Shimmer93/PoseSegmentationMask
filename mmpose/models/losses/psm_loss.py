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
        loss = torch.mean(loss)
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

        # reduced_weight = torch.sum(target_weight, dim=1, keepdim=True) / target_weight.size(1)

        loss = 0
        for i in range(point_logits.size(1)):
            # loss_i = self.criterion(point_logits[:, i, i].squeeze(), point_labels[:, i]) * reduced_weight
            # loss += torch.mean(loss_i)
            preds = point_logits[:,i,...]
            pos_mask = (point_labels == i+1)
            neg_mask = (point_labels != i+1)
            pos_preds = preds[pos_mask]
            pos_gt = torch.ones_like(pos_preds)
            neg_preds = preds[neg_mask]
            neg_gt = torch.zeros_like(neg_preds)
            
            loss_i_pos = self.criterion(pos_preds, pos_gt)
            loss_i_neg = self.criterion(neg_preds, neg_gt)
            # preds_ = torch.cat((pos_preds, neg_preds), dim=0)
            # gts_ = torch.cat((pos_gt, neg_gt), dim=0)
            # loss += self.criterion(preds_, gts_)
            
            if target_weight is not None and self.use_target_weight:
                weights = target_weight[:,i:i+1].repeat(point_logits.size(0)//target_weight.size(0), point_logits.size(-1))
                loss_i_pos = loss_i_pos * weights[pos_mask]
                loss_i_neg = loss_i_neg * weights[neg_mask]
                
            loss += (1 - self.neg_weight) * torch.mean(loss_i_pos) + \
                self.neg_weight * torch.mean(loss_i_neg)
            
        # loss += F.mse_loss(pred_hmaps, gt_hmaps)

        return loss * self.loss_weight

@MODELS.register_module()
class JointSegTrainLoss2(nn.Module):
    def __init__(self, loss_weight=1.0, neg_weight=0.8, use_target_weight=False):
        super(JointSegTrainLoss2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_weight = loss_weight
        self.neg_weight = neg_weight
        self.use_target_weight = use_target_weight

    def forward(self, pred_masks, gt_masks, target_weight=None):
        reduced_weight = (torch.sum(target_weight, dim=1) / target_weight.size(1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = self.criterion(pred_masks, gt_masks) * reduced_weight
        loss = torch.mean(loss)
        return loss * self.loss_weight
    
@MODELS.register_module()
class JointSegTrainLoss3(nn.Module):
    def __init__(self, loss_weight=1.0, use_target_weight=False):
        super(JointSegTrainLoss3, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.loss_weight = loss_weight
        self.use_target_weight = use_target_weight

    def forward(self, point_logits_list, point_labels_list, target_weight=None):
        loss = 0
        for j, (point_logits, point_labels) in enumerate(zip(point_logits_list, point_labels_list)):
            # reduced_weight = torch.sum(target_weight, dim=1, keepdim=True) / target_weight.size(1)
            # print(point_logits[:, j][point_labels == 0].shape, point_labels[point_labels == 0].shape)
            assert point_logits[:, j].shape == point_labels.shape, f'{point_logits[:, j].shape} != {point_labels.shape}'
            # print(point_logits[:, j].shape, point_labels.shape)
            loss += self.criterion(point_logits[:, j], point_labels)
            # loss += 0.2 * torch.mean(self.criterion(point_logits[:, j][point_labels == 1], point_labels[point_labels == 1]))
            # loss += 0.8 * torch.mean(self.criterion(point_logits[:, j][point_labels == 0], point_labels[point_labels == 0]))
        loss = torch.mean(loss)
        return loss * self.loss_weight
    
def charbonnier(x, alpha=0.25, epsilon=1.e-9):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)


def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.sum(s_loss)/b


def photometric_loss(warped, frm0):
    h, w = warped.shape[2:]
    frm0 = F.interpolate(frm0, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(warped - frm0)
    p_loss = torch.sum(p_loss, dim=1)/3
    return torch.sum(p_loss)/frm0.size(0)


def unsup_loss(pred_flows, warped_imgs, frm0, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
    bce = 0
    smooth = 0
    for i in range(len(weights)):
        bce += weights[i] * photometric_loss(warped_imgs[i], frm0)
        smooth += weights[i] * smoothness_loss(pred_flows[i])

    loss = bce + smooth
    return loss, bce, smooth

@MODELS.register_module()
class UnsupFlowLoss(nn.Module):
    def __init__(self, gamma, loss_weight=1.0):
        super(UnsupFlowLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred_flows, frm0, frm1):
        warped_frms = [self._stn(flow, frm1) for flow in pred_flows]

        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = unsup_loss(pred_flows, warped_frms, frm0, weights)[0]

        return loss
    
    def _generate_grid(self, B, H, W, device):
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        grid = grid.to(device)
        return grid
    
    def _stn(self, flow, frm):
        b, _, h, w = flow.shape
        frm = F.interpolate(frm, size=(h, w), mode='bilinear', align_corners=True)
        flow = flow.permute(0, 2, 3, 1)

        grid = flow + self._generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1
        warped_frm = F.grid_sample(frm, grid, align_corners=True)

        return warped_frm
    
def EPE(flow_pred, flow_true, mask=None, real=False):

    if real:
        batch_size, _, h, w = flow_true.shape
        flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=False)
    else:
        batch_size, _, h, w = flow_pred.shape
        flow_true = F.interpolate(flow_true, (h, w), mode='area')
    if mask != None:
        return torch.norm(flow_pred - flow_true, 2, 1, keepdim=True)[mask].mean()
    else:
        return torch.norm(flow_pred - flow_true, 2, 1).mean()


def EPE_all(flows_pred, flow_true, mask=None, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):

    loss = 0

    for i in range(len(weights)):
        loss += weights[i] * EPE(flows_pred[i], flow_true, mask, real=False)

    return loss

class EPELoss(nn.Module):
    def __init__(self, gamma):
        super(EPELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred_flows, flow, mask):
        n_iters = len(pred_flows)
        weights = [self.gamma ** (n_iters - i - 1) for i in range(n_iters)]
        loss = EPE_all(pred_flows, flow, mask, weights)
        return loss
    
@MODELS.register_module()
class KeypointFlowLoss(nn.Module):
    def __init__(self, loss_weight=1.0, gamma=0.8):
        super(KeypointFlowLoss, self).__init__()
        self.epe_loss = EPELoss(gamma)
        self.loss_weight = loss_weight

    def forward(self, pred_flows, kps, target_weight=None):
        gt_flow = torch.zeros_like(pred_flows[-1], device=pred_flows[-1].device)
        B, J, H, W = gt_flow.size()
        
        kps0 = kps[:, 0, ...]
        kps1 = kps[:, 1, ...]
        disps = kps1 - kps0
        for i in range(B):
            for kp0, kp1, disp in zip(kps0[i], kps1[i], disps[i]):
                if kp0[0] < 0 or kp0[1] < 0 or kp1[0] < 0 or kp1[1] < 0:
                    continue
                if kp0[0] >= W or kp0[1] >= H or kp1[0] >= W or kp1[1] >= H:
                    continue
                gt_flow[i, :, int(kp0[1]), int(kp0[0])] = disp

        gt_mask = (gt_flow.norm(dim=1, keepdim=True) > 0)
        loss = self.epe_loss(pred_flows, gt_flow, gt_mask)
        return loss * self.loss_weight

@MODELS.register_module()
class FlowLoss(nn.Module):
    def __init__(self, loss_weight=1.0, gamma=0.8):
        super(FlowLoss, self).__init__()
        self.sup_loss = KeypointFlowLoss(loss_weight, gamma)
        self.unsup_loss = UnsupFlowLoss(gamma, loss_weight)
        self.loss_weight = loss_weight

    def forward(self, pred_flows, kps, frm0, frm1):
        sup_loss = self.sup_loss(pred_flows, kps)
        unsup_loss = self.unsup_loss(pred_flows, frm0, frm1)
        return sup_loss + unsup_loss