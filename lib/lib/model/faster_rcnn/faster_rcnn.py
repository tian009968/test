import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import torchvision.ops as ops
import torchvision.transforms as transforms
import torch.fft
from math import sqrt

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        inv_data, spe_data = self.filter_one(im_data)
        base_feat_inv = self.RCNN_base(inv_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_inv, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat_inv, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat_inv, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self, args):
        self._init_modules(args)
        self._init_weights()

    def create_extra_architecture(self, args):
        self._init_extra_modules(args)
    
    def supcon_two_side(self, feat_inv, feat_spe, temp):
        assert len(feat_inv.shape)==2
        assert len(feat_spe.shape)==2

        batch_size = feat_inv.shape[0]
        feat_total = torch.cat((feat_inv,feat_spe), dim=0)
        feat_total_dot = torch.div(torch.matmul(feat_total, feat_total.T), temp)

        mask_total = (torch.ones((batch_size * 2, batch_size * 2)) - torch.eye(batch_size * 2)).float().cuda()
        mask_total_ = (torch.ones((batch_size, batch_size)) - torch.eye(batch_size)).float().cuda()
        feat_total_dot = feat_total_dot * mask_total

        # for numerical stability
        logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
        feat_total_dot = feat_total_dot - logits_max.detach()
        feat_total_dot = feat_total_dot * mask_total

        feat_inv_dot = feat_total_dot[0:batch_size, 0:batch_size]
        feat_spe_dot = feat_total_dot[batch_size:batch_size*2, batch_size:batch_size*2]
        feat_cat_dot = torch.cat((feat_inv_dot,feat_spe_dot), dim=0)

        # compute log_prob
        feat_total_dot = torch.exp(feat_total_dot)
        feat_cat_dot = feat_cat_dot - torch.log(feat_total_dot.sum(1, keepdim=True)+1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_inv = (feat_cat_dot[0:batch_size] * mask_total_).sum(1) / batch_size
        mean_log_prob_spe = (feat_cat_dot[batch_size:batch_size*2] * mask_total_).sum(1) / batch_size
        mean_log_prob = torch.cat((mean_log_prob_inv, mean_log_prob_spe), dim=0)

        # loss
        loss = -mean_log_prob
        loss = loss.mean()

        return loss

    def supcon_one_side(self, feat_inv, feat_spe, temp):
        assert len(feat_inv.shape)==2
        assert len(feat_spe.shape)==2

        batch_size = feat_inv.shape[0]
        feat_total = torch.cat((feat_inv,feat_spe), dim=0)
        feat_total_dot = torch.div(torch.matmul(feat_total, feat_total.T), temp)

        mask_total = (torch.ones((batch_size * 2, batch_size * 2)) - torch.eye(batch_size * 2)).float().cuda()
        mask_total_ = (torch.ones((batch_size, batch_size)) - torch.eye(batch_size)).float().cuda()
        feat_total_dot = feat_total_dot * mask_total

        # for numerical stability
        logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
        feat_total_dot = feat_total_dot - logits_max.detach()
        feat_total_dot = feat_total_dot * mask_total

        feat_inv_dot = feat_total_dot[0:batch_size, 0:batch_size]

        # compute log_prob
        feat_total_dot = torch.exp(feat_total_dot)
        feat_inv_dot = feat_inv_dot - torch.log(feat_total_dot[0:batch_size].sum(1, keepdim=True)+1e-8)
        feat_inv_dot = feat_inv_dot * mask_total_

        # compute mean of log-likelihood over positive
        mean_log_prob_inv = feat_inv_dot.sum(1) / batch_size

        # loss
        loss = -mean_log_prob_inv
        loss = loss.mean()

        return loss
