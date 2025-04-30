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

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

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

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, args):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        contrast_one = []
        contrast_two = []
        scaling_factor = [1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.125, 0.0625]

        # feed image data to base model to obtain base feature map
        if not args.setting[0] == 4:
            inv_data, spe_data = self.filter_one(im_data)
            if (args.setting[1] == 1 or args.setting[2] == 1) and (args.contrastive_type == 3 or args.contrastive_type == 4 or args.contrastive_type == 5):
                inv_inv_data, inv_spe_data = self.filter_one(inv_data)
                spe_inv_data, spe_spe_data = self.filter_one(spe_data)

            if args.setting[1] == 1:
                if args.contrastive_type == 0 or args.contrastive_type == 1:
                    base_feat_inv_img = self.RCNN_base[:args.img_position](inv_data)
                    base_feat_spe_img = self.RCNN_base[:args.img_position](spe_data)
                    con_feat_inv_img, con_feat_spe_img = self.mlp_one(base_feat_inv_img, base_feat_spe_img)
                    contrast_one = [con_feat_inv_img, con_feat_spe_img]
                elif args.contrastive_type == 2:
                    base_feat_inv_img = self.RCNN_base[:args.img_position](inv_data)
                    base_feat_spe_img = self.RCNN_base[:args.img_position](spe_data)
                    con_feat_inv_img, con_feat_spe_img = self.mlp_one(base_feat_inv_img, base_feat_spe_img)
                    with torch.no_grad():
                        mom_base_feat_inv_img = self.momentum_RCNN_base[:args.img_position](inv_data)
                        mom_base_feat_spe_img = self.momentum_RCNN_base[:args.img_position](spe_data)
                        mom_con_feat_inv_img, mom_con_feat_spe_img = self.momentum_mlp_one(mom_base_feat_inv_img, mom_base_feat_spe_img)
                    contrast_one = [con_feat_inv_img, con_feat_spe_img, mom_con_feat_inv_img, mom_con_feat_spe_img]
                elif args.contrastive_type == 3 or args.contrastive_type == 4 or args.contrastive_type == 5:
                    base_feat_inv_img = self.RCNN_base[:args.img_position](inv_data)
                    base_feat_spe_img = self.RCNN_base[:args.img_position](spe_data)
                    base_feat_inv_inv_img = self.RCNN_base[:args.img_position](inv_inv_data)
                    base_feat_inv_spe_img = self.RCNN_base[:args.img_position](inv_spe_data)
                    base_feat_spe_inv_img = self.RCNN_base[:args.img_position](spe_inv_data)
                    base_feat_spe_spe_img = self.RCNN_base[:args.img_position](spe_spe_data)
                    con_feat_inv_img, con_feat_spe_img = self.mlp_one(base_feat_inv_img, base_feat_spe_img)
                    con_feat_inv_inv_img, con_feat_inv_spe_img = self.mlp_one(base_feat_inv_inv_img, base_feat_inv_spe_img)
                    con_feat_spe_inv_img, con_feat_spe_spe_img = self.mlp_one(base_feat_spe_inv_img, base_feat_spe_spe_img)
                    contrast_one = [con_feat_inv_img, con_feat_inv_inv_img, con_feat_inv_spe_img, con_feat_spe_img, con_feat_spe_spe_img, con_feat_spe_inv_img]
                else:
                    raise ValueError('wrong contrastive type')
            else:
                contrast_one = []

            if args.setting[2] == 1:
                if args.contrastive_type == 0 or args.contrastive_type == 1:
                    base_feat_inv_ins = self.RCNN_base[:args.ins_position](inv_data)
                    base_feat_spe_ins = self.RCNN_base[:args.ins_position](spe_data)
                    base_feat_inv_ins_ = ops.roi_align(base_feat_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_spe_ins_ = ops.roi_align(base_feat_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    con_feat_inv_ins, con_feat_spe_ins = self.mlp_two(base_feat_inv_ins_, base_feat_spe_ins_)
                    contrast_two = [con_feat_inv_ins, con_feat_spe_ins]
                elif args.contrastive_type == 2:
                    base_feat_inv_ins = self.RCNN_base[:args.ins_position](inv_data)
                    base_feat_spe_ins = self.RCNN_base[:args.ins_position](spe_data)
                    base_feat_inv_ins_ = ops.roi_align(base_feat_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_spe_ins_ = ops.roi_align(base_feat_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    con_feat_inv_ins, con_feat_spe_ins = self.mlp_two(base_feat_inv_ins_, base_feat_spe_ins_)
                    with torch.no_grad():
                        mom_base_feat_inv_ins = self.momentum_RCNN_base[:args.ins_position](inv_data)
                        mom_base_feat_spe_ins = self.momentum_RCNN_base[:args.ins_position](spe_data)
                        mom_base_feat_inv_ins_ = ops.roi_align(mom_base_feat_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                        mom_base_feat_spe_ins_ = ops.roi_align(mom_base_feat_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                        mom_con_feat_inv_ins, mom_con_feat_spe_ins = self.momentum_mlp_two(mom_base_feat_inv_ins_, mom_base_feat_spe_ins_)
                    contrast_two = [con_feat_inv_ins, con_feat_spe_ins, mom_con_feat_inv_ins, mom_con_feat_spe_ins]
                elif args.contrastive_type == 3 or args.contrastive_type == 4 or args.contrastive_type == 5:
                    base_feat_inv_ins = self.RCNN_base[:args.ins_position](inv_data)
                    base_feat_spe_ins = self.RCNN_base[:args.ins_position](spe_data)
                    base_feat_inv_inv_ins = self.RCNN_base[:args.ins_position](inv_inv_data)
                    base_feat_inv_spe_ins = self.RCNN_base[:args.ins_position](inv_spe_data)
                    base_feat_spe_inv_ins = self.RCNN_base[:args.ins_position](spe_inv_data)
                    base_feat_spe_spe_ins = self.RCNN_base[:args.ins_position](spe_spe_data)
                    base_feat_inv_ins_ = ops.roi_align(base_feat_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_spe_ins_ = ops.roi_align(base_feat_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_inv_inv_ins_ = ops.roi_align(base_feat_inv_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_inv_spe_ins_ = ops.roi_align(base_feat_inv_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_spe_inv_ins_ = ops.roi_align(base_feat_spe_inv_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    base_feat_spe_spe_ins_ = ops.roi_align(base_feat_spe_spe_ins, self.change(gt_boxes), [5,5], scaling_factor[args.ins_position])
                    con_feat_inv_ins, con_feat_spe_ins = self.mlp_two(base_feat_inv_ins_, base_feat_spe_ins_)
                    con_feat_inv_inv_ins, con_feat_inv_spe_ins = self.mlp_two(base_feat_inv_inv_ins_, base_feat_inv_spe_ins_)
                    con_feat_spe_inv_ins, con_feat_spe_spe_ins = self.mlp_two(base_feat_spe_inv_ins_, base_feat_spe_spe_ins_)
                    contrast_two = [con_feat_inv_ins, con_feat_inv_inv_ins, con_feat_inv_spe_ins, con_feat_spe_ins, con_feat_spe_spe_ins, con_feat_spe_inv_ins]
                else:
                    raise ValueError('wrong contrastive type')
                
            else:
                contrast_two = []
        
            if args.img_position >= args.ins_position:
                if args.setting[1] == 1:
                    base_feat_inv = self.RCNN_base[args.img_position:](base_feat_inv_img)
                elif args.setting[2] == 1:
                    base_feat_inv = self.RCNN_base[args.ins_position:](base_feat_inv_ins)
                else:
                    base_feat_inv = self.RCNN_base(inv_data)
            else:
                if args.setting[2] == 1:
                    base_feat_inv = self.RCNN_base[args.ins_position:](base_feat_inv_ins)
                elif args.setting[1] == 1:
                    base_feat_inv = self.RCNN_base[args.img_position:](base_feat_inv_img)
                else:
                    base_feat_inv = self.RCNN_base(inv_data)
        else:
            if args.setting[1] == 1 and args.contrastive_type == 0:
                base_feat_img = self.RCNN_base[:args.img_position](im_data)
                inv_feat, spe_feat = self.filter_one(base_feat_img)
                con_feat_inv_img, con_feat_spe_img = self.mlp_one(inv_feat, spe_feat)
                contrast_one = [con_feat_inv_img, con_feat_spe_img]
                contrast_two = []
                base_feat_inv = self.RCNN_base[args.img_position:](inv_feat)
            else:
                raise ValueError('code not completion')

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

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, contrast_one, contrast_two
    
    def forward_test(self, im_data, im_info, gt_boxes, num_boxes):
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

    def forward_test_spatial(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat_img = self.RCNN_base[:3](im_data)
        inv_feat, spe_feat = self.filter_one(base_feat_img)
        base_feat_inv = self.RCNN_base[3:](inv_feat)

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

    def forward_baseline(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

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
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

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
    
    def forward_test_visualize(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        inv_data, spe_data = self.filter_one(im_data)
        base_feat_inv = self.RCNN_base[:3](inv_data)
        base_feat_spe = self.RCNN_base[:3](spe_data)
        inv_data, spe_data = self.filter_one.forward_process(im_data)

        return inv_data, spe_data, base_feat_inv, base_feat_spe
    
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
    
    def change(self, input):
        a, b, c = input.size()
        output = []
        for i in range(a):
            for j in range(b):
                if input[i,j,4] == 1:
                    output.append(torch.Tensor([[i, input[i,j,0], input[i,j,1], input[i,j,2], input[i,j,3]]]))
        out = torch.cat(output, dim=0).cuda()
        return out
    
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

    def supcon_cycle_two_side(self, feat, temp):

        batch_size = feat[0].shape[0]
        mask_total = (torch.ones((batch_size * 3, batch_size * 3)) - torch.eye(batch_size * 3)).float().cuda()
        mask_total_ = (torch.ones((batch_size * 2, batch_size * 2)) - torch.eye(batch_size * 2)).float().cuda()

        feat_total = torch.cat((feat[0],feat[1],feat[2]), dim=0)
        feat_total_dot = torch.div(torch.matmul(feat_total, feat_total.T), temp) * mask_total

        # for numerical stability
        logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
        feat_total_dot = feat_total_dot - logits_max.detach()
        feat_total_dot = feat_total_dot * mask_total

        feat_inv_dot = feat_total_dot[0:batch_size*2, 0:batch_size*2]

        # compute log_prob
        feat_inv_total_dot = torch.exp(feat_total_dot[0:batch_size*2])
        feat_inv_dot = feat_inv_dot - torch.log(feat_inv_total_dot.sum(1, keepdim=True)+1e-8)
        feat_inv_dot = feat_inv_dot * mask_total_

        # compute mean of log-likelihood over positive
        mean_log_prob_inv = feat_inv_dot.sum(1) / (batch_size * 2)

        feat_total = torch.cat((feat[3],feat[4],feat[5]), dim=0)
        feat_total_dot = torch.div(torch.matmul(feat_total, feat_total.T), temp) * mask_total

        # for numerical stability
        logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
        feat_total_dot = feat_total_dot - logits_max.detach()
        feat_total_dot = feat_total_dot * mask_total

        feat_spe_dot = feat_total_dot[0:batch_size*2, 0:batch_size*2]

        # compute log_prob
        feat_spe_total_dot = torch.exp(feat_total_dot[0:batch_size*2])
        feat_spe_dot = feat_spe_dot - torch.log(feat_spe_total_dot.sum(1, keepdim=True)+1e-8)
        feat_spe_dot = feat_spe_dot * mask_total_

        # compute mean of log-likelihood over positive
        mean_log_prob_spe = feat_spe_dot.sum(1) / (batch_size * 2)

        mean_log_prob = torch.cat((mean_log_prob_inv, mean_log_prob_spe), dim=0)

        # loss
        loss = -mean_log_prob
        loss = loss.mean()

        return loss

    def supcon_cycle_one_side(self, feat, temp):

        batch_size = feat[0].shape[0]
        mask_total = (torch.ones((batch_size * 3, batch_size * 3)) - torch.eye(batch_size * 3)).float().cuda()
        mask_total_ = (torch.ones((batch_size * 2, batch_size * 2)) - torch.eye(batch_size * 2)).float().cuda()

        feat_total = torch.cat((feat[0],feat[1],feat[2]), dim=0)
        feat_total_dot = torch.div(torch.matmul(feat_total, feat_total.T), temp) * mask_total

        # for numerical stability
        logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
        feat_total_dot = feat_total_dot - logits_max.detach()
        feat_total_dot = feat_total_dot * mask_total

        feat_inv_dot = feat_total_dot[0:batch_size*2, 0:batch_size*2]

        # compute log_prob
        feat_inv_total_dot = torch.exp(feat_total_dot[0:batch_size*2])
        feat_inv_dot = feat_inv_dot - torch.log(feat_inv_total_dot.sum(1, keepdim=True)+1e-8)
        feat_inv_dot = feat_inv_dot * mask_total_

        # compute mean of log-likelihood over positive
        mean_log_prob_inv = feat_inv_dot.sum(1) / (batch_size * 2)

        # loss
        loss = -mean_log_prob_inv
        loss = loss.mean()

        return loss

    def cycle_two_side(self, feat, temp):

        batch_size = feat[0].shape[0]
        mask = torch.eye(batch_size).float().cuda()

        feat_inv_inv = (torch.div(torch.matmul(feat[0], feat[1].T), temp) * mask).sum(1, keepdim=True)
        feat_inv_spe = (torch.div(torch.matmul(feat[0], feat[2].T), temp) * mask).sum(1, keepdim=True)
        feat_inv = torch.cat((feat_inv_inv, feat_inv_spe), dim=1)

        # compute log_prob
        feat_inv = torch.exp(feat_inv)
        feat_inv_inv = (feat_inv_inv - torch.log(feat_inv.sum(1, keepdim=True)+1e-8)).sum(1)

        feat_spe_spe = (torch.div(torch.matmul(feat[3], feat[4].T), temp) * mask).sum(1, keepdim=True)
        feat_spe_inv = (torch.div(torch.matmul(feat[3], feat[5].T), temp) * mask).sum(1, keepdim=True)
        feat_spe = torch.cat((feat_spe_spe, feat_spe_inv), dim=1)

        # compute log_prob
        feat_spe = torch.exp(feat_spe)
        feat_spe_spe = (feat_spe_spe - torch.log(feat_spe.sum(1, keepdim=True)+1e-8)).sum(1)

        mean_log_prob = torch.cat((feat_inv_inv, feat_spe_spe), dim=0)

        # loss
        loss = -mean_log_prob
        loss = loss.mean()

        return loss



    def moco_loss(self, feat_inv, feat_spe, mom_feat_inv, mom_feat_spe, stage, temp):
        assert feat_inv.shape==mom_feat_inv.shape and len(feat_inv.shape)==2
        assert feat_spe.shape==mom_feat_spe.shape and len(feat_spe.shape)==2
        if stage == 'one':
            #feat_inv: bsx128, l_pos: bsxbs
            l_pos = torch.einsum("nc,ck->nk", [feat_inv, mom_feat_inv.T])
            mask = (torch.ones((feat_inv.shape[0], feat_inv.shape[0])) - torch.eye(feat_inv.shape[0])).float().cuda()
            l_pos = l_pos * mask
            #feat_inv: bsx128, queue_one_specific: 128xK, l_neg: bsxk
            l_neg = torch.einsum("nc,ck->nk", [feat_inv, self.queue_one_specific.clone().detach()])
            #feat_total_dot: bsx(bs+K)
            feat_total_dot = torch.div(torch.cat([l_pos, l_neg], dim=1), temp)
            
            logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
            feat_total_dot = feat_total_dot - logits_max.detach()

            feat_inv_dot = feat_total_dot[:, 0:feat_inv.shape[0]]
            feat_inv_dot = feat_inv_dot - torch.log(torch.exp(feat_total_dot).sum(1, keepdim=True)+1e-8)
            feat_inv_dot = feat_inv_dot * mask
            loss_inv = (feat_inv_dot.sum(1) / feat_inv.shape[0]).mean()

            #feat_spe: bsx128, l_pos: bsxbs
            l_pos = torch.einsum("nc,ck->nk", [feat_spe, mom_feat_spe.T])
            mask = (torch.ones((feat_spe.shape[0], feat_spe.shape[0])) - torch.eye(feat_spe.shape[0])).float().cuda()
            l_pos = l_pos * mask
            #feat_spe: bsx128, queue_one_invariant: 128xK, l_neg: bsxk
            l_neg = torch.einsum("nc,ck->nk", [feat_spe, self.queue_one_invariant.clone().detach()])
            #feat_total_dot: bsx(bs+K)
            feat_total_dot = torch.div(torch.cat([l_pos, l_neg], dim=1), temp)
            
            logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
            feat_total_dot = feat_total_dot - logits_max.detach()

            feat_spe_dot = feat_total_dot[:, 0:feat_spe.shape[0]]
            feat_spe_dot = feat_spe_dot - torch.log(torch.exp(feat_total_dot).sum(1, keepdim=True)+1e-8)
            feat_spe_dot = feat_spe_dot * mask
            loss_spe = (feat_spe_dot.sum(1) / feat_spe.shape[0]).mean()

            loss = -(loss_inv + loss_spe) / 2


        else:
            #feat_inv: bsx128, l_pos: bsxbs
            l_pos = torch.einsum("nc,ck->nk", [feat_inv, mom_feat_inv.T])
            mask = (torch.ones((feat_inv.shape[0], feat_inv.shape[0])) - torch.eye(feat_inv.shape[0])).float().cuda()
            l_pos = l_pos * mask
            #feat_inv: bsx128, queue_two_specific: 128xK, l_neg: bsxk
            l_neg = torch.einsum("nc,ck->nk", [feat_inv, self.queue_two_specific.clone().detach()])
            #feat_total_dot: bsx(bs+K)
            feat_total_dot = torch.div(torch.cat([l_pos, l_neg], dim=1), temp)
            
            logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
            feat_total_dot = feat_total_dot - logits_max.detach()

            feat_inv_dot = feat_total_dot[:, 0:feat_inv.shape[0]]
            feat_inv_dot = feat_inv_dot - torch.log(torch.exp(feat_total_dot).sum(1, keepdim=True) + 1e-8)
            feat_inv_dot = feat_inv_dot * mask
            loss_inv = (feat_inv_dot.sum(1) / feat_inv.shape[0]).mean()

            #feat_spe: bsx128, l_pos: bsxbs
            l_pos = torch.einsum("nc,ck->nk", [feat_spe, mom_feat_spe.T])
            mask = (torch.ones((feat_spe.shape[0], feat_spe.shape[0])) - torch.eye(feat_spe.shape[0])).float().cuda()
            l_pos = l_pos * mask
            #feat_spe: bsx128, queue_two_invariant: 128xK, l_neg: bsxk
            l_neg = torch.einsum("nc,ck->nk", [feat_spe, self.queue_two_invariant.clone().detach()])
            #feat_total_dot: bsx(bs+K)
            feat_total_dot = torch.div(torch.cat([l_pos, l_neg], dim=1), temp)
            
            logits_max, _ = torch.max(feat_total_dot, dim=1, keepdim=True)
            feat_total_dot = feat_total_dot - logits_max.detach()

            feat_spe_dot = feat_total_dot[:, 0:feat_spe.shape[0]]
            feat_spe_dot = feat_spe_dot - torch.log(torch.exp(feat_total_dot).sum(1, keepdim=True) + 1e-8)
            feat_spe_dot = feat_spe_dot * mask
            loss_spe = (feat_spe_dot.sum(1) / feat_spe.shape[0]).mean()

            loss = -(loss_inv + loss_spe) / 2
        
        return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, which_type):
        size = keys.shape[0]
        if which_type == 'one_specific':
            ptr = int(self.queue_one_specific_ptr)
            if (ptr + size) <= self.queue_one_specific.shape[1]:
                self.queue_one_specific[:, ptr:ptr + size] = keys.T
                ptr = ptr + size
                self.queue_one_specific_ptr[0] = ptr
            elif size <= self.queue_one_specific.shape[1]:
                self.queue_one_specific[:, ptr:] = keys.T[:, :self.queue_one_specific.shape[1]-ptr]
                self.queue_one_specific[:, :size-self.queue_one_specific.shape[1]+ptr] = keys.T[:, self.queue_one_specific.shape[1]-ptr:]
                ptr = size - self.queue_one_specific.shape[1] + ptr
                self.queue_one_specific_ptr[0] = ptr
            else:
                raise ValueError('enqueue length is too large')
        elif which_type == 'one_invariant':
            ptr = int(self.queue_one_invariant_ptr)
            if (ptr + size) <= self.queue_one_invariant.shape[1]:
                self.queue_one_invariant[:, ptr:ptr + size] = keys.T
                ptr = ptr + size
                self.queue_one_invariant_ptr[0] = ptr
            elif size <= self.queue_one_invariant.shape[1]:
                self.queue_one_invariant[:, ptr:] = keys.T[:, :self.queue_one_invariant.shape[1]-ptr]
                self.queue_one_invariant[:, :size-self.queue_one_invariant.shape[1]+ptr] = keys.T[:, self.queue_one_invariant.shape[1]-ptr:]
                ptr = size - self.queue_one_invariant.shape[1] + ptr
                self.queue_one_invariant_ptr[0] = ptr
            else:
                raise ValueError('enqueue length is too large')
        elif which_type == 'two_specific':
            ptr = int(self.queue_two_specific_ptr)
            if (ptr + size) <= self.queue_two_specific.shape[1]:
                self.queue_two_specific[:, ptr:ptr + size] = keys.T
                ptr = ptr + size
                self.queue_two_specific_ptr[0] = ptr
            elif size <= self.queue_two_specific.shape[1]:
                self.queue_two_specific[:, ptr:] = keys.T[:, :self.queue_two_specific.shape[1]-ptr]
                self.queue_two_specific[:, :size-self.queue_two_specific.shape[1]+ptr] = keys.T[:, self.queue_two_specific.shape[1]-ptr:]
                ptr = size - self.queue_two_specific.shape[1] + ptr
                self.queue_two_specific_ptr[0] = ptr
            else:
                raise ValueError('enqueue length is too large')
        elif which_type == 'two_invariant':
            ptr = int(self.queue_two_invariant_ptr)
            if (ptr + size) <= self.queue_two_invariant.shape[1]:
                self.queue_two_invariant[:, ptr:ptr + size] = keys.T
                ptr = ptr + size
                self.queue_two_invariant_ptr[0] = ptr
            elif size <= self.queue_two_invariant.shape[1]:
                self.queue_two_invariant[:, ptr:] = keys.T[:, :self.queue_two_invariant.shape[1]-ptr]
                self.queue_two_invariant[:, :size-self.queue_two_invariant.shape[1]+ptr] = keys.T[:, self.queue_two_invariant.shape[1]-ptr:]
                ptr = size - self.queue_two_invariant.shape[1] + ptr
                self.queue_two_invariant_ptr[0] = ptr
            else:
                raise ValueError('enqueue length is too large')
        else:
            raise ValueError('Dequeue and enqueue error')

    @torch.no_grad()
    def _momentum_update(self, args, m=0.999):
        for param_q, param_k in zip(self.RCNN_base.parameters(), self.momentum_RCNN_base.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        
        if not args.setting[1] == 0 and args.projector:
            for param_q, param_k in zip(self.mlp_one.parameters(), self.momentum_mlp_one.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1.0 - m)

        if not args.setting[2] == 0 and args.projector:
            for param_q, param_k in zip(self.mlp_two.parameters(), self.momentum_mlp_two.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1.0 - m)