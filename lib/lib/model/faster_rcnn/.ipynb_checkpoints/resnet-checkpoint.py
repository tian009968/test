from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from extra import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=50, pretrained=False, class_agnostic=False):
    self.model_path = '/model/kunyuwangpKtx/resnet50_pretrained/resnet50.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self, args):
    resnet = resnet50()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    if args.fix:
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def _init_extra_modules(self, args):

    layer_channel = [64,64,256,512,1024]

    if (not args.setting[1] == 0 or not args.setting[2] == 0) and args.contrastive_type == 2:
      resnet = resnet50()
      self.momentum_RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
         resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
      for param, param_momentum in zip(self.RCNN_base.parameters(), self.momentum_RCNN_base.parameters()):
            param_momentum.data.copy_(param.data)  # initialize
            param_momentum.requires_grad = False  # not update by gradient

      if args.setting[0] == 0:
        self.filter_one = Filter_Two_Side()
      elif args.setting[0] == 1:
        self.filter_one = Filter_One_Side() 
      elif args.setting[0] == 2:
        self.filter_one = Conv_Two_Side() 
      elif args.setting[0] == 3:
        self.filter_one = Conv_One_Side() 
      elif args.setting[0] == 4:
        self.filter_one = Spa_Conv()
      elif args.setting[0] == 5:
        self.filter_one = Filter_One_Side_Sigmoid() 
      else:
        raise ValueError('Hyper-parameter does not comply with the rule, this is the baseline')

      if args.setting[1] == 1:
        if args.projector:
          self.mlp_one = ContrastiveMLP_v2(layer_channel[args.img_position - 3], 128)
          self.momentum_mlp_one = ContrastiveMLP_v2(layer_channel[args.img_position - 3], 128)
          for param, param_momentum in zip(self.mlp_one.parameters(), self.momentum_mlp_one.parameters()):
              param_momentum.data.copy_(param.data)  # initialize
              param_momentum.requires_grad = False  # not update by gradient
          self.register_buffer("queue_one_invariant", torch.randn(128, args.queue_length))
          self.queue_one_invariant = F.normalize(self.queue_one_invariant, dim=0)
          self.register_buffer("queue_one_invariant_ptr", torch.zeros(1, dtype=torch.long))
          self.register_buffer("queue_one_specific", torch.randn(128, args.queue_length))
          self.queue_one_specific = F.normalize(self.queue_one_specific, dim=0)
          self.register_buffer("queue_one_specific_ptr", torch.zeros(1, dtype=torch.long))
        else:
          self.mlp_one = ContrastiveMLP(layer_channel[args.img_position - 3], 128)
          self.momentum_mlp_one = ContrastiveMLP(layer_channel[args.img_position - 3], 128)
          for param, param_momentum in zip(self.mlp_one.parameters(), self.momentum_mlp_one.parameters()):
              param_momentum.data.copy_(param.data)  # initialize
              param_momentum.requires_grad = False  # not update by gradient
          self.register_buffer("queue_one_invariant", torch.randn(layer_channel[args.img_position - 3], args.queue_length))
          self.queue_one_invariant = F.normalize(self.queue_one_invariant, dim=0)
          self.register_buffer("queue_one_invariant_ptr", torch.zeros(1, dtype=torch.long))
          self.register_buffer("queue_one_specific", torch.randn(layer_channel[args.img_position - 3], args.queue_length))
          self.queue_one_specific = F.normalize(self.queue_one_specific, dim=0)
          self.register_buffer("queue_one_specific_ptr", torch.zeros(1, dtype=torch.long))
      else:
        pass

      if args.setting[2] == 1:
        if args.projector:
          self.mlp_two = ContrastiveMLP_v2(layer_channel[args.ins_position - 3], 128)
          self.momentum_mlp_two = ContrastiveMLP_v2(layer_channel[args.ins_position - 3], 128)
          for param, param_momentum in zip(self.mlp_two.parameters(), self.momentum_mlp_two.parameters()):
              param_momentum.data.copy_(param.data)  # initialize
              param_momentum.requires_grad = False  # not update by gradient
          self.register_buffer("queue_two_invariant", torch.randn(128, args.queue_length*args.object_per_image))
          self.queue_two_invariant = F.normalize(self.queue_two_invariant, dim=0)
          self.register_buffer("queue_two_invariant_ptr", torch.zeros(1, dtype=torch.long))
          self.register_buffer("queue_two_specific", torch.randn(128, args.queue_length*args.object_per_image))
          self.queue_two_specific = F.normalize(self.queue_two_specific, dim=0)
          self.register_buffer("queue_two_specific_ptr", torch.zeros(1, dtype=torch.long))
        else:
          self.mlp_two = ContrastiveMLP(layer_channel[args.ins_position - 3], 128)
          self.momentum_mlp_two = ContrastiveMLP(layer_channel[args.ins_position - 3], 128)
          for param, param_momentum in zip(self.mlp_two.parameters(), self.momentum_mlp_two.parameters()):
              param_momentum.data.copy_(param.data)  # initialize
              param_momentum.requires_grad = False  # not update by gradient
          self.register_buffer("queue_two_invariant", torch.randn(layer_channel[args.ins_position - 3], args.queue_length*args.object_per_image))
          self.queue_two_invariant = F.normalize(self.queue_two_invariant, dim=0)
          self.register_buffer("queue_two_invariant_ptr", torch.zeros(1, dtype=torch.long))
          self.register_buffer("queue_two_specific", torch.randn(layer_channel[args.ins_position - 3], args.queue_length*args.object_per_image))
          self.queue_two_specific = F.normalize(self.queue_two_specific, dim=0)
          self.register_buffer("queue_two_specific_ptr", torch.zeros(1, dtype=torch.long))
      else:
        pass

    else:
      if args.setting[0] == 0:
        self.filter_one = Filter_Two_Side()
      elif args.setting[0] == 1:
        self.filter_one = Filter_One_Side() 
      elif args.setting[0] == 2:
        self.filter_one = Conv_Two_Side() 
      elif args.setting[0] == 3:
        self.filter_one = Conv_One_Side() 
      elif args.setting[0] == 4:
        self.filter_one = Spa_Conv()
      elif args.setting[0] == 5:
        self.filter_one = Filter_One_Side_Sigmoid() 
      else:
        raise ValueError('Hyper-parameter does not comply with the rule, this is the baseline')

      if args.setting[1] == 1:
        if args.projector:
          self.mlp_one = ContrastiveMLP_v2(layer_channel[args.img_position - 3], 128)
        else:
          self.mlp_one = ContrastiveMLP(layer_channel[args.img_position - 3], 128)
      else:
        pass

      if args.setting[2] == 1:
        if args.projector:
          self.mlp_two = ContrastiveMLP_v2(layer_channel[args.ins_position - 3], 128)
        else:
          self.mlp_two = ContrastiveMLP(layer_channel[args.ins_position - 3], 128)
      else:
        pass

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
