# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 10:08
# @Author  : keevinzha
# @File    : submodules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(in_dim, out_dim, kernel, stride=1, pad='same'):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )

def conv2d_relu(in_dim, out_dim, kernel, stride=1, pad='same'):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad),
        nn.ReLU()
    )
