# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 11:03
# @Author  : keevinzha
# @File    : model_entry.py
from model.base.pFistaNet import CNN
import torch.nn as nn


def select_model(args):
    type2model = {
        'pFistaNet': CNN(args)
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
