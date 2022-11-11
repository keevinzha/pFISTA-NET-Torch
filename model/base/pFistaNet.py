# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 10:13
# @Author  : keevinzha
# @File    : pFistaNet.py
import torch
import torch.nn as nn


from utils.Tools_torch import torch_fft2c, torch_complex2double
from module.modules import *


args = []


class pFistaNet(nn.Module):
    def __init__(self, number_blocks, gamma_init, lammda_init, DC_lamda_init, grad):
        super(pFistaNet, self).__init__()
        self.gamma_step = nn.Parameter(torch.Tensor(gamma_init), requires_grad=grad)
        self.spars_layer = Sparse_layer(args=args)
        self.DC_first_layer = DC_layer_first()
        self.softThr_layer = SoftThr_layer(lammda_init, number_blocks, grad)
        self.DC_last_layer = DC_layer_last(DC_lamda_init, grad)

    def forward(self, input):
        inp_img, mask = input
        inp_feq = torch_fft2c(inp_img)
        x_pFISTA = inp_img
        out_img_list = []
        for i in range(self.number_blocks):
            x_temp = self.DC_first_layer(x_pFISTA, mask, inp_feq, self.gamma_step)
            Sparse = self.spars_layer(x_temp)
            soft_fista =self.softThr_layer([Sparse, self.gamma_step])
            x_pFISTA = x_temp + self.spars_layer(soft_fista)
            x_pFISTA = self.DC_last_layer([x_pFISTA, mask, inp_feq])

            out_img_list.append(x_pFISTA)

        return out_img_list[-1]


