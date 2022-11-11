# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 21:02
# @Author  : keevinzha
# @File    : modules.py

import torch
import torch.nn as nn
from torch.autograd import Variable

from model.submodules import conv2d_relu, conv2d_bn_relu
from utils.Tools_torch import torch_complex2double, torch_double2complex, torch_fft2c, torch_ifft2c



class Conv_layer(nn.Module):
    def __init__(self, args, convert=False):
        super().__init__()
        self.convert = convert
        self.conv2d_relu = conv2d_relu(2, args.filter_num, 3)
        laryers = []
        for i in range(args.layer_num-2):
            laryers.append(conv2d_bn_relu(args.filter_num, args.filter_num, 3))
        self.conv2d_bn_relu = nn.Sequential(*laryers)
        self.conv2d = nn.Conv2d(args.filter_num, args.filter_num, kernel_size=3, padding='same')

    def forward(self, input):
        if self.convert:
            input = torch_complex2double(input)
        x = self.conv2d_relu(input)
        x = self.conv2d_bn_relu(x)
        x = self.conv2d(x)
        return x

class SoftThr_layer(nn.Module):
    def __init__(self, lammda_Init, num_Block, grad):
        super(SoftThr_layer, self).__init__()
        self.num_Block = num_Block
        self.grad = grad
        self.lammda = nn.Parameter(torch.Tensor(lammda_Init), requires_grad=grad)


    def soft_threash(self, x, thresh):
        x = torch_complex2double(x)
        abs_x = torch.abs(x)
        x_opt = torch.sign(x)*(abs_x-thresh+torch.abs(abs_x-thresh))/2
        x_opt = torch_double2complex(x_opt)
        return x_opt

    def forward(self, input):
        Sparse, gamma_step = input
        soft_thr = gamma_step*self.lammda
        soft_fista = self.soft_threash(Sparse, soft_thr)
        return soft_fista


class DC_layer_last(nn.Module):
    def __init__(self, DC_lamda_Init, grad):
        super(DC_layer_last, self).__init__()
        self.DC_lamda_Init = DC_lamda_Init
        self.grad = grad
        self.DC_lamda = nn.Parameter(torch.Tensor(self.DC_lamda_Init), requires_grad=grad)

    def forward(self, input):
        X_img, DC_mask, inp_feq = input
        X_freq = torch_fft2c(X_img)
        DC_lamda = self.DC_lamda
        DC_lamda = DC_lamda.to(torch.complex64)
        freq_sample = DC_mask * torch.div(DC_lamda * X_freq + inp_feq, (DC_lamda + 1))
        freq_no_sample = (1 - DC_mask) * X_freq
        X_DC_freg = freq_sample + freq_no_sample
        X_DC = torch_ifft2c(X_DC_freg)

        return X_DC
    

class DC_layer_first(nn.Module):
    def __init__(self):
        super(DC_layer_first, self).__init__()

    def forward(self, X, mask, Y_freq, gamma):
        DC = torch_ifft2c(mask*(Y_freq - mask * torch_fft2c(X)))
        data_grad = gamma * torch.real(DC), gamma *  torch.imag(DC) #return Tensor
        data_grad = data_grad.to(torch.complex64)
        Ts = torch.add(X, data_grad)
        return Ts

class Sparse_layer(nn.Module):
    def __init__(self, args):
        super(Sparse_layer, self).__init__()
        self.conv_layer = Conv_layer(args, convert=False)
        self.conv2d = nn.Conv2d(args.filter_num, 2, args.filter_size, padding='same')
    def forward(self, input):
        input = torch_complex2double(input)
        x = self.conv_layer(input)
        conv_out = self.conv2d(x)
        conv_out = torch_double2complex(conv_out)
        return conv_out
