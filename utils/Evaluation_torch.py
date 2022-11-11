# -*- coding: utf-8 -*-
"""
Evaluation code - pytorch

Created on 2022/06/25

@author: Zi Wang

If you want to use this code, please cite following paper:

Email: Xiaobo Qu (quxiaobo@xmu.edu.cn) CC: Zi Wang (wangziblake@163.com)
Homepage: http://csrc.xmu.edu.cn

Affiliations: Computational Sensing Group (CSG), Departments of Electronic Science, Xiamen University, Xiamen 361005, China
All rights are reserved by CSG.
"""

import torch
import numpy as np
from Tools.Tools_torch import *


def loss_mse_all_torch(pre_complex, label_complex):  # k-space loss
    pred_length = 5  # [niter, batch_size, ncoil, kx, ky]  complex
    error = torch.abs(label_complex - pre_complex)

    cost_all = torch.mean(torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[2:]), dim=1), dim=0)
    return cost_all


def loss_mse_last_torch(pre_complex, label_complex):  # k-space loss
    pred_length = 4  # [batch_size, ncoil, kx, ky]  complex
    error = torch.abs(label_complex - pre_complex)

    cost_all = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
    return cost_all


def loss_mse_all_image_torch(pre_complex, label_complex, niter):  # image loss

    # [niter, batch_size, ncoil, kx, ky]  complex k-space input
    label_complex = torch_ifft2c(label_complex)
    pred_length = 5 - 1
    cost_all = 0

    for i in range(niter):
        pre_complex_single = torch_ifft2c(pre_complex[i])

        error = torch.abs(label_complex - pre_complex_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def loss_mse_all_sosimage_torch(pre_complex, label_complex, niter):  # sos image loss

    # [niter, batch_size, ncoil, kx, ky]  complex k-space input
    label_sos = torch_sos(torch_ifft2c(label_complex))
    pred_length = 4 - 1
    cost_all = 0

    for i in range(niter):
        pre_sos_single = torch_sos(torch_ifft2c(pre_complex[i]))

        error = torch.abs(label_sos - pre_sos_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def loss_mse_all_combimage_torch(pre_complex, label_complex, CSM, niter):  # CSM-combined image loss

    # [niter, batch_size, ncoil, kx, ky]  complex combimage input
    CSM_conj = torch.conj(CSM)
    label_complex = torch_ifft2c(label_complex)  # complex multi-coil
    label_complex_single = torch.sum(label_complex * CSM_conj, dim=1, keepdim=True)  # complex single-coil
    pred_length = 5 - 1
    cost_all = 0

    for i in range(niter):
        pre_complex_single = pre_complex[i]

        error = torch.abs(label_complex_single - pre_complex_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def loss_mse_last_image_torch(pre_complex, label_complex):  # image loss

    # k-space input
    label_complex = torch_ifft2c(label_complex)
    pre_complex = torch_ifft2c(pre_complex)

    pred_length = 4  # [batch_size, ncoil, kx, ky]  complex

    error = torch.abs(label_complex - pre_complex)
    cost_all = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
    return cost_all


def loss_mse_all_1D_torch(pre_complex, label_complex, niter):  # k-space loss

    # [niter, batch_size, ncoil, ky]  complex k-space input
    label_complex = label_complex
    pred_length = 4 - 1
    cost_all = 0

    for i in range(niter):
        pre_complex_single = pre_complex[i]

        error = torch.abs(label_complex - pre_complex_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def loss_mse_all_image_1D_torch(pre_complex, label_complex, niter):  # image loss

    # [niter, batch_size, ncoil, ky]  complex k-space input
    label_complex = torch_ifft1c(label_complex)
    pred_length = 4 - 1
    cost_all = 0

    for i in range(niter):
        pre_complex_single = torch_ifft1c(pre_complex[i])

        error = torch.abs(label_complex - pre_complex_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def loss_mse_all_sosimage_1D_torch(pre_complex, label_complex, niter):  # image loss

    # [niter, batch_size, ncoil, ky]  complex k-space input
    label_sos = torch_sos(torch_ifft1c(label_complex))
    pred_length = 3 - 1
    cost_all = 0

    for i in range(niter):
        pre_sos_single = torch_sos(torch_ifft1c(pre_complex[i]))

        error = torch.abs(label_sos - pre_sos_single)
        cost_single = torch.mean(torch.sum(torch.square(error), dim=tuple(range(pred_length))[1:]), dim=0)
        cost_all = cost_all + cost_single
    cost_all = cost_all / niter
    return cost_all


def RLNE_torch(pre_complex, label_complex):
    pred_length = 4  # [batch_size, ncoil, kx, ky]  complex

    # k-space input
    label_complex = torch_ifft2c(label_complex)
    pre_complex = torch_ifft2c(pre_complex)

    # axis_sum = (1, 2, 3)
    axis_sum = tuple(range(pred_length))[1:]
    # [batch_size, number_channels * 2, ...]
    label_complex_2 = torch.cat([torch.real(label_complex), torch.imag(label_complex)], dim=axis_sum[1])
    pre_complex_2 = torch.cat([torch.real(pre_complex), torch.imag(pre_complex)], dim=axis_sum[1])
    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_complex_2 - pre_complex_2), dim=axis_sum))
    l2_true = torch.sqrt(torch.sum(torch.square(label_complex_2), dim=axis_sum))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne


def RLNE_combimage_torch(pre_complex, label_complex, CSM):
    pred_length = 4  # [batch_size, ncoil, kx, ky]  complex

    # label: k-space input  rec: combimage input
    CSM_conj = torch.conj(CSM)
    label_complex = torch_ifft2c(label_complex)
    label_complex = torch.sum(label_complex * CSM_conj, dim=1, keepdim=True)  # complex single-coil

    # axis_sum = (1, 2, 3)
    axis_sum = tuple(range(pred_length))[1:]
    # [batch_size, number_channels * 2, ...]
    label_complex_2 = torch.cat([torch.real(label_complex), torch.imag(label_complex)], dim=axis_sum[1])
    pre_complex_2 = torch.cat([torch.real(pre_complex), torch.imag(pre_complex)], dim=axis_sum[1])
    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_complex_2 - pre_complex_2), dim=axis_sum))
    l2_true = torch.sqrt(torch.sum(torch.square(label_complex_2), dim=axis_sum))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne


def RLNE_combimage_torch_v2(pre_complex, label_complex, CSM):
    pred_length = 4  # [batch_size, ncoil, kx, ky]  complex

    # label: k-space input  rec: combimage input
    label_complex = torch_ifft2c(label_complex)  # complex multi-coil
    pre_complex = pre_complex * CSM  # complex multi-coil

    # axis_sum = (1, 2, 3)
    axis_sum = tuple(range(pred_length))[1:]
    # [batch_size, number_channels * 2, ...]
    label_complex_2 = torch.cat([torch.real(label_complex), torch.imag(label_complex)], dim=axis_sum[1])
    pre_complex_2 = torch.cat([torch.real(pre_complex), torch.imag(pre_complex)], dim=axis_sum[1])
    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_complex_2 - pre_complex_2), dim=axis_sum))
    l2_true = torch.sqrt(torch.sum(torch.square(label_complex_2), dim=axis_sum))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne


def RLNE_sosimage_torch(pre_complex, label_complex):
    # [batch_size, ncoil, kx, ky]  complex

    # k-space input
    label_sos = torch_sos(torch_ifft2c(label_complex))
    pre_sos = torch_sos(torch_ifft2c(pre_complex))

    # axis_sum = (1, 2)
    pred_length = 3
    axis_sum = tuple(range(pred_length))[1:]
    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_sos - pre_sos), dim=axis_sum))
    l2_true = torch.sqrt(torch.sum(torch.square(label_sos), dim=axis_sum))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne


def RLNE_1D_torch(pre_complex, label_complex):
    pred_length = 3  # [batch_size, ncoil, ky]  complex

    # k-space input
    label_complex = torch_ifft1c(label_complex)
    pre_complex = torch_ifft1c(pre_complex)

    # axis_sum = (1, 2)
    axis_sum = tuple(range(pred_length))[1:]
    # [batch_size, number_channels * 2, ...]
    label_complex_2 = torch.cat([torch.real(label_complex), torch.imag(label_complex)], dim=axis_sum[1])
    pre_complex_2 = torch.cat([torch.real(pre_complex), torch.imag(pre_complex)], dim=axis_sum[1])
    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_complex_2 - pre_complex_2), dim=axis_sum))
    l2_true = torch.sqrt(torch.sum(torch.square(label_complex_2), dim=axis_sum))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne


def RLNE_sosimage_1D_torch(pre_complex, label_complex):
    # [batch_size, ncoil, ky]  complex

    # k-space input
    label_sos = torch_sos(torch_ifft1c(label_complex))
    pre_sos = torch_sos(torch_ifft1c(pre_complex))

    # [batch_size, ]
    error_sum = torch.sqrt(torch.sum(torch.square(label_sos - pre_sos), dim=1))
    l2_true = torch.sqrt(torch.sum(torch.square(label_sos), dim=1))
    # 1
    rlne = torch.mean(error_sum / l2_true, dim=0)
    return rlne
