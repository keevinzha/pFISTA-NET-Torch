# -*- coding: utf-8 -*-
"""
Tools code - pytorch

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


def np_fft2c(x):

    # x = [nslice, kx, ky, ncoil]
    x = np.transpose(x, (0, 3, 1, 2))  # [nslice, ncoil, kx, ky]
    _, _, kx, ky = np.float32(x.shape)
    kxky = np.complex64(kx * ky + 0j)
    axes = (-2, -1)
    x = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kxky)
    x = np.transpose(x, (0, 2, 3, 1))  # [nslice, kx, ky, ncoil]
    return x


def np_ifft2c(x):

    # x = [nslice, kx, ky, ncoil]
    x = np.transpose(x, (0, 3, 1, 2))  # [nslice, ncoil, kx, ky]
    _, _, kx, ky = np.float32(x.shape)
    kxky = np.complex64(kx * ky + 0j)
    axes = (-2, -1)
    x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kxky)
    x = np.transpose(x, (0, 2, 3, 1))  # [nslice, kx, ky, ncoil]
    return x


def torch_fft2c(x):
    # x = [nslice, ncoil, kx, ky]

    kx = int(x.shape[-2])
    ky = int(x.shape[-1])

    dim = (-2, -1)
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(kx*ky)
    return x  # [nslice, ncoil, kx, ky]


def torch_ifft2c(x):
    # x = [nslice, ncoil, kx, ky]

    kx = int(x.shape[-2])
    ky = int(x.shape[-1])

    dim = (-2, -1)
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(kx*ky)
    return x  # [nslice, ncoil, kx, ky]


def np_fft1c(x):
    # x = [nslice, ky, ncoil]
    x = np.transpose(x, (0, 2, 1))  # [nslice, ncoil, ky]
    _, _, ky = np.float32(x.shape)
    ky = np.complex64(ky + 0j)
    axes = -1
    x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
    x = np.transpose(x, (0, 2, 1))  # [nslice, ky, ncoil]
    return x


def np_ifft1c(x):
    # x = [nslice, ky, ncoil]
    x = np.transpose(x, (0, 2, 1))  # [nslice, ncoil, ky]
    _, _, ky = np.float32(x.shape)
    ky = np.complex64(ky + 0j)
    axes = -1
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
    x = np.transpose(x, (0, 2, 1))  # [nslice, ky, ncoil]
    return x


def torch_fft1c(x):
    # x = [nslice, ncoil, ky]

    ky = int(x.shape[-1])

    dim = -1
    x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(ky)
    return x  # [nslice, ncoil, ky]


def torch_ifft1c(x):
    # x = [nslice, ncoil, ky]

    ky = int(x.shape[-1])

    dim = -1
    x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(ky)
    return x  # [nslice, ncoil, ky]


def np_fft1c_hybrid(x, dim):
    # x = [nslice, kx, ky, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 1))  # [nslice, ky, ncoil, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kx)
        x = np.transpose(x, (0, 3, 1, 2))  # [nslice, kx, ky, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ncoil, ky]
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ky, ncoil]
    return x


def np_ifft1c_hybrid(x, dim):
    # x = [nslice, kx, ky, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 1))  # [nslice, ky, ncoil, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kx)
        x = np.transpose(x, (0, 3, 1, 2))  # [nslice, kx, ky, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ncoil, ky]
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ky, ncoil]
    return x


def torch_fft1c_hybrid(x, dimension):
    # x = [nslice, ncoil, kx, ky]
    if dimension == 1:
        x = x.permute(0, 3, 1, 2)  # [nslice, ky, ncoil, kx]
        kx = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(kx)
        x = x.permute(0, 2, 3, 1)  # [nslice, ncoil, kx, ky]

    if dimension == 2:
        x = x.permute(0, 2, 1, 3)  # [nslice, kx, ncoil, ky]
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(ky)
        x = x.permute(0, 2, 1, 3)  # [nslice, ncoil, kx, ky]
    return x


def torch_ifft1c_hybrid(x, dimension):
    # x = [nslice, ncoil, kx, ky]
    if dimension == 1:
        x = x.permute(0, 3, 1, 2)  # [nslice, ky, ncoil, kx]
        kx = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(kx)
        x = x.permute(0, 2, 3, 1)  # [nslice, ncoil, kx, ky]

    if dimension == 2:
        x = x.permute(0, 2, 1, 3)  # [nslice, kx, ncoil, ky]
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(ky)
        x = x.permute(0, 2, 1, 3)  # [nslice, ncoil, kx, ky]
    return x


def torch_complex2double(x):  # input x=[1-2j, 2+3j, 4+9j]
    x_real, x_imag = torch.real(x), torch.imag(x)
    # if dim=-1, torch.cat([real, imag], dim=1)=[ 1.  2.  4. -2.  3.  9.]
    x_double = torch.cat([x_real, x_imag], dim=1)
    return x_double


def torch_double2complex(x):
    nchannel = x.shape[1]
    nchannel = int(nchannel / 2)
    x_real, x_imag = x[:, :nchannel, ...], x[:, nchannel:, ...]
    x_complex = torch.complex(x_real, x_imag)  # complex
    return x_complex


def np_sos(x):
    # x = [nslice, kx, ky, ncoil] complex
    x = np.sum(np.abs(x**2), axis=-1)
    x = x**(1.0/2)
    return x  # x = [nslice, kx, ky] real


def torch_sos(x):
    # x = [nslice, ncoil, kx, ky] complex
    x = torch.sum(torch.abs(x**2), dim=1)
    x = x**(1.0/2)
    return x  # x = [nslice, kx, ky] real


def np_strict_dc(mask_coil, y_input, x_rec):
    k_sample = mask_coil * y_input
    k_no_sample = (1 - mask_coil) * x_rec
    k_dc = k_sample + k_no_sample
    return k_dc
