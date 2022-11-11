# -*- coding: utf-8 -*-
"""
Loaddata code - torch

Created on 2022/06/25

@author: Zi Wang

If you want to use this code, please cite following paper:

Email: Xiaobo Qu (quxiaobo@xmu.edu.cn) CC: Zi Wang (wangziblake@163.com)
Homepage: http://csrc.xmu.edu.cn

Affiliations: Computational Sensing Group (CSG), Departments of Electronic Science, Xiamen University, Xiamen 361005, China
All rights are reserved by CSG.
"""

import h5py
import os
import numpy as np
from Tools.Tools_torch import *


def read_data_size(path_dataset):
    with h5py.File(path_dataset) as h5file:

        # input k-space
        input_k = h5file['k_input'][:]  # [8, 225, 224]
        input_k = input_k['real'] + input_k['imag'] * 1j
        if input_k.ndim == 2:
            input_k = np.expand_dims(input_k, axis=-1)  # [225, 224, nslice]
            input_k = np.expand_dims(input_k, axis=0)  # [1, 225, 224, nslice]
        elif input_k.ndim == 3:
            input_k = np.expand_dims(input_k, axis=-1)  # [8, 225, 224, nslice]
        input_k = np.transpose(input_k, (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        nslice, kx, ky, ncoil = input_k.shape
    return nslice, kx, ky, ncoil


def loaddata_Train_maxnorm(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:

        #  input k-space
        #  Complex
        Training_k_space = h5file['k_input'][:]  # [8, 225, 224]
        Training_k_space = Training_k_space['real'] + Training_k_space['imag'] * 1j
        if Training_k_space.ndim == 2:
            Training_k_space = np.expand_dims(Training_k_space, axis=-1)  # [225, 224, nslice]
            Training_k_space = np.expand_dims(Training_k_space, axis=0)  # [1, 225, 224, nslice]
        elif Training_k_space.ndim == 3:
            Training_k_space = np.expand_dims(Training_k_space, axis=-1)  # [8, 225, 224, nslice]
        Training_k_space = np.transpose(Training_k_space.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Training_image = np_ifft2c(Training_k_space)

        #  kx: frequency encoding
        #  ky: phase encoding
        _, kx, ky, ncoil = Training_k_space.shape

        #  label k-space
        #  Complex
        label_k_space = h5file['k_label'][:]
        label_k_space = label_k_space['real'] + label_k_space['imag'] * 1j
        if label_k_space.ndim == 2:
            label_k_space = np.expand_dims(label_k_space, axis=-1)
            label_k_space = np.expand_dims(label_k_space, axis=0)
        elif label_k_space.ndim == 3:
            label_k_space = np.expand_dims(label_k_space, axis=-1)
        label_k_space = np.transpose(label_k_space.astype(np.complex64), (3, 2, 1, 0))
        # label_image = np_ifft2c(label_k_space)

        #  mask coil
        #  Float
        mask_coil = h5file['mask_coil'][:]  # [8, 225, 224]
        if mask_coil.ndim == 2:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [225, 224, nslice]
            mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 225, 224, nslice]
        elif mask_coil.ndim == 3:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [8, 225, 224, nslice]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Training_mask_coil = mask_coil * (1 + 0j)  # Complex

        #  Normalization
        # factor_norm = np.amax(np.abs(Training_k_space), axis=(-3, -2, -1), keepdims=True)
        factor_norm = np.amax(np.abs(Training_image), axis=(-3, -2, -1), keepdims=True)
        # factor_norm = 1  # No normalization

        # Training_ilabels_Norm = label_image / factor_norm   # image label: [1, 224, 225, 8]
        Training_klabels_Norm = label_k_space / factor_norm  # k-space label: [1, 224, 225, 8]

        Training_inputs_Norm = Training_k_space / factor_norm  # k-space input: [1, 224, 225, 8]

        return Training_inputs_Norm, Training_klabels_Norm, Training_mask_coil, kx, ky, ncoil


def loaddata_Train_stdnorm(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:

        #  input k-space
        #  Complex
        Training_k_space = h5file['k_input'][:]  # [8, 225, 224]
        Training_k_space = Training_k_space['real'] + Training_k_space['imag'] * 1j
        if Training_k_space.ndim == 2:
            Training_k_space = np.expand_dims(Training_k_space, axis=-1)  # [225, 224, nslice]
            Training_k_space = np.expand_dims(Training_k_space, axis=0)  # [1, 225, 224, nslice]
        elif Training_k_space.ndim == 3:
            Training_k_space = np.expand_dims(Training_k_space, axis=-1)  # [8, 225, 224, nslice]
        Training_k_space = np.transpose(Training_k_space.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        # Training_image = np_ifft2c(Training_k_space)

        #  kx: frequency encoding
        #  ky: phase encoding
        _, kx, ky, ncoil = Training_k_space.shape

        #  label k-space
        #  Complex
        label_k_space = h5file['k_label'][:]
        label_k_space = label_k_space['real'] + label_k_space['imag'] * 1j
        if label_k_space.ndim == 2:
            label_k_space = np.expand_dims(label_k_space, axis=-1)
            label_k_space = np.expand_dims(label_k_space, axis=0)
        elif label_k_space.ndim == 3:
            label_k_space = np.expand_dims(label_k_space, axis=-1)
        label_k_space = np.transpose(label_k_space.astype(np.complex64), (3, 2, 1, 0))
        # label_image = np_ifft2c(label_k_space)

        #  mask coil
        #  Float
        mask_coil = h5file['mask_coil'][:]  # [8, 225, 224]
        if mask_coil.ndim == 2:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [225, 224, nslice]
            mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 225, 224, nslice]
        elif mask_coil.ndim == 3:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [8, 225, 224, nslice]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Training_mask_coil = mask_coil * (1 + 0j)  # Complex

        #  Normalization
        factor_norm = np.std(Training_k_space, axis=(-3, -2, -1), ddof=1)
        # factor_norm = np.amax(np_sos(Training_image), axis=(-2, -1))
        # factor_norm = 1  # No normalization

        # Training_ilabels_Norm = label_image / factor_norm   # image label: [1, 224, 225, 8]
        Training_klabels_Norm = label_k_space / factor_norm  # k-space label: [1, 224, 225, 8]

        Training_inputs_Norm = Training_k_space / factor_norm  # k-space input: [1, 224, 225, 8]

        return Training_inputs_Norm, Training_klabels_Norm, Training_mask_coil, kx, ky, ncoil


def loaddata_Test_maxnorm(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:

        #  input k-space
        #  Complex
        Testing_k_space = h5file['k_input'][:]  # [8, 225, 224]
        Testing_k_space = Testing_k_space['real'] + Testing_k_space['imag'] * 1j
        if Testing_k_space.ndim == 2:
            Testing_k_space = np.expand_dims(Testing_k_space, axis=-1)  # [225, 224, nslice]
            Testing_k_space = np.expand_dims(Testing_k_space, axis=0)  # [1, 225, 224, nslice]
        elif Testing_k_space.ndim == 3:
            Testing_k_space = np.expand_dims(Testing_k_space, axis=-1)  # [8, 225, 224, nslice]
        Testing_k_space = np.transpose(Testing_k_space.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Testing_image = np_ifft2c(Testing_k_space)

        _, kx, ky, ncoil = Testing_k_space.shape

        #  mask coil
        #  Float
        mask_coil = h5file['mask_coil'][:]  # [8, 225, 224]
        if mask_coil.ndim == 2:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [225, 224, nslice]
            mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 225, 224, nslice]
        elif mask_coil.ndim == 3:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [8, 225, 224, nslice]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Testing_mask_coil = mask_coil * (1 + 0j)  # Complex

        #  Normalization
        # factor_norm = np.amax(np.abs(Testing_k_space), axis=(-3, -2, -1), keepdims=True)
        factor_norm = np.amax(np.abs(Testing_image), axis=(-3, -2, -1), keepdims=True)
        # factor_norm = 1  # No normalization

        Testing_inputs_Norm = Testing_k_space / factor_norm  # k-space input: [1, 224, 225, 8]

        return Testing_inputs_Norm, Testing_mask_coil, factor_norm


def loaddata_Test_stdnorm(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:

        #  input k-space
        #  Complex
        Testing_k_space = h5file['k_input'][:]  # [8, 225, 224]
        Testing_k_space = Testing_k_space['real'] + Testing_k_space['imag'] * 1j
        if Testing_k_space.ndim == 2:
            Testing_k_space = np.expand_dims(Testing_k_space, axis=-1)  # [225, 224, nslice]
            Testing_k_space = np.expand_dims(Testing_k_space, axis=0)  # [1, 225, 224, nslice]
        elif Testing_k_space.ndim == 3:
            Testing_k_space = np.expand_dims(Testing_k_space, axis=-1)  # [8, 225, 224, nslice]
        Testing_k_space = np.transpose(Testing_k_space.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        # Testing_image = np_ifft2c(Testing_k_space)

        _, kx, ky, ncoil = Testing_k_space.shape

        #  mask coil
        #  Float
        mask_coil = h5file['mask_coil'][:]  # [8, 225, 224]
        if mask_coil.ndim == 2:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [225, 224, nslice]
            mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 225, 224, nslice]
        elif mask_coil.ndim == 3:
            mask_coil = np.expand_dims(mask_coil, axis=-1)  # [8, 225, 224, nslice]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]
        Testing_mask_coil = mask_coil * (1 + 0j)  # Complex

        #  Normalization
        factor_norm = np.std(Testing_k_space, axis=(-3, -2, -1), ddof=1)
        # factor_norm_i = np.amax(np_sos(Testing_image), axis=(-2, -1))
        # factor_norm_i = 1  # No normalization

        Testing_inputs_Norm = Testing_k_space / factor_norm  # k-space input: [1, 224, 225, 8]

        return Testing_inputs_Norm, Testing_mask_coil, factor_norm


def loaddata_CSM(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'CSM_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:

        #  CSM
        #  Complex
        CSM = h5file['CSM'][:]  # [8, 225, 224]
        CSM = CSM['real'] + CSM['imag'] * 1j

        CSM = np.expand_dims(CSM, axis=-1)  # [8, 225, 224, nslice]
        CSM = np.transpose(CSM.astype(np.complex64), (3, 2, 1, 0))  # [nslice, 224, 225, 8]

        return CSM

