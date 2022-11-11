# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 16:18
# @Author  : keevinzha
# @File    : options.py

import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--image_channel', type=int, default=2, help='inchannel, img+real')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=5)
    parser.add_argument('')

def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--')

def parse_test_args(parser):
    parser = parse_common_args(parser)


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args

def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type)

if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()