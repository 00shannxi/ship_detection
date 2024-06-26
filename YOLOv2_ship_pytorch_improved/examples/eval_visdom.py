import os
import argparse
import logging as log
import time
from statistics import mean
import numpy as np
import torch
from torchvision import transforms as tf
from pprint import pformat

import sys
sys.path.insert(0, '.')

import brambox.boxes as bbb
import vedanet as vn
from utils.envs import initEnv
#可视化检测框并保存图像

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    parser.add_argument('model_name', help='model name', default=None)
    args = parser.parse_args()

    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name=args.model_name)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)

    # init and run eng
    t1 = time.time()
    eng_t = vn.engine.VOCTest_visdom(hyper_params)
    t2 = time.time()


    log.info(f'\n{t2-t1} seconds ')

