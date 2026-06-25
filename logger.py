#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
import time
import sys
import logging
import os  # 新增：用于创建目录

import torch.distributed as dist


def setup_logger(logpth):
    # 关键修复1：确保日志目录存在，不存在则创建
    if not osp.exists(logpth):
        os.makedirs(logpth, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    
    # 生成日志文件名（保持原有逻辑）
    logfile = 'MACTFusion-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    
    # 关键修复2：判断分布式是否初始化，避免未初始化时报错
    if dist.is_available() and dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    
    # 配置日志（保持原有逻辑）
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())