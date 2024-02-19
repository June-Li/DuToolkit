# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 20:47
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/DB/DBv0/")))

import random
import time
import shutil
import traceback
from importlib import import_module
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from MODELALG.DET.DB.DBv0.networks import build_model, build_loss
from MODELALG.DET.DB.DBv0.postprocess import build_post_process
from MODELALG.DET.DB.DBv0.datasets import build_dataloader
from MODELALG.DET.DB.DBv0.utils import (
    get_logger,
    weight_init,
    load_checkpoint,
    save_checkpoint,
)
from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.metrics import DetMetric


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="val")
    parser.add_argument(
        "--config",
        type=str,
        default="config/det_test_db_config.py",
        help="train config file path",
    )
    args = parser.parse_args()
    # 解析.py文件
    config_path = os.path.abspath(os.path.expanduser(args.config))
    assert os.path.isfile(config_path)
    if config_path.endswith(".py"):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        return mod.config
    else:
        raise IOError("Only py type are supported now!")


def evaluate(net, val_loader, to_use_device, post_process, metric, half_flag=True):
    net.eval()
    if half_flag:
        net.half()
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            start = time.time()
            if half_flag:
                output = net.forward(batch_data["img"].to(to_use_device).half())
            else:
                output = net.forward(batch_data["img"].to(to_use_device))
            boxes, scores = post_process(output.cpu().numpy(), batch_data["shape"])
            total_frame += batch_data["img"].size()[0]
            total_time += time.time() - start
            raw_metric = metric(batch_data, (boxes, scores))
            raw_metrics.append(raw_metric)
    metrics = metric.gather_measure(raw_metrics)
    result_dict = {
        "recall": metrics["recall"].avg,
        "precision": metrics["precision"].avg,
        "hmean": metrics["fmeasure"].avg,
    }
    return result_dict


def main():
    # ===> 获取配置文件参数
    cfg = parse_args()
    val_options = cfg.val_options
    to_use_device = torch.device(
        val_options["device"]
        if torch.cuda.is_available() and ("cuda" in val_options["device"])
        else "cpu"
    )

    # ===> build network
    net = build_model(cfg["model"])
    net = net.to(to_use_device)
    net.eval()

    # ===> resume from checkpoint
    resume_from = val_options["resume_from"]
    net, _, _ = load_checkpoint(net, resume_from, to_use_device)
    net = nn.DataParallel(net)

    # ===> data loader
    eval_loader = build_dataloader(cfg.dataset.eval)

    # ===> post_process
    post_process = build_post_process(cfg["post_process"])

    # ===> val
    metric = DetMetric()
    eval_dict = evaluate(
        net, eval_loader, to_use_device, post_process, metric, half_flag=False
    )
    print(eval_dict)


if __name__ == "__main__":
    main()
