# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 20:47
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/DB/DBv0/")))

import time
import numpy as np
from importlib import import_module
import torch
from torch import nn
from tqdm import tqdm
import argparse

from MODELALG.DET.DB.DBv0.networks import build_model
from MODELALG.DET.DB.DBv0.postprocess import build_post_process
from MODELALG.DET.DB.DBv0.datasets import build_dataloader
from MODELALG.DET.DB.DBv0.utils import weight_init, load_checkpoint
from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.metrics import DetMetric


def parse_cfg(args):
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


def test_op(net, val_loader, to_use_device, post_process):
    """
    在验证集上评估模型

    :param net: 网络
    :param val_loader: 验证集 dataloader
    :param to_use_device: device
    :param logger: logger类对象
    :param post_process: 后处理类对象
    :param metric: 根据网络输出和 label 计算 acc 等指标的类对象
    :return:  一个包含 eval_loss，eval_acc和 norm_edit_dis 的 dict,
        例子： {
                'recall':0,
                'precision': 0.99,
                'hmean': 0.9999,
                }
    """
    metric = DetMetric()
    net.eval()
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            start = time.time()
            output = net.forward(batch_data["img"].to(to_use_device))
            boxes, scores = post_process(
                output.cpu().numpy(), batch_data["shape"]
            )  # , is_output_polygon=metric.is_output_polygon

            # with torch.no_grad():
            #     out = net(batch_data['img'].to(to_use_device))
            # out = out.cpu().numpy()
            # boxes, scores = post_process(out, batch_data['shape'])

            total_frame += batch_data["img"].size()[0]
            total_time += time.time() - start
            raw_metric = metric(batch_data, (boxes, scores))
            raw_metrics.append(raw_metric)
    metrics = metric.gather_measure(raw_metrics)
    net.train()
    result_dict = {
        "recall": metrics["recall"].avg,
        "precision": metrics["precision"].avg,
        "hmean": metrics["fmeasure"].avg,
    }
    for k, v in result_dict.items():
        print(k, ": ", v)
    print("FPS: ", total_frame / total_time)

    return result_dict


def main(args):
    cfg = parse_cfg(args)
    to_use_device = select_device(cfg["test_options"]["device"])
    net = build_model(cfg["model"])
    if not cfg["model"]["backbone"]["pretrained"]:  # 使用 pretrained
        net.apply(weight_init)
    net = nn.DataParallel(net)
    net = net.to(to_use_device)
    net.eval()
    net, _resumed_optimizer, global_state = load_checkpoint(
        net,
        cfg["test_options"]["resume_from"],
        to_use_device,
        third_name=cfg["test_options"]["third_party_name"],
    )
    eval_loader = build_dataloader(cfg.dataset.eval)
    post_process = build_post_process(cfg["post_process"])
    test_op(net, eval_loader, to_use_device, post_process)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python test.py
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--config",
        type=str,
        default=cur_dir + "/config/det_test_db_config.py",
        help="train config file path",
    )
    args = parser.parse_args()
    main(args)
