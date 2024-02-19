# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 20:47
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/DB/DBv0/")))

import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from MODELALG.DET.DB.DBv0.utils import draw_bbox
from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.networks import build_model
from MODELALG.DET.DB.DBv0.datasets.det_modules import ResizeShortSize, ResizeFixedSize
from MODELALG.DET.DB.DBv0.postprocess import build_post_process
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook


class Detector:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.model = build_model(cfg["model"])

        # writer = SummaryWriter('./data/tensorboard')
        # writer.add_graph(self.model, input_to_model=torch.rand(1, 3, 512, 512))
        # writer.close()
        print(notebook.list())
        # notebook.start("--logdir ./data/tensorboard")


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python detector_text.py
    model = Detector("/volume/weights/Detector_text_model.pt")
