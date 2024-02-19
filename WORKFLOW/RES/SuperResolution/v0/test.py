# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 17:46
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing
from MODELALG.RES.MPRNet.MPRNetv0 import utils
from MODELALG.RES.MPRNet.MPRNetv0.data_RGB import get_validation_data
from MODELALG.RES.MPRNet.MPRNetv0.MPRNet import MPRNet

torch.multiprocessing.set_sharing_strategy("file_system")

opt = yaml.load(open(cur_dir + "/config/test.yml", "r", encoding="utf-8").read())
gpus = ",".join([str(i) for i in opt["GPU"]])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
torch.backends.cudnn.benchmark = True

mode = opt["MODEL"]["MODE"]
session = opt["MODEL"]["SESSION"]

val_dir = opt["TEST"]["TEST_DIR"]

# Model
model_restoration = MPRNet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# Resume
utils.load_checkpoint(model_restoration, opt["TEST"]["RESUME"])

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

val_dataset = get_validation_data(val_dir, {"patch_size": opt["TEST"]["TEST_PS"]})
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=opt["TEST"]["BATCH_SIZE"],
    shuffle=False,
    num_workers=8,
    drop_last=False,
    pin_memory=True,
)

print("===> Loading datasets")

model_restoration.eval()
psnr_val_rgb = []
for ii, data_val in enumerate(tqdm(val_loader), 0):
    target = data_val[0].cuda()
    input_ = data_val[1].cuda()
    with torch.no_grad():
        restored = model_restoration(input_)
    restored = restored[0]
    for res, tar in zip(restored, target):
        psnr_val_rgb.append(utils.torchPSNR(res, tar))
psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
print("[PSNR: %.4f]" % (psnr_val_rgb))
