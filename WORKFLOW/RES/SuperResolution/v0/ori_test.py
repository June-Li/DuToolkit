# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 15:18
# @Author  : lijun
import os
import sys
import io

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from MODELALG.RES.MPRNet.MPRNetv0 import utils

from MODELALG.RES.MPRNet.MPRNetv0.data_RGB import get_test_data
from MODELALG.RES.MPRNet.MPRNetv0.MPRNet import MPRNet
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description="Image Deraining using MPRNet")
parser.add_argument(
    "--input_dir",
    default=cur_dir + "/test_data/my_imgs_0/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir",
    default=cur_dir + "/test_out/my_imgs_0/",
    type=str,
    help="Directory for results",
)
parser.add_argument(
    "--weights",
    default=root_dir + "/MODEL/RES/MPRNet/MPRNetv0/DeSeal/20211122/best.pt",
    type=str,
    help="Path to weights",
)
parser.add_argument("--gpus", default="1", type=str, help="CUDA_VISIBLE_DEVICES")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

test_dataset = get_test_data(args.input_dir, img_options={})
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True,
)

result_dir = os.path.join(args.result_dir)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # if ii % 10 != 0:
        #     continue
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_ = data_test[0].cuda()
        print(type(input_), input_.size())
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored[0], 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img(
                (os.path.join(result_dir, filenames[batch] + ".jpg")), restored_img
            )
