# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 15:18
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import time
import cv2
import PIL as PI
from PIL import Image
import torch
from skimage import img_as_ubyte
import torchvision.transforms.functional as TF
from MODELALG.RES.MPRNet.MPRNetv0 import utils
from MODELALG.RES.MPRNet.MPRNetv0.MPRNet import MPRNet
from MODELALG.RES.MPRNet.MPRNetv0.utils.torch_utils import select_device


class De:
    def __init__(self, model_path, img_size=512, batch_size=4, gpu="0"):
        self.weights = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.gpu = gpu
        self.device = select_device(self.gpu)
        self.model = MPRNet()
        utils.load_checkpoint(self.model, self.weights, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.half = (
            False  # self.device.type != 'cpu'  # half precision only supported on CUDA
        )
        if self.half:
            self.model.half()  # to FP16

    def inference(self, imgs_ori):
        """
        去印章模型
        :param imgs_ori: 抠出来的印章，cv的BGR格式
        :return: 去印章后的图，cv的BGR格式
        """
        imgs = []
        for img in imgs_ori:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PI.Image.fromarray(img)
            img = TF.to_tensor(img).to(self.device)
            if self.half:
                img = img.half()
            imgs.append(img)

        out_imgs = []
        idxs = list(range(len(imgs)))
        for idx in range(0, len(imgs), self.batch_size):
            # start = time.time()
            batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
            batch_imgs = [imgs[idx_] for idx_ in batch_idxs]
            batch_imgs = torch.stack(batch_imgs)
            with torch.no_grad():
                pred = self.model(batch_imgs)
                pred = torch.clamp(pred[0], 0, 1)
                pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()
                for o_img in pred:
                    out_imgs.append(
                        cv2.cvtColor(img_as_ubyte(o_img), cv2.COLOR_RGB2BGR)
                    )
            # print('per batch use time: ', time.time()-start)
        return out_imgs


if __name__ == "__main__":
    base_path = cur_dir + "/test_data/experiment/"
    img_path_list = os.listdir(base_path)
    img_list = []
    for img_path in img_path_list:
        img_list.append(cv2.imread(base_path + img_path))

    R = De(
        root_dir + "/MODEL/RES/MPRNet/MPRNetv0/SuperResolution/20211126/best.pt",
        img_size=512,
        batch_size=1,
    )
    out = R.inference(img_list)
    for i, img in enumerate(out):
        cv2.imwrite(
            cur_dir + "/test_out/experiment/" + img_path_list[i].split("/")[-1], img
        )
