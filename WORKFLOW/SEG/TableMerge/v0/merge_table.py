# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 15:57
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(
    os.path.abspath(os.path.join(root_dir, "MODELALG/SEG/SPLERGEMERGE/SPLERGEMERGEv0/"))
)

import time
import numpy as np
import cv2
import pickle
import torch

from MODELALG.SEG.SPLERGEMERGE.SPLERGEMERGEv0.libs.dataloader import MergeTableDataset
from MODELALG.SEG.SPLERGEMERGE.SPLERGEMERGEv0.libs.model import MergeModel
from MODELALG.SEG.SPLERGEMERGE.SPLERGEMERGEv0.libs import utils


class mergeor:
    def __init__(self, model_path, device="0", half_flag=False):
        self.weights = model_path
        self.device = utils.select_device(device)
        self.half_flag = half_flag
        self.model = MergeModel().to(self.device)
        self.model.load_state_dict(
            torch.load(self.weights, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA
        if self.half_flag and self.half:
            self.model.half()

    def inference(self, img_ori, ccboxes, row_col_index):
        img = MergeTableDataset.make_mask_img(img_ori, ccboxes)
        img, row_col_index = MergeTableDataset.resize(img, row_col_index)
        mask_row, mask_col, mask_gird = MergeTableDataset.make_binary_img(
            img, row_col_index
        )
        img = MergeTableDataset.transform(img)
        input_feature = np.concatenate(
            [img, np.stack([mask_row, mask_col, mask_gird], axis=0)], axis=0
        )
        input_feature = torch.tensor(input_feature, dtype=torch.float32).to(self.device)
        input_feature = torch.unsqueeze(input_feature, dim=0)
        D1_probs, D2_probs, R1_probs, R2_probs = self.model(input_feature)

        th = 0.5
        [D2, R2] = map(
            lambda elem: elem.cpu().detach().numpy().squeeze(axis=0).squeeze(axis=0),
            [D2_probs, R2_probs],
        )
        D2 = np.array(
            (D2 * ((np.max(D2, axis=0, keepdims=True) - D2) < 0.2)) > th, dtype=int
        )
        R2 = np.array(
            (R2 * ((np.max(R2, axis=1, keepdims=True) - R2) < 0.2)) > th, dtype=int
        )
        return D2, R2


if __name__ == "__main__":
    from tqdm import tqdm

    input_path = cur_dir + "/test_data/HKYJ/"
    output_path = input_path.replace("/test_data/", "/test_out/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    table_mergeor = mergeor(
        model_path=root_dir
        + "/MODEL/SEG/SPLERGEMERGE/SPLERGEMERGEv0/TableMerge/20220706/merge_model_latest.pth"
    )
    filename_list = os.listdir(input_path + "images/")
    for filename in tqdm(filename_list):
        filename_prefix = filename[::-1].split(".", 1)[-1][::-1]
        img = cv2.imread(input_path + "images/" + filename)

        ccboxes = []
        lines = open(
            input_path + "hypertxt/" + filename_prefix + ".txt", "r"
        ).readlines()
        for line in lines:
            box = eval(line.split("|")[-1].strip())
            ccboxes.append(box)

        seg = pickle.load(open(input_path + "seg/" + filename_prefix + ".pkl", "rb"))

        D2, R2 = table_mergeor.inference(img, ccboxes, seg)

        pickle.dump(D2, open(output_path + filename_prefix + "_D2.pkl", "wb"))
        pickle.dump(R2, open(output_path + filename_prefix + "_R2.pkl", "wb"))
