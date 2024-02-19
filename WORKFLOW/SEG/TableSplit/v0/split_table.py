# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 15:57
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(
    os.path.abspath(os.path.join(root_dir, "MODELALG/SEG/SPLERGESPLIT/SPLERGESPLITv0/"))
)

import cv2
import numpy as np
import torch
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs import utils
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.model import SplitModel


class Splitor:
    def __init__(self, model_path, device="0", half_flag=False):
        self.weights = model_path
        self.device = utils.select_device(device)
        self.half_flag = half_flag
        self.model = SplitModel(eval_mode=True).to(self.device)
        self.model.load_state_dict(
            torch.load(self.weights, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA
        if self.half_flag and self.half:
            self.model.half()

    def inference(self, img_ori):
        H, W, C = img_ori.shape
        image_trans = img_ori.transpose((2, 0, 1)).astype("float32")
        resized_image = utils.resize_image(image_trans)
        input_image = utils.normalize_numpy_image(resized_image).unsqueeze(0)
        rpn_out, cpn_out = self.model(input_image.to(self.device))

        rpn_image = utils.probs_to_image(
            rpn_out.detach().clone(), input_image.shape, 1
        ).cpu()
        cpn_image = utils.probs_to_image(
            cpn_out.detach().clone(), input_image.shape, 0
        ).cpu()

        row_indices, col_indices = utils.get_indices(
            rpn_image, cpn_image, H / input_image.shape[2]
        )
        return row_indices, col_indices


if __name__ == "__main__":
    from tqdm import tqdm

    input_path = cur_dir + "/test_data/my_imgs_4/"
    output_path = input_path.replace("/test_data/", "/test_out/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    table_splitor = Splitor(
        model_path=root_dir
        + "/MODEL/SEG/SPLERGESPLIT/SPLERGESPLITv0/TableSplit/20220715/best.pt"
    )
    # table_splitor = Splitor(model_path='/MODEL/SEG/SPLERGESPLIT/SPLERGESPLITv0/TableSplit/20220715/best.pt')
    # table_splitor = Splitor(model_path=root_dir + '/MODEL/SEG/SPLERGESPLIT/SPLERGESPLITv0/TableSplit/20220715/split_oriimg_model.pth')
    # table_splitor = Splitor(model_path='/workspace/JuneLi/bbtv/ExpCode/deep-splerge/checkpoints/HKYJ/split_model.pth')
    filename_list = os.listdir(input_path)
    for filename in tqdm(filename_list):
        img = cv2.imread(input_path + filename)
        row_indices, col_indices = table_splitor.inference(img)
        for idx_1, elem_1 in enumerate(row_indices[:-1]):
            for idx_2, elem_2 in enumerate(col_indices[:-1]):
                cv2.rectangle(
                    img,
                    (elem_2, elem_1),
                    (col_indices[idx_2 + 1], row_indices[idx_1 + 1]),
                    (0, 0, 255),
                    1,
                )
        cv2.imwrite(output_path + filename, img)
