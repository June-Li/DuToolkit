# -*- coding: utf-8 -*-
# @Time    : 2024/10/17
# @Author  : lijun
import copy
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "WORKFLOW/OTHER/Pix2Text/")))

import re
import threading
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image

from MODELALG.utils.common import Log, draw_poly_boxes
from WORKFLOW.OTHER.Pix2Text.pix2text.formula_detector import MathFormulaDetector
from WORKFLOW.OTHER.Pix2Text.pix2text.text_formula_ocr import y_overlap

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Detector:
    def __init__(
        self,
        model_path="/volume/weights/pix2img_model/1.1/mfd-onnx/mfd-v20240618.onnx",
        resized_shape=768,
        device="cuda:0",
        half_flag=False,
        conf_thres=0.65,
        batch_size=1,
    ):
        with torch.no_grad():
            self.resized_shape = resized_shape
            self.half_flag = half_flag
            self.weights = model_path
            self.device = device
            self.conf_thres = conf_thres
            self.batch_size = batch_size
            self.mfd_model = MathFormulaDetector(
                model_path=self.weights,
                device=self.device,
            )
            # self.half = "cpu" not in self.device
            # if self.half_flag and self.half:
            #     self.mfd_model.half()  # to FP16

    def __call__(self, imgs):
        """
        input:
            imgs: opencv读取的图片格式;
        Returns:
             boxes_list: [[[x0, y0], [x1, y1], ……], [……], ……]
             scores: [0.87, 0.75, ……]
        """
        try:
            with lock:
                boxes_list, scores_list = [], []
                for img_cv in imgs:
                    img = cv2.cvtColor(copy.deepcopy(img_cv), cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    w, h = img.size
                    ratio = self.resized_shape / w
                    resized_shape = (int(h * ratio), self.resized_shape)  # (H, W)
                    analyzer_outs = self.mfd_model(
                        img.copy(), resized_shape=resized_shape
                    )
                    boxes, scores = [], []
                    for elem in analyzer_outs:
                        if (
                            elem["type"] == "isolated"
                            and elem["score"] > self.conf_thres
                        ):
                            boxes.append(np.array(elem["box"], dtype=int).tolist())
                            scores.append(elem["score"])
                    # img_cv = draw_poly_boxes(img_cv, boxes)
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                return boxes_list, scores_list
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    detector = Detector()
    path = "/volume/test_data/多场景数据测试/formula-img/"
    image_name_list = os.listdir(path)
    for image_name in image_name_list:
        if image_name != "formula-03.png":
            continue
        img_ori = cv2.imread(path + image_name)

        start = time.time()
        boxes_list, _ = detector([img_ori])
        print("per img use time: ", time.time() - start)
