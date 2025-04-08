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
from ultralytics import YOLO

from MODELALG.utils.common import Log, draw_boxes, draw_poly_boxes
from WORKFLOW.OTHER.Pix2Text.pix2text.formula_detector import MathFormulaDetector
from WORKFLOW.OTHER.Pix2Text.pix2text.text_formula_ocr import y_overlap

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Detector:
    def __init__(
        self,
        model_path="/volume/weights/Detector_formula_model.pt",  # mineru的公式检测模型
        resized_shape=1888,
        device="cuda:0",
        half_flag=False,
        conf_thres=0.25,
        batch_size=16,
    ):
        with torch.no_grad():
            self.half_flag = half_flag
            self.weights = model_path
            self.device = device
            self.conf_thres = conf_thres
            self.batch_size = batch_size
            self.resized_shape = resized_shape
            self.mfd_model = YOLO(self.weights).to(self.device)
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
                for index in range(0, len(imgs), self.batch_size):
                    mfd_res = [
                        image_res.cpu()
                        for image_res in self.mfd_model.predict(
                            imgs[index : index + self.batch_size],
                            imgsz=self.resized_shape,
                            conf=self.conf_thres,
                            iou=0.45,
                            verbose=False,
                            device=self.device,
                        )
                    ]
                    for image_res in mfd_res:
                        box_list, score_list = [], []
                        for box in image_res.boxes:
                            if box.cls[0].tolist() == 0:  # 0为段落内嵌公式，1为独立公式
                                continue
                            f_b = np.array(box.xyxy[0], dtype=int).tolist()
                            box_list.append(
                                [
                                    [f_b[0], f_b[1]],
                                    [f_b[2], f_b[1]],
                                    [f_b[2], f_b[3]],
                                    [f_b[0], f_b[3]],
                                ]
                            )
                            score_list.append(box.conf[0].tolist())
                    boxes_list.append(box_list)
                    scores_list.append(score_list)
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
        show_img = copy.deepcopy(img_ori)
        show_img = draw_poly_boxes(show_img, np.array(boxes_list[0]))
        print("per img use time: ", time.time() - start)
