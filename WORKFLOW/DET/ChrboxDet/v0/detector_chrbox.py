# -*- coding: utf-8 -*-
# @Time    : 2021/6/4 17:27
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/YOLO/YOLOv5/")))
import threading
import time
import traceback

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from MODELALG.utils.common import Log

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Detector:
    def __init__(
        self,
        model_path,
        img_size=960,
        conf_thres=0.1,
        iou_thres=0.7,
        batch_size=16,
        device="cuda:0",
        half_flag=False,
    ):
        with torch.no_grad():
            # Load a model
            self.model = YOLO(model_path)
            self.img_size = img_size
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.batch_size = batch_size
            self.device = device if ':' not in device else device.split(':')[-1]
            self.half_flag = half_flag

            logger.info(" ···-> load model succeeded!")

    def __call__(self, imgs):
        """
        input:
            img_ori: opencv读取的图片格式;
        Returns:
             boxes: [[x0, y0, x1, y1], [……], ……]
             confes: [0.87, 0.75, ……]
             clses: [0, 0, ……]
        """
        try:
            with lock:
                outs = []

                starttime = time.time()
                results = []
                for batch_idx in range(0, len(imgs), self.batch_size):
                    b_imgs = imgs[batch_idx : batch_idx + self.batch_size]
                    cut_idx = 0
                    if len(b_imgs) == 1:
                        cut_idx = 1
                        b_imgs = [np.ones((32, 100, 3), dtype=np.uint8) * 255] + b_imgs
                    for i in range(3):
                        try:
                            o = self.model(
                                b_imgs,
                                device=self.device,
                                imgsz=self.img_size,
                                batch=self.batch_size,
                                conf=self.conf_thres,
                                iou=self.iou_thres,
                                half=self.half_flag,
                                verbose=False,
                            )[cut_idx:]
                            break
                        except Exception as e:
                            pass
                    results.extend(o)
                logger.info(
                    " ···-> chrbox detector inference time: {}".format(
                        time.time() - starttime
                    )
                )

                for idx_0, result in enumerate(results):
                    boxes, confes, clses = [], [], []
                    for box_ori in result.boxes:
                        boxes.append(
                            np.array(box_ori.xyxy[0].tolist(), dtype=int).tolist()
                        )
                        confes.append(float(box_ori.conf[0]))
                        clses.append(0)
                    outs.append([boxes, confes, clses])

                return outs
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    detector = Detector("/volume/weights/Detector_chrbox_model.pt")
    path = "/volume/test_data/rec/"
    image_name_list = os.listdir(path)
    for image_name in image_name_list[:]:
        # if image_name != "15-37.jpg":
        #     continue
        img_ori = cv2.imread(path + image_name)
        start = time.time()
        boxes, confes, clses = detector([img_ori])[0]
        print("per img use time: ", time.time() - start)

        for box in boxes:
            c1, c2 = (box[0], box[1]), (box[2], box[3])
            cv2.rectangle(
                img_ori, c1, c2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA
            )
        print()
        # out_path = path.replace("/test_data/", "/test_out/")
        # if not os.path.exists(out_path):
        #     os.makedirs(out_path)
        # cv2.imwrite(path.replace("/test_data/", "/test_out/") + image_name, img_ori)
