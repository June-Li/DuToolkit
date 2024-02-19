# -*- coding: utf-8 -*-
# @Time    : 2021/6/4 17:27
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/YOLO/YOLOv3/")))

import numpy as np
import cv2
import torch
from MODELALG.DET.YOLO.YOLOv3.models.experimental import attempt_load
from MODELALG.DET.YOLO.YOLOv3.utils.datasets import letterbox
from MODELALG.DET.YOLO.YOLOv3.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from MODELALG.DET.YOLO.YOLOv3.utils.torch_utils import select_device


class Detector:
    def __init__(
        self,
        model_path,
        img_size=640,
        augment=False,
        conf_thres=0.7,
        iou_thres=0.5,
        gpu=True,
    ):
        with torch.no_grad():
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.augment = augment
            self.weights = model_path
            self.device = select_device("0" if gpu else "cpu")
            self.model = attempt_load(
                self.weights, map_location=self.device
            )  # load FP32 model
            self.imgsz = check_img_size(
                img_size, s=self.model.stride.max()
            )  # check img_size
            self.half = (
                self.device.type != "cpu"
            )  # half precision only supported on CUDA
            if self.half:
                self.model.half()  # to FP16

    def inference(self, img_ori):
        """
        input:
            img_ori: opencv读取的图片格式;
        Returns:
             boxes: [[x0, y0, x1, y1], [……], ……]
             confes: [0.87, 0.75, ……]
             clses: [0, 0, ……]
        """
        boxes = []
        confes = []
        clses = []
        with torch.no_grad():
            # prepare data
            img = letterbox(img_ori, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], img_ori.copy().shape
                    ).round()
                    for *xyxy, conf, cls in reversed(det):
                        boxes.append(
                            [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        )
                        confes.append(float(conf.cpu().numpy()))
                        clses.append(int(cls.cpu().numpy()))

        return boxes, confes, clses


if __name__ == "__main__":
    path = cur_dir + "/test_data/my_imgs_1/1.jpg"
    img_ori = cv2.imread(path)
    detector = Detector(
        os.path.abspath(root_dir + "/MODEL/DET/YOLO/YOLOv3/TableDet/20210531/best.pt")
    )
    boxes, _, _ = detector.inference(img_ori)

    for box in boxes:
        c1, c2 = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(img_ori, c1, c2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite(path.replace("/test_data/", "/test_out/"), img_ori)
    print(boxes)
