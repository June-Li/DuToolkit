# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 11:05
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/YOLO/SSLYOLOv5/")))

import time
import cv2
import traceback
import torch
import numpy as np
import threading

from MODELALG.DET.YOLO.SSLYOLOv5.models.experimental import attempt_load
from MODELALG.DET.YOLO.SSLYOLOv5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)

# from MODELALG.DET.YOLO.SSLYOLOv5.utils.torch_utils import select_device
from WORKFLOW.DET.SealDet.v2.utils.datasets_semi import letterbox
from MODELALG.utils.common import Log, select_device


logger = Log(__name__).get_logger()
lock = threading.Lock()

class Detector:
    def __init__(
        self,
        model_path,
        img_size=1280,
        augment=False,
        conf_thres=0.7,
        iou_thres=0.5,
        device="cuda:0",
        half_flag=False,
    ):
        with torch.no_grad():
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.augment = augment
            self.half_flag = half_flag
            self.weights = model_path
            self.device = select_device(device)
            self.model = attempt_load(
                self.weights, map_location="cpu"
            )  # load FP32 model
            self.model.to(self.device).float()
            self.stride = self.model.stride.max()
            self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size
            self.half = (
                self.device.type != "cpu"
            )  # half precision only supported on CUDA
            if self.half_flag and self.half:
                self.model.half()  # to FP16
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
                for img_ori in imgs:
                    boxes = []
                    confes = []
                    clses = []
                    with torch.no_grad():
                        # prepare data
                        img = letterbox(
                            img_ori, new_shape=self.imgsz, stride=self.stride.cpu().numpy()
                        )[0]
                        img = img[:, :, ::-1].transpose(2, 0, 1)
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device)
                        img = (
                            img.half() if self.half_flag and self.half else img.float()
                        )  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        # Inference
                        pred = self.model(img)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            if len(det):
                                det[:, :4] = scale_coords(
                                    img.shape[2:], det[:, :4], img_ori.shape
                                ).round()
                                for *xyxy, conf, cls in reversed(det):
                                    boxes.append(
                                        [
                                            int(xyxy[0]),
                                            int(xyxy[1]),
                                            int(xyxy[2]),
                                            int(xyxy[3]),
                                        ]
                                    )
                                    confes.append(float(conf.cpu().numpy()))
                                    clses.append(int(cls.cpu().numpy()))
                    outs.append([boxes, confes, clses])
                return outs
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    detector = Detector(
        os.path.abspath(
            root_dir
            + "/MODEL/DET/YOLO/SSLYOLOv5/SealDet/20210720/runs/train/exp4/weights/best.pt"
        ),
        img_size=1280,
        conf_thres=0.1,
        gpu="2",
    )
    path = cur_dir + "/test_data/my_imgs_0/"
    image_name_list = os.listdir(path)
    for image_name in image_name_list:
        img_ori = cv2.imread(path + image_name)

        # expand = 500
        # bg_value = int(np.argmax(np.bincount(img_ori.flatten(order='C'))))
        # ori_h, ori_w = np.shape(img_ori)[0], np.shape(img_ori)[1]
        # expand_image = np.ones((ori_h + expand, ori_w + expand, 3), dtype=np.uint8) * bg_value
        # expand_image[expand // 2:expand // 2 + ori_h, expand // 2:expand // 2 + ori_w] = img_ori
        # img_ori = expand_image

        start = time.time()
        boxes, confes, _ = detector(img_ori)[0]
        print("per img use time: ", time.time() - start)

        for box, conf in zip(boxes, confes):
            c1, c2 = (box[0], box[1]), (box[2], box[3])
            cv2.rectangle(
                img_ori, c1, c2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA
            )
            cv2.putText(
                img_ori,
                str(round(conf, 3)),
                c1,
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
            )

        out_path = path.replace("/test_data/", "/test_out/")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(path.replace("/test_data/", "/test_out/") + image_name, img_ori)
