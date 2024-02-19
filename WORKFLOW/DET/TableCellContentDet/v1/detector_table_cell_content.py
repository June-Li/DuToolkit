# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 14:56
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/YOLO/YOLOv5/")))

import numpy as np
import cv2
import traceback
import torch
from MODELALG.DET.YOLO.YOLOv5.models.experimental import attempt_load
from MODELALG.DET.YOLO.YOLOv5.utils.datasets import letterbox
from MODELALG.DET.YOLO.YOLOv5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from MODELALG.DET.YOLO.YOLOv5.utils.torch_utils import select_device
from MODELALG.utils.common import Log


logger = Log(__name__).get_logger()


class Detector:
    def __init__(
        self,
        model_path,
        img_size=1280,
        augment=False,
        conf_thres=0.5,
        iou_thres=0.5,
        device="0",
        half_flag=False,
    ):
        """
        :param model_path:
        :param img_size:
        :param augment:
        :param conf_thres:
        :param iou_thres:
        :param gpu: gpu索引0，1，2，3，或者cpu
        """
        with torch.no_grad():
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.augment = augment
            self.half_flag = half_flag
            self.weights = model_path
            self.device = select_device(device)
            self.model = attempt_load(
                self.weights, map_location=self.device
            )  # load FP32 model
            self.stride = self.model.stride.max()
            self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size
            self.half = (
                self.device.type != "cpu"
            )  # half precision only supported on CUDA
            if self.half_flag and self.half:
                self.model.half()  # to FP16
            logger.info(" ···-> load model succeeded!")

    def inference(self, img_ori):
        """
        input:
            img_ori: opencv读取的图片格式;
        Returns:
             boxes: [[x0, y0, x1, y1], [……], ……]
             confes: [0.87, 0.75, ……]
             clses: [0, 0, ……]
        """
        try:
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
                            # boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                            boxes.append(
                                [
                                    max(int(xyxy[0]) - 3, 0),
                                    int(xyxy[1]),
                                    min(int(xyxy[2]) + 3, np.shape(img_ori)[1]),
                                    int(xyxy[3]),
                                ]
                            )
                            confes.append(float(conf.cpu().numpy()))
                            clses.append(int(cls.cpu().numpy()))
            return boxes, confes, clses
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    import time

    # CUDA_VISIBLE_DEVICES=1 python detector_table_cell.py
    detector = Detector(
        os.path.abspath("/volume/weights/Detector_table_cell_content_model.pt"),
        # os.path.abspath('/workspace/JuneLi/bbtv/SensedealImgAlg/MODEL/DET/YOLO/YOLOv5/TableCellContentDet/20220406/last.pt'),
        img_size=1280,
        conf_thres=0.5,
        iou_thres=0.3,
        device="0",
    )
    path = cur_dir + "/test_data/hkyj/"

    image_name_list = os.listdir(path)
    for image_name in image_name_list:
        img_ori = cv2.imread(path + image_name)
        start = time.time()
        boxes, _, _ = detector.inference(img_ori)
        print("use time: ", time.time() - start)

        for box in boxes:
            c1, c2 = (box[0], box[1]), (box[2], box[3])
            cv2.rectangle(
                img_ori, c1, c2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA
            )

        out_path = path.replace("/test_data/", "/test_out/")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(
            path.replace("/test_data/", "/test_out/") + "/" + image_name, img_ori
        )
        # cv2.imwrite(path.replace('/test_data/', '/test_out/') + '/' + image_name.replace('.jpg', '.png'), make_mask_img(img_ori, boxes))

        out_str = ""
        for box in boxes:
            # x_center = (box[0] + box[2]) / 2 / img_ori.shape[1]
            # y_center = (box[1] + box[3]) / 2 / img_ori.shape[0]
            # box_w = abs(box[0] - box[2]) / img_ori.shape[1]
            # box_h = abs(box[1] - box[3]) / img_ori.shape[0]
            # box = [0, x_center, y_center, box_w, box_h]
            out_str += " ".join([str(i) for i in box]) + "\n"

        open(
            path.replace("/test_data/", "/test_out/")
            + image_name.replace(".jpg", ".txt"),
            "w",
        ).write(out_str)
        # print(boxes)
