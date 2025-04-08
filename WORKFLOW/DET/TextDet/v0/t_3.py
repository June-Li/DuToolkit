# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 20:47
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/DB/DBv0/")))

import cv2
import time
import traceback
import torch
import numpy as np
from torchvision import transforms
from MODELALG.DET.DB.DBv0.utils import draw_bbox
from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.networks import build_model
from MODELALG.DET.DB.DBv0.datasets.det_modules import ResizeShortSize, ResizeFixedSize
from MODELALG.DET.DB.DBv0.postprocess import build_post_process
from MODELALG.utils.common import Log
from MODELALG.utils import common


logger = Log(__name__).get_logger()


class Detector:
    def __init__(
        self,
        model_path,
        expand_pix=2,
        post_process_num_works=4,
        device="0",
        half_flag=False,
    ):
        self.expand_pix = expand_pix
        self.device = device
        self.half_flag = half_flag
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        cfg["post_process"]["num_works"] = post_process_num_works
        self.model = build_model(cfg["model"])
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        self.model.load_state_dict(state_dict)

        self.device = select_device(self.device)
        self.model.to(self.device)
        self.model.eval()
        if self.half_flag and self.device.type != "cpu":
            self.model.half()
        self.resize = ResizeFixedSize(736, False)
        cfg["post_process"]["type"] = "DBPostProcessV3"
        self.post_process = build_post_process(cfg["post_process"])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg["dataset"]["train"]["dataset"]["mean"],
                    std=cfg["dataset"]["train"]["dataset"]["std"],
                ),
            ]
        )
        logger.info(" ···-> load model succeeded!")

    def inference(self, img):
        """
        该接口用来检测文本行；
        :param img: opencv读取格式图片
        :return box_array: 文本行的box
                eg：
                    [[[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
                     ……]
        :return score_array: 文本行置信度得分
                eg：
                    [0.951253, ……]
        """
        try:
            # 预处理根据训练来
            data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
            data = self.resize(data)
            tensor = self.transform(data["img"])
            tensor = tensor.unsqueeze(dim=0)
            tensor = tensor.to(self.device)
            if self.half_flag and self.device.type != "cpu":
                tensor = tensor.half()
            starttime = time.time()
            with torch.no_grad():
                out = self.model(tensor)
            # print("模型推理耗时: ", time.time() - starttime)
            starttime = time.time()
            out = out.cpu().numpy()
            # print("*" * 30 + "结果放到cpu耗时: ", time.time() - starttime)
            if self.half_flag and self.device.type != "cpu":
                out = out.astype(np.float32)
            starttime = time.time()
            box_array, score_array = self.post_process(out, data["shape"])
            # print("后处理耗时: ", time.time() - starttime)
            box_array, score_array = box_array[0], score_array[0]
            if len(box_array) > 0:
                idx = [x.sum() > 0 for x in box_array]
                box_array = [box_array[i] for i, v in enumerate(idx) if v]
                score_array = [score_array[i] for i, v in enumerate(idx) if v]
            else:
                box_array, score_array = [], []
            box_array, score_array = np.array(box_array), np.array(score_array)

            h, w, _ = np.shape(img)
            box_array = np.array(
                [
                    [
                        [
                            max(box[0][0] - self.expand_pix, 0),
                            max(box[0][1] - self.expand_pix, 0),
                        ],
                        [
                            min(box[1][0] + self.expand_pix, w - 1),
                            max(box[1][1] - self.expand_pix, 0),
                        ],
                        [
                            min(box[2][0] + self.expand_pix, w - 1),
                            min(box[2][1] + self.expand_pix, h - 1),
                        ],
                        [
                            max(box[3][0] - self.expand_pix, 0),
                            min(box[3][1] + self.expand_pix, h - 1),
                        ],
                    ]
                    for box in box_array
                ]
            )

            return box_array, score_array
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python detector_text.py
    path = "/volume/test_data/my_imgs_0/38.jpg"
    # path = "/volume/test_data/fake/0.jpg"
    img = cv2.imread(path)
    # model = Detector(os.path.abspath(root_dir + '/MODEL/DET/DB/DBv0/TextDet/20210601/best.pt'), gpu='0')
    model = Detector(
        os.path.abspath("/volume/weights/Detector_text_model.pt"),
        device="0",
    )
    starttime = time.time()
    box_array, score_array = model.inference(img)
    print("80-t3-HiJuneLi文本检测总耗时: ", time.time() - starttime)
    # while True:
    #     starttime = time.time()
    #     box_array, score_array = model.inference(img)
    #     print("80-t3-HiJuneLi文本检测总耗时: ", time.time() - starttime)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_bbox(img, box_array)
    print()

    # out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # cv2.imwrite(path.replace("/test_data/", "/test_out/"), img)
