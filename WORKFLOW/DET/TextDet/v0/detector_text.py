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
import copy
import time
import traceback
import torch
import numpy as np
from torchvision import transforms

# import threading

from MODELALG.DET.DB.DBv0.utils import draw_bbox

# from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.networks import build_model
from MODELALG.DET.DB.DBv0.datasets.det_modules import ResizeShortSize, ResizeFixedSize
from MODELALG.DET.DB.DBv0.postprocess import build_post_process
from MODELALG.utils import common
from MODELALG.utils.common import Log, select_device, draw_poly_boxes


logger = Log(__name__).get_logger()
# lock = threading.Lock()


class Detector:
    def __init__(
        self,
        model_path,
        device="cuda:0",
        half_flag=False,
        unclip_ratio=2.0,
        expand_left_radio=0.0,
        expand_right_radio=0.0,
    ):
        self.device = device
        self.half_flag = half_flag
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        cfg["post_process"]["box_thresh"] = 0.2
        cfg["post_process"]["unclip_ratio"] = unclip_ratio
        cfg["post_process"]["expand_left_radio"] = expand_left_radio
        cfg["post_process"]["expand_right_radio"] = expand_right_radio
        self.cfg = cfg
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

    def __call__(self, img):
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
            # with lock:
            # 预处理根据训练来
            # total_time = time.time()
            data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
            data = self.resize(data)
            tensor = self.transform(data["img"])
            tensor = tensor.unsqueeze(dim=0)
            tensor = tensor.to(self.device)
            if self.half_flag and self.device.type != "cpu":
                tensor = tensor.half()
            # starttime = time.time()
            with torch.no_grad():
                out = self.model(tensor)
                # traced_script_module = torch.jit.trace(self.model, tensor)
                # traced_script_module.save("/workspace/JuneLi/a.pt")
            # print("模型推理耗时: ", time.time() - starttime)
            # starttime = time.time()
            out = out.cpu().numpy()
            # print("结果放到cpu耗时：", time.time() - starttime)
            # starttime = time.time()
            if self.half_flag and self.device.type != "cpu":
                out = out.astype(np.float32)
            # print("结果类型转换耗时: ", time.time() - starttime)
            # starttime = time.time()
            radio = [
                tensor.shape[2] / img.shape[0],
                tensor.shape[3] / img.shape[1],
            ]
            post_process_out = self.post_process(
                {"res": out},
                [
                    -1,
                    [list(tensor.shape[2:]) + radio],
                ],
            )
            # print("后处理耗时: ", time.time() - starttime)
            box_array, score_array = (
                post_process_out[0]["points"],
                post_process_out[0]["scores"],
            )
            if len(box_array) > 0:
                idx = [x.sum() > 0 for x in box_array]
                box_array = [box_array[i] for i, v in enumerate(idx) if v]
                score_array = [score_array[i] for i, v in enumerate(idx) if v]

                box_array = np.array(box_array, dtype=float)
                box_array[:, :, 0] /= radio[1]
                box_array[:, :, 1] /= radio[0]
                box_array = np.array(box_array, dtype=int).tolist()
            else:
                box_array, score_array = [], []

            # print(
            #     "DB总耗时：",
            #     time.time() - total_time,
            # )

            # for idx, box in enumerate(box_array):
            #     h = box[3][1] - box[0][1]
            #     box_array[idx][0][0] = max(
            #         box_array[idx][0][0]
            #         - int(h * self.cfg["post_process"]["expand_left_radio"]),
            #         0,
            #     )
            #     box_array[idx][1][0] = min(
            #         box_array[idx][1][0]
            #         + int(h * self.cfg["post_process"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][2][0] = min(
            #         box_array[idx][2][0]
            #         + int(h * self.cfg["post_process"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][3][0] = max(
            #         box_array[idx][3][0]
            #         - int(h * self.cfg["post_process"]["expand_left_radio"]),
            #         0,
            #     )
            step = 1
            for idx, box in enumerate(box_array):
                h = box[3][1] - box[0][1]
                # 移动右边界
                offset_right = 1
                for osr in range(
                    offset_right,
                    int(h * self.cfg["post_process"]["expand_right_radio"]),
                    step,
                ):
                    if (box[1][0] + osr) >= img.shape[1]:
                        offset_right = img.shape[1] - box[1][0] - 1
                        break
                    if np.var(img[box[0][1] : box[-1][1], box[1][0] + osr]) > 50:
                        continue
                    offset_right = osr
                    break
                if offset_right > 0:
                    box_array[idx][1][0] = min(
                        box_array[idx][1][0] + offset_right,
                        img.shape[1],
                    )
                    box_array[idx][2][0] = min(
                        box_array[idx][2][0] + offset_right,
                        img.shape[1],
                    )
                # 移动左边界
                offset_left = 0
                for osl in range(
                    offset_left,
                    int(h * self.cfg["post_process"]["expand_left_radio"]),
                    step,
                ):
                    if (box[0][0] - osl) < 0:
                        offset_left = box[0][0]
                        break
                    if np.var(img[box[0][1] : box[-1][1], box[0][0] - osl]) > 100:
                        continue
                    offset_left = osl
                    break
                if offset_left > 0:
                    box_array[idx][0][0] = max(
                        box_array[idx][0][0] - offset_left,
                        0,
                    )
                    box_array[idx][3][0] = max(
                        box_array[idx][3][0] - offset_left,
                        0,
                    )
            return box_array, score_array
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python detector_text.py
    path = "/volume/test_data/my_imgs_0/53.jpg"
    # path = "/volume/test_data/fake/0.jpg"
    img = cv2.imread(path)
    # model = Detector(os.path.abspath(root_dir + '/MODEL/DET/DB/DBv0/TextDet/20210601/best.pt'), gpu='0')
    model = Detector(
        os.path.abspath("/volume/weights/Detector_text_model.pt"),
        device="cuda:0",
        unclip_ratio=2.0,
        expand_left_radio=0.2,
        expand_right_radio=0.7,
    )
    starttime = time.time()
    box_array, score_array = model(img)
    print("80-HiJuneLi文本检测总耗时: ", time.time() - starttime)
    # while True:
    #     starttime = time.time()
    #     box_array, score_array = model(img)
    #     print("80-HiJuneLi文本检测总耗时: ", time.time() - starttime)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = common.draw_poly_boxes(img, box_array)
    print()

    # out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # cv2.imwrite(path.replace("/test_data/", "/test_out/"), img)
