# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 20:47
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/DET/DB/DBv0/")))

import torch
import cv2
import numpy as np
from importlib import import_module
from torchvision import transforms
from MODELALG.DET.DB.DBv0.utils import draw_bbox
from MODELALG.DET.DB.DBv0.utils.torch_utils import select_device
from MODELALG.DET.DB.DBv0.networks import build_model
from MODELALG.DET.DB.DBv0.datasets.det_modules import ResizeShortSize, ResizeFixedSize
from MODELALG.DET.DB.DBv0.postprocess import build_post_process


def parse_cfg(config_path):
    config_path = os.path.abspath(os.path.expanduser(config_path))
    assert os.path.isfile(config_path)
    if config_path.endswith(".py"):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        return mod.config
    else:
        raise IOError("Only py type are supported now!")


class Detector:
    def __init__(self, model_path, expand_pix=2, gpu="0"):
        self.expand_pix = expand_pix
        self.gpu = gpu
        ckpt = torch.load(model_path, map_location="cpu")
        # cfg = ckpt['cfg']
        cfg = parse_cfg(cur_dir + "/config/det_train_db_config.py")
        self.model = build_model(cfg["model"])
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        self.model.load_state_dict(state_dict)

        self.device = select_device(self.gpu)
        self.model.to(self.device)
        self.model.eval()
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
        # 预处理根据训练来
        data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
        data = self.resize(data)
        tensor = self.transform(data["img"])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.cpu().numpy()
        box_array, score_array = self.post_process(out, data["shape"])
        box_array, score_array = box_array[0], score_array[0]
        if len(box_array) > 0:
            idx = [x.sum() > 0 for x in box_array]
            box_array = [box_array[i] for i, v in enumerate(idx) if v]
            score_array = [score_array[i] for i, v in enumerate(idx) if v]
        else:
            box_array, score_array = [], []
        box_array, score_array = np.array(box_array), np.array(score_array)

        h, w, _ = np.shape(img)
        # box_array = np.array([[[max(box[0][0] - self.expand_pix, 0), max(box[0][1] - self.expand_pix, 0)],
        #                        [min(box[1][0] + self.expand_pix, w - 1), max(box[1][1] - self.expand_pix, 0)],
        #                        [min(box[2][0] + self.expand_pix, w - 1), min(box[2][1] + self.expand_pix, h - 1)],
        #                        [max(box[3][0] - self.expand_pix, 0), min(box[3][1] + self.expand_pix, h - 1)]]
        #                       for box in box_array])

        return box_array, score_array


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python detector_text.py
    image_name_list = [
        "1209710983_3.jpg",
        "1209711373_2.jpg",
        "1209711469_13.jpg",
        "baidu_zhangcheng_36.jpg",
    ]
    model = Detector(
        os.path.abspath(cur_dir + "/output/DBNet/checkpoint/latest.pth"), gpu="3"
    )
    for image_name in image_name_list:
        print("processing image name: ", image_name)
        path = cur_dir + "/test_data/my_imgs_4/" + image_name
        img = cv2.imread(path)
        # model = Detector(os.path.abspath(root_dir + '/MODEL/DET/DB/DBv0/TextDet/20210601/best.pt'), gpu='0')
        box_array, score_array = model.inference(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_bbox(img, box_array)

        out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(path.replace("/test_data/", "/test_out/"), img)
