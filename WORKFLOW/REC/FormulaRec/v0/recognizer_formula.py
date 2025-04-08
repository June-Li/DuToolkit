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

from MODELALG.DET.YOLO.YOLOv5.models.experimental import attempt_load
from MODELALG.DET.YOLO.YOLOv5.utils.datasets import letterbox
from MODELALG.DET.YOLO.YOLOv5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)

# from MODELALG.DET.YOLO.YOLOv5.utils.torch_utils import select_device
from MODELALG.utils.common import Log, draw_poly_boxes, select_device
from WORKFLOW.OTHER.Pix2Text.pix2text import Pix2Text, merge_line_texts
from WORKFLOW.OTHER.Pix2Text.pix2text.formula_detector import MathFormulaDetector
from WORKFLOW.OTHER.Pix2Text.pix2text.latex_ocr import LatexOCR
from WORKFLOW.OTHER.Pix2Text.pix2text.text_formula_ocr import y_overlap
from WORKFLOW.OTHER.Pix2Text.pix2text.utils import (
    adjust_line_height,
    merge_adjacent_bboxes,
    sort_boxes,
)

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Recognizer:
    def __init__(
        self,
        model_path="/volume/weights/pix2img_model/1.1/mfr-onnx",
        device="cuda:0",
        half_flag=False,
        conf_thres=0.5,
    ):
        with torch.no_grad():
            self.half_flag = half_flag
            self.weights = model_path
            self.device = device
            self.conf_thres = conf_thres
            self.latex_ocr_model = LatexOCR(
                model_dir=self.weights,
                device=device,
            )
            # self.half = "cpu" not in self.device
            # if self.half_flag and self.half:
            #     self.latex_ocr_model.half()  # to FP16
            self.languages = ("en", "ch_sim")
            logger.info(" ···-> load model succeeded!")

    def _post_process(self, outs):
        match_pairs = [
            (",", ",，"),
            (".", ".。"),
            ("?", "?？"),
        ]
        formula_tag = "^[（\(]\d+(\.\d+)*[）\)]$"

        def _match(a1, a2):
            matched = False
            for b1, b2 in match_pairs:
                if a1 in b1 and a2 in b2:
                    matched = True
                    break
            return matched

        for idx, line_boxes in enumerate(outs):
            if (
                any([_lang in ("ch_sim", "ch_tra") for _lang in self.languages])
                and len(line_boxes) > 1
                and line_boxes[-1]["type"] == "text"
                and line_boxes[-2]["type"] != "text"
            ):
                if line_boxes[-1]["text"].lower() == "o":
                    line_boxes[-1]["text"] = "。"
            if len(line_boxes) > 1:
                # 去掉边界上多余的标点
                for _idx2, box in enumerate(line_boxes[1:]):
                    if (
                        box["type"] == "text"
                        and line_boxes[_idx2]["type"] == "embedding"
                    ):  # if the current box is text and the previous box is embedding
                        if _match(line_boxes[_idx2]["text"][-1], box["text"][0]) and (
                            not line_boxes[_idx2]["text"][:-1].endswith("\\")
                            and not line_boxes[_idx2]["text"][:-1].endswith(r"\end")
                        ):
                            line_boxes[_idx2]["text"] = line_boxes[_idx2]["text"][:-1]
                # 把 公式 tag 合并到公式里面去
                for _idx2, box in enumerate(line_boxes[1:]):
                    if (
                        box["type"] == "text"
                        and line_boxes[_idx2]["type"] == "isolated"
                    ):  # if the current box is text and the previous box is embedding
                        if y_overlap(line_boxes[_idx2], box, key="position") > 0.9:
                            if re.match(formula_tag, box["text"]):
                                # 去掉开头和结尾的括号
                                tag_text = box["text"][1:-1]
                                line_boxes[_idx2]["text"] = line_boxes[_idx2][
                                    "text"
                                ] + " \\tag{{{}}}".format(tag_text)
                                new_xmax = max(
                                    line_boxes[_idx2]["position"][2][0],
                                    box["position"][2][0],
                                )
                                line_boxes[_idx2]["position"][1][0] = line_boxes[_idx2][
                                    "position"
                                ][2][0] = new_xmax
                                box["text"] = ""

            outs[idx] = [box for box in line_boxes if box["text"].strip()]
        return outs

    def __call__(self, imgs):
        """
        input:
            imgs: opencv读取的图片格式;
        Returns:
            latexs: ['{\\cal L}_{d e t}={\\cal L}_{c l s}+{\\cal L}_{r e g}+{\\cal L}_{\\theta}.', ……]
            scores: [0.9879366755485535, ……]
        """
        try:
            with lock:
                crop_patches = []
                for img in imgs:
                    img = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    crop_patches.append(img)
                mf_results = self.latex_ocr_model.recognize(
                    crop_patches, batch_size=1, **{}
                )
                latexs, scores = [], []
                for elem in mf_results:
                    if elem["score"] > self.conf_thres:
                        latexs.append(elem["text"])
                        scores.append(elem["score"])
                    else:
                        latexs.append("")
                        scores.append(0)
                # outs = []
                #
                # if boxes is None:
                #     boxes = []
                #     top = 0
                #     for img in imgs:
                #         boxes.append(
                #             [
                #                 [0, top],
                #                 [img.shape[1], top],
                #                 [img.shape[1], img.shape[0]],
                #                 [0, img.shape[0]],
                #             ]
                #         )
                #         top = img.shape[0] + 50
                #
                # for box, patch_out in zip(boxes, mf_results):
                #     text = patch_out["text"]
                #     outs.append(
                #         {
                #             "type": "isolated",
                #             "text": text,
                #             "position": np.array(box),
                #             "score": patch_out["score"],
                #         }
                #     )
                # outs = sort_boxes(outs, key="position")
                # outs = [merge_adjacent_bboxes(bboxes) for bboxes in outs]

                # outs = adjust_line_height(outs, 3000, max_expand_ratio=0.2)
                # outs = self._post_process(outs)
                return latexs, scores
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    detector = Recognizer()
    path = "/volume/test_data/多场景数据测试/formula-elem-img/"
    image_name_list = os.listdir(path)
    imgs = []
    for image_name in image_name_list:
        img_ori = cv2.imread(path + image_name)
        imgs.append(img_ori)
    for i in range(10):
        start = time.time()
        outs = detector(imgs)
        print("total use time: ", time.time() - start)
