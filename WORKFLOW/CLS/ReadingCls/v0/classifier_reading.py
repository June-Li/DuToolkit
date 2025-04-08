# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 17:27
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(root_dir)
import copy
import os
import statistics
import threading
import time
import traceback

import numpy as np
import torch
from transformers import LayoutLMv3ForTokenClassification

from MODELALG.utils import common
from MODELALG.utils.common import Log
from WORKFLOW.CLS.ReadingCls.v0.layoutreader.v3.helpers import (
    boxes2inputs,
    parse_logits,
    prepare_inputs,
)

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Classifier:
    def __init__(
        self,
        model_path="/volume/weights/layoutreader/",
        device="cuda:0",
    ):
        with torch.no_grad():
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_path
            ).to(device)
            logger.info(" ···-> reader load model succeeded!")

    def get_line_height(self, blocks):
        page_line_height_list = []
        for block in blocks:
            if block["type"] in ["text"]:
                for line in block["lines"]:
                    bbox = line["bbox"]
                    if (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) > 2:
                        page_line_height_list.append(int(bbox[3] - bbox[1]))
        if len(page_line_height_list) > 0:
            return statistics.median(page_line_height_list)
        else:
            return 10

    def insert_lines_into_block(self, block_bbox, line_height, page_w, page_h):
        # block_bbox是一个元组(x0, y0, x1, y1)，其中(x0, y0)是左下角坐标，(x1, y1)是右上角坐标
        x0, y0, x1, y1 = np.array(block_bbox, dtype=int).tolist()

        block_height = y1 - y0
        block_weight = x1 - x0

        # 如果block高度小于n行正文，则直接返回block的bbox
        if line_height * 2 < block_height:
            if (
                block_height > page_h * 0.25
                and page_w * 0.5 > block_weight > page_w * 0.25
            ):  # 可能是双列结构，可以切细点
                lines = int(block_height / line_height) + 1
            else:
                # 如果block的宽度超过0.4页面宽度，则将block分成3行(是一种复杂布局，图不能切的太细)
                if block_weight > page_w * 0.4:
                    lines = 3
                    line_height = (y1 - y0) / lines
                elif block_weight > page_w * 0.25:  # （可能是三列结构，也切细点）
                    lines = int(block_height / line_height) + 1
                else:  # 判断长宽比
                    if block_height / block_weight > 1.2:  # 细长的不分
                        return [[x0, y0, x1, y1]]
                    else:  # 不细长的还是分成两行
                        lines = 2
                        line_height = (y1 - y0) / lines

            # 确定从哪个y位置开始绘制线条
            current_y = y0

            # 用于存储线条的位置信息[(x0, y), ...]
            lines_positions = []

            for i in range(lines):
                lines_positions.append(
                    [
                        x0,
                        current_y,
                        x1,
                        min(round(current_y + line_height), round(y1)),
                    ]
                )
                current_y += line_height
            return lines_positions
        else:
            return [[x0, y0, x1, y1]]

    def get_lines_box(self, fix_blocks, page_w, page_h, line_height):
        page_line_list = []

        def add_lines_to_block(b):
            line_bboxes = self.insert_lines_into_block(
                b["bbox"], line_height, page_w, page_h
            )
            page_line_list.extend(line_bboxes)

        for block in fix_blocks:
            if block["type"] in ["text", "title"]:
                if len(block["lines"]) == 0 or (
                    block["type"] in ["title"]
                    and len(block["lines"]) == 1
                    and (block["bbox"][3] - block["bbox"][1]) > line_height * 2
                ):
                    add_lines_to_block(block)
                else:
                    for line in block["lines"]:
                        bbox = line["bbox"]
                        page_line_list.append(bbox)
            elif block["type"] in ["figure", "table", "formula"]:
                add_lines_to_block(block)

        if len(page_line_list) > 510:  # layoutreader最高支持512line
            return None

        # 使用layoutreader排序
        x_scale = 1000.0 / page_w
        y_scale = 1000.0 / page_h
        boxes = [
            [
                page_line_list[i][0] * float(x_scale),
                page_line_list[i][1] * float(y_scale),
                page_line_list[i][2] * float(x_scale),
                page_line_list[i][3] * float(y_scale),
            ]
            for i in range(len(page_line_list))
        ]

        return boxes, page_line_list, x_scale, y_scale

    def __call__(self, blocks, page_w, page_h):
        """
        input:
            img_ori: opencv读取的图片格式;
        Returns:
            outs: 识别结果
        """
        try:
            with lock:
                line_height = 10  # self.get_line_height(blocks)
                lines_box, page_line_list, x_scale, y_scale = self.get_lines_box(
                    blocks, page_w, page_h, line_height
                )
                inputs = boxes2inputs(np.array(lines_box, dtype=int).tolist())
                inputs = prepare_inputs(inputs, self.model)
                logits = self.model(**inputs).logits.cpu().squeeze(0)
                orders = parse_logits(logits, len(lines_box))
                sorted_bboxes = [page_line_list[i] for i in orders]
                return sorted_bboxes
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    classifier = Classifier()

    fix_blocks = [
        {
            "bbox": [0, 0, 900, 32],
            "type": "text",
            "lines": [{"bbox": [0, 0, 900, 32], "spans": []}],
        },
        {
            "bbox": [500, 40, 900, 72],
            "type": "text",
            "lines": [{"bbox": [500, 40, 900, 72], "spans": []}],
        },
        {
            "bbox": [0, 40, 400, 72],
            "type": "text",
            "lines": [{"bbox": [0, 40, 400, 72], "spans": []}],
        },
        {
            "bbox": [0, 100, 900, 132],
            "type": "text",
            "lines": [{"bbox": [0, 100, 900, 132], "spans": []}],
        },
    ]
    for i in range(1):
        start = time.time()
        outs = classifier(fix_blocks, 1000, 1400)
        print(outs)
        print(time.time() - start)
