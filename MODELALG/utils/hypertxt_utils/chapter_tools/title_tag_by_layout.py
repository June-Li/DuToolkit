"""
Copyright (c) 2025 sensedeal
All rights reserved.

File: title_tag_by_layout.py
Author: lijun
Date: 2025/01/14
Description: 根据layout信息增加章节信息
"""

import json

import numpy as np

from MODELALG.utils import common
from MODELALG.utils.common import Log, cal_iou_parallel

logger = Log(__name__).get_logger()


def title_tag(du_hypertxt):
    """
    根据layout信息增加章节信息
    """
    sd_hypertxt = json.loads(du_hypertxt["hypertxt"]["sd_hypertxt"])
    for idx_0, page in enumerate(du_hypertxt["total_result"]):
        title_boxes = []
        for elem in page["layout_info"]["layout"]:
            if elem["subtype"] == "title":
                title_boxes.append(elem["bbox"])
        if len(title_boxes) == 0:
            continue

        chr_boxes = []
        chr_idxs = []
        for idx_1, elem in enumerate(sd_hypertxt["context"]):
            if elem["type"] != "text":
                continue
            if elem["page_idx"] != idx_0 + 1:
                continue
            chr_boxes.append(elem["text_box"])
            chr_idxs.append(idx_1)
        if len(chr_boxes) == 0:
            continue

        iou_matrix = cal_iou_parallel(chr_boxes, title_boxes, cal_type=-1)
        iou_matrix = np.max(iou_matrix, axis=1)
        is_title_idx = np.where(iou_matrix > 0.6)[0]
        for idx in is_title_idx:
            sd_hypertxt["context"][chr_idxs[idx]]["is_title"] = True
    du_hypertxt["hypertxt"]["sd_hypertxt"] = json.dumps(
        sd_hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64
    )
    return du_hypertxt
