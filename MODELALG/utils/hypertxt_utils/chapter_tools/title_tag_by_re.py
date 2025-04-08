"""
Copyright (c) 2025 sensedeal
All rights reserved.

File: title_tag_by_layout.py
Author: lijun
Date: 2025/01/14
Description: 根据正则表达式增加章节信息
"""

import json
import re

from MODELALG.utils import common
from MODELALG.utils.common import Log
from MODELALG.utils.hypertxt_utils.chapter_tools import title_tag_by_layout

logger = Log(__name__).get_logger()
CHAPTER_COMMON_PATTERNS = [
    r"^第(.{1,6})节",  # 第一节
    r"^第(.{1,6})章",  # 第一章
    r"^第(.{1,6})条",  # 第一条
    r"^第(.{1,6})部分",  # 第一部分
    r"^([一二三四五六七八九十百千零]{1,6}) ",  # 一+空格
    r"^([一二三四五六七八九十百千零]{1,6})、",  # 一、
    r"^\([一二三四五六七八九十百千零]{1,6}\)",  # (一)
    r"^（([一二三四五六七八九十百千零]{1,6})）",  # （一）
    r"[一二三四五六七八九十百千零]{1,6}\)",  # 一)
    r"([一二三四五六七八九十百千零]{1,6})）",  # 一）
    # r"^([0-9]{1,3}) ",  # 1+空格
    r"^([0-9]{1,3})、",  # 1、
    r"^\(([0-9]{1,3})\)",  # (1)
    r"^（([0-9]{1,3})）",  # （1）
    r"^([0-9]{1,3})\)",  # 1)
    r"^([0-9]{1,3})）",  # 1）
    r"^([0-9]{1,3})．",  # 1．中文点
    r"^([0-9]{1,3})\. ",  # 1．英文点+空格
    r"^([0-9]{1,3})\.",  # 1. 英文点无空格
]


def title_tag(du_hypertxt):
    """
    根据layout信息增加章节信息
    """
    sd_hypertxt = json.loads(du_hypertxt["hypertxt"]["sd_hypertxt"])
    for idx_0, elem in enumerate(sd_hypertxt["context"]):
        if elem["type"] == "text":
            for pattern in CHAPTER_COMMON_PATTERNS:
                if re.match(pattern, elem["text"]) and len(elem["text"]) < 40:
                    sd_hypertxt["context"][idx_0]["is_title"] = True
                    break
    du_hypertxt["hypertxt"]["sd_hypertxt"] = json.dumps(
        sd_hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64
    )
    du_hypertxt = title_tag_by_layout.title_tag(du_hypertxt)
    return du_hypertxt
