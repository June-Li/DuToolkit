"""
Copyright (c) 2025 sensedeal
All rights reserved.

File: title_tag_by_layout.py
Author: lijun
Date: 2025/01/14
Description: 根据llm增加章节信息
"""

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import json

import numpy as np

from MODELALG.utils import common
from MODELALG.utils.common import Log
from WORKFLOW.OTHER.llm_api.v0.llm_processor import get_answer

logger = Log(__name__).get_logger()


def title_tag(du_hypertxt):
    """
    根据llm增加章节信息
    """
    prompt = str(
        open(
            root_dir
            + du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["title_tag_llm"][
                "prompt_file_path"
            ],
            "r",
            encoding="utf-8",
        ).read()
    )
    sd_hypertxt = json.loads(du_hypertxt["hypertxt"]["sd_hypertxt"])
    step = 50
    content, content_idx = [], []
    idx_0 = 0
    while True:
        if len(content) >= step or idx_0 >= len(sd_hypertxt["context"]):
            prompt += "\n" + str(content)
            result = get_answer(
                prompt,
                "text",
                url=du_hypertxt["configs"]["API_BASE"],
                model=du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"][
                    "title_tag_llm"
                ]["model"],
                max_retries=du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"][
                    "title_tag_llm"
                ]["max_retries"],
            )
            try:
                result = eval(result)
            except:
                raise Exception("标注章节失败")
            if len(result) != len(content):
                raise Exception("标注章节失败")
            for idx_1, elem in enumerate(result):
                if elem[0]:
                    sd_hypertxt["context"][content_idx[idx_1]]["is_title"] = True
            content, content_idx = [], []
            if idx_0 >= len(sd_hypertxt["context"]):
                break
        if sd_hypertxt["context"][idx_0]["type"] == "text":
            content.append(sd_hypertxt["context"][idx_0]["text"])
            content_idx.append(idx_0)
        idx_0 += 1
    du_hypertxt["hypertxt"]["sd_hypertxt"] = json.dumps(
        sd_hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64
    )
    return du_hypertxt
