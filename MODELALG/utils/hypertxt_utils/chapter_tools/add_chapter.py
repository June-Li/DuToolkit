"""
Copyright (c) 2025 sensedeal
All rights reserved.

File: add_chapter.py
Author: lijun
Date: 2025/01/14
Description: 增加章节信息
"""

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

from MODELALG.utils.common import Log
from MODELALG.utils.hypertxt_utils.chapter_tools import (
    generate_chapter_by_llm,
    generate_chapter_by_re,
    title_tag_by_layout,
    title_tag_by_llm,
    title_tag_by_re,
)

logger = Log(__name__).get_logger()


# 增加章节信息
def add_chapter_info(du_hypertxt):
    """
    增加章节信息
    """
    du_hypertxt = title_tag(du_hypertxt)
    du_hypertxt = generate_chapter(du_hypertxt)
    return du_hypertxt


def title_tag(du_hypertxt):
    """
    增加章节信息
    """
    if (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["title_tag_based_on"]
        == "layout"
    ):
        du_hypertxt = title_tag_by_layout.title_tag(du_hypertxt)
    elif (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["title_tag_based_on"]
        == "llm"
    ):
        try:
            du_hypertxt = title_tag_by_llm.title_tag(du_hypertxt)
        except Exception as e:
            logger.warning("llm章节标注失败，使用re算法")
            du_hypertxt = title_tag_by_re.title_tag(du_hypertxt)
    elif (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["title_tag_based_on"]
        == "re"
    ):
        du_hypertxt = title_tag_by_re.title_tag(du_hypertxt)
    else:
        raise ValueError(
            "Chapter title_tag_based_on must be layout, llm or re, but got %s"
            % du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"][
                "title_tag_based_on"
            ]
        )
    return du_hypertxt


def generate_chapter(du_hypertxt):
    """
    根据章节信息生成章节
    """
    if (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["chapter_based_on"]
        == "llm"
    ):
        try:
            du_hypertxt = generate_chapter_by_llm.generate_chapter(du_hypertxt)
        except Exception as e:
            logger.warning("llm章节生成失败，使用re算法")
            du_hypertxt = generate_chapter_by_re.generate_chapter(du_hypertxt)
    elif (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["chapter_based_on"]
        == "re"
    ):
        du_hypertxt = generate_chapter_by_re.generate_chapter(du_hypertxt)
    else:
        raise ValueError(
            "Chapter chapter_based_on must be llm or re, but got %s"
            % du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["chapter_based_on"]
        )
    return du_hypertxt
