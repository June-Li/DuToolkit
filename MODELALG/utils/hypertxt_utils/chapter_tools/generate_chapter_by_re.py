import json
import re
from collections import OrderedDict

import numpy as np

from MODELALG.utils import common
from MODELALG.utils.common import Log

logger = Log(__name__).get_logger()

C_P = [
    [r"^第([0-9一二三四五六七八九十百千零]{1,6})部分", 0],  # 第一部分 or 第1部分
    [r"^第([0-9一二三四五六七八九十百千零]{1,6})章", 1],  # 第一章 or 第1章
    [r"^第([0-9一二三四五六七八九十百千零]{1,6})节", 2],  # 第一节 or 第1节
    [r"^第([0-9一二三四五六七八九十百千零]{1,6})条", 3],  # 第一条 or 第1条
    [r"^([一二三四五六七八九十百千零]{1,6}) ", 4],  # 一+空格
    [r"^([一二三四五六七八九十百千零]{1,6})、", 4],  # 一、
    [r"^([一二三四五六七八九十百千零]{1,6}) ", 4],  # 一
    [r"^([一二三四五六七八九十百千零]{1,6})．", 4],  # 一．中文点
    [r"^([一二三四五六七八九十百千零]{1,6})\. ", 4],  # 一． 英文点+空格
    [r"^([一二三四五六七八九十百千零]{1,6})\.", 4],  # 一． 英文点无空格
    [r"^\([一二三四五六七八九十百千零]{1,6}\)", 5],  # (一)
    [r"^（([一二三四五六七八九十百千零]{1,6})）", 5],  # （一）
    [r"^([一二三四五六七八九十百千零]{1,6})\)", 5],  # 一)
    [r"^([一二三四五六七八九十百千零]{1,6})）", 5],  # 一）
    [r"^([0-9]{1,2}) ", 6],  # 1+空格
    [r"^([0-9]{1,2})、", 6],  # 1、
    [r"^([0-9]{1,2})．", 6],  # 1．中文点
    [r"^([0-9]{1,2})\. ", 6],  # 1．英文点+空格
    [r"^([0-9]{1,2})\.", 6],  # 1. 英文点无空格
    [r"^\(([0-9]{1,2})\)", 7],  # (1)
    [r"^（([0-9]{1,2})）", 7],  # （1）
    [r"^([0-9]{1,2})\)", 7],  # 1)
    [r"^([0-9]{1,2})）", 7],  # 1）
]


def generate_chapter(du_hypertxt):
    """
    根据章节信息生成章节
    """
    sd_hypertxt = json.loads(du_hypertxt["hypertxt"]["sd_hypertxt"])
    title_list = []
    title_id_list = []
    for idx_0, elem in enumerate(sd_hypertxt["context"]):
        if "is_title" in elem and elem["is_title"]:
            title_list.append(elem["text"])
            title_id_list.append(idx_0)

    # re生成章节
    result = [[None, title_list[i]] for i in range(len(title_list))]
    if len(title_list) != 0:
        c_list = [1, 1, 1, 1, 1, 1, 1, 1]
        c_change_flag_list = [False, False, False, False, False, False, False, False]
        for idx_0, title in enumerate(title_list):
            p_stage = None
            for idx_1, p in enumerate(C_P):
                r = re.match(p[0], title)
                if r:
                    try:
                        numbers = common.extract_numbers(r.group())
                        if len(numbers) != 0:
                            c_list[p[1]] = numbers[0]
                            c_change_flag_list[p[1]] = True
                    except Exception as e:
                        print(e)
                        numbers = []
                    p_stage = p[1]
                    if p_stage < len(c_list) - 1:
                        c_list[p_stage + 1 :] = [1] * len(c_list[p_stage + 1 :])
                    break
            if p_stage is not None:
                result[idx_0][0] = ".".join(map(str, c_list[: p_stage + 1]))
        # 如果某个级别从头到尾都一样，那把这个维度取消掉
        for idx_0, elem in enumerate(result):
            if elem[0] is None:
                continue
            c_split_list = elem[0].split(".")
            c = ".".join(
                np.array(c_split_list)[c_change_flag_list[: len(c_split_list)]]
            )
            if c == "":
                c = None
            result[idx_0][0] = c

        # 将章节信息加入到sd_hypertxt
        for idx_0, elem in enumerate(result):
            if elem[0] is not None:
                sd_hypertxt["context"][title_id_list[idx_0]]["cid"] = elem[0]

    # 生成sid、pid、cid
    sid, pid, cid = 1, 1, "0"
    for idx_0, elem in enumerate(sd_hypertxt["context"]):
        if elem["cid"] is not None:
            sid, pid, cid = 1, 1, elem["cid"]
        if elem["type"] == "text":
            sentence_len = len(re.split(r"[。！？!?.]", elem["text"]))
        else:
            sentence_len = 1
        sd_hypertxt["context"][idx_0]["sid"] = sid  # [sid, sid + sentence_len - 1]
        sd_hypertxt["context"][idx_0]["pid"] = pid
        sd_hypertxt["context"][idx_0]["cid"] = cid
        sid += sentence_len
        pid += 1
    sd_hypertxt["chapters"] = dict([elem for elem in result if elem[0] is not None])
    du_hypertxt["hypertxt"]["sd_hypertxt"] = json.dumps(
        sd_hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64
    )
    return du_hypertxt
