#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#   pass
#
# Author: Li Xiuming
# Last Modified: 2020-10-29
############################################################

# from decimal import Decimal
from operator import itemgetter
from MODELALG.utils.pdfplumber.utils import to_list
from .settings import (
    X_TOLERANCE,
    PARA_TOLERANCE_PERCENT,
    PAGE_PARA_TOLERANCE_PERCENT,
    LINE_CHAR_TOLERANCE,
    START_EN_LENGTH,
    START_NUM_LENGTH,
    LINE_MIN_X1_P,
)
from .utils import line_chars_to_text, line_chars_to_coordinate

# from process_meta import get_min_page_size
import regex as re
import logging

logger = logging.getLogger(__name__)


def cluster_chars(chars):
    """对chars进行分行聚类"""
    if len(chars) == 0:
        return []
    if len(chars) == 1:
        if chars[0]["text"][0] == " ":
            return []
        else:
            return [chars]

    groups = []
    chars = list(sorted(chars, key=itemgetter("doctop")))
    current_group = [chars[0]]
    last = chars[0]
    for char in chars[1:]:
        # 下一个字符的上边界大于上一个字符的下边界就换行
        if char["top"] <= last["bottom"]:
            current_group.append(char)
        else:
            groups.append(current_group)
            current_group = [char]
        last = char
    groups.append(current_group)

    # 删除全是空字符的行
    final_groups = []
    for group in groups:
        is_all_space = True
        for char in group:
            if "text" in char and len(char["text"]) > 0 and char["text"][0] != " ":
                is_all_space = False
                break
        if not is_all_space:
            final_groups.append(group)

    return final_groups


def new_cluster_chars(chars):
    """对chars进行分行聚类"""
    if not chars:
        return []

    chars = sorted(chars, key=lambda x: x["x0"])
    groups = create_groups(chars)

    # 删除全是空字符的行
    final_groups = [
        group for group in groups if any(is_not_space(char) for char in group)
    ]
    return final_groups


def create_groups(chars):
    groups = []
    reference_doctop = chars[0]["doctop"]
    current_group = [chars[0]]

    for char in chars[1:]:
        if can_be_grouped(char, current_group[-1], reference_doctop):
            current_group.append(char)
        else:
            groups.append(current_group)
            current_group = [char]
            reference_doctop = char["doctop"]

    if current_group:
        groups.append(current_group)

    return groups


def can_be_grouped(char, last_char, reference_doctop):
    """Determine if two chars can be grouped in the same line."""
    vertical_distance = abs(char["doctop"] - reference_doctop)
    horizontal_distance = char["x0"] - last_char["x1"]
    close_threshold = float("3.0") * char["size"]

    return (
        vertical_distance <= get_same_line_threshold(char)
        and horizontal_distance <= close_threshold
    )


def get_same_line_threshold(char):
    """Calculate the same line threshold for a char."""
    return float("0.33") * char["size"]


def is_not_space(char):
    """Check if the character is not a space."""
    return char["text"].strip() != ""


def collate_line_chars(chars, tolerance=X_TOLERANCE):
    """对每一行的chars按照x0进行排序"""
    tolerance = float(tolerance)
    coll = []
    last_x1 = None
    chars_sorted = sorted(chars, key=itemgetter("x0"))

    # 删除最后一个为空的字符
    while True:
        if chars_sorted[-1]["text"][0] == " ":
            chars_sorted.pop(-1)
        else:
            break

    # add at 20211215, 删除靠得特别近的字符，pdf加粗可能会重字
    last_x0 = chars_sorted[0]
    # print([(c["x0"], c["x1"], c["y0"], c["y1"], c["text"]) for c in chars_sorted])
    i = 1
    while i < len(chars_sorted):
        if (
            chars_sorted[i]["x0"] <= chars_sorted[i - 1]["x0"] + float(0.5)
            and chars_sorted[i]["text"] == chars_sorted[i - 1]["text"]
        ):
            chars_sorted.pop(i)
            continue
        i += 1

    # 添加空格
    for char in chars_sorted:
        if (last_x1 is not None) and (char["x0"] > (last_x1 + tolerance)):
            coll.append(" ")
        last_x1 = char["x1"]
        coll.append(char)
    return coll


def segment_to_lines(segments, x_tolerance=X_TOLERANCE):
    """对含有chars的块进行分行"""
    segment_lines = []

    for segment in segments:
        if not isinstance(segment, list):
            segment_lines.append(segment)
        elif len(segment) == 0:
            segment_lines.append(segment)
        else:
            chars = to_list(segment)
            clusters = cluster_chars(chars)  # cluster_chars, new_cluster_chars
            line_chars = [
                collate_line_chars(cluster, x_tolerance) for cluster in clusters
            ]
            segment_lines.append(line_chars)
    return segment_lines


def is_title(line, max_size, page_width, min_x0, max_x1):
    # 徐志昂2022-01-11修改，文本前面存在较多空格，导致被识别为标题
    line = list(filter(lambda x: x != " " and x["text"] != " ", line))
    if line[0]["size"] < max_size - float(0.5):
        return False
    mid_position = (line[0]["x0"] + line[-1]["x1"]) / 2
    bias = line[0]["width"] * float(0.5)
    if mid_position < page_width / 2 - bias or mid_position > page_width / 2 + bias:
        return False

    if line[0]["x0"] < min_x0 + float(0.5) and line[-1]["x1"] > max_x1 - float(0.5):
        return False
    # 徐志昂2023-01-12添加：因为有很多文本被错误分成title导致分句，这里添加限制。
    if line[-1]["x1"] - line[0]["x0"] > max_x1 - min_x0 - float(30):
        return False

    for char in line:
        if char == " ":
            continue
        if char["text"] in [",", "，", "。", "!", "！", ";", "；", "："]:
            return False

    return True


def get_symbol_num(para, left_quotationmarks, left_cn_brackets, left_en_brackets):
    for line in para:
        for char in line:
            if char == " ":
                continue
            if "text" not in char:
                continue
            if not char["text"]:
                continue
            if char["text"][0] == " ":
                continue
            if char["text"] == "(":
                left_en_brackets += 1
            if char["text"] == "（":
                left_cn_brackets += 1
            if char["text"] == "《":
                left_quotationmarks += 1
            if char["text"] == ")":
                left_en_brackets -= 1
            if char["text"] == "）":
                left_cn_brackets -= 1
            if char["text"] == "》":
                left_quotationmarks -= 1
    return left_quotationmarks, left_cn_brackets, left_en_brackets


def collate_para_chars(
    line_chars,
    page_bbox,
    last_segment_flag,
    min_size,
    max_size,
    min_x0,
    max_x1,
    tolerance=PARA_TOLERANCE_PERCENT,
    page_tolerance=PAGE_PARA_TOLERANCE_PERCENT,
    line_char_tolerance=LINE_CHAR_TOLERANCE,
):
    # last_segment_flag：是否是当前页面的最后一部分
    """段落换行输出"""
    title_start = [[{"text": "<title>\n"}]]
    title_end = [[{"text": "\n</title>"}]]
    title_flag = False
    # line_chars 是 segment_line(list)，这里与0和1进行比较存在疑问，是否应该是len(line_chars)
    if line_chars == 0:
        return []
    if line_chars == 1:
        return line_chars

    page_width = page_bbox[2] - page_bbox[0]
    tolerance = float(tolerance)
    page_tolerance = float(page_tolerance)
    line_char_tolerance = float(line_char_tolerance)

    # 分段要求：1、一行以常见符号结束；2、是该页最后一行；
    #         3、该行最后一个字符小于该部分中所有chars的x1*阈值
    #         4、该行第一个字符的size和上一行最后一个字符的size差距大于阈值，则上一行分段
    #         4、该行最后一个字符大于该部分中所有chars的x1*阈值但下一行的第一个字符不小于该部分中所有chars的x0*阈值
    # 不分段标准：1、该行最后一个字符大于该部分中所有chars的x1*阈值且下一行的第一个字符小于该部分中所有chars的x0*阈值
    #           2、如果之前的行中存在没有匹配的"《 "或者"（"或者"("，就不分段
    #           3、上一行最后一个字符与last_min_x1的差值小于该行非中文字符串的长度则不分段
    page_char_min_x0 = min([line[0]["x0"] for line in line_chars])
    page_char_max_x1 = max([line[-1]["x1"] for line in line_chars])

    tolerance_width = tolerance * (page_char_max_x1 - page_char_min_x0)
    first_max_x0 = page_char_min_x0 + tolerance_width
    last_min_x1 = max((page_char_max_x1 - tolerance_width), page_tolerance * page_width)
    # 是否换行的标识
    flag = False
    # 左书名号和左括号的数目，初始为0
    left_quotationmarks, left_cn_brackets, left_en_brackets = 0, 0, 0

    paras = []
    para = []
    title = []
    last_len = 0
    str_len = 0
    for i, line in enumerate(line_chars):
        if len(line) == 0 or line is None:
            continue

        if (
            is_title(line, max_size, page_width, min_x0, max_x1)
            and i != len(line_chars) - 1
        ):
            if i == 0:
                title_flag = True
            # 判断当前行与上一行是否属于同一标题
            if len(title) > 0:
                last_bottom = line_chars[i - 1][0]["bottom"]
                cur_top = line[0]["top"]
                if (cur_top - last_bottom) < line[0]["height"] + min_size * float(0.5):
                    title.append(line)
                    continue
                else:
                    if len(para) != 0:
                        paras.append(para)
                        para = []
                    # paras.append(title_start+title+title_end)
                    paras.append(title)
                    title = []
            # 当前行若非本页第一行则：当前行与上一行的行间距>当前行字体的height*Demical(2)
            elif i > 0:
                last_bottom = line_chars[i - 1][0]["bottom"]
                cur_top = line[0]["top"]
                if (cur_top - last_bottom) > line[0]["height"] * float(1.2):
                    title.append(line)
                    continue
            else:
                title.append(line)
                continue
        elif len(title) > 0:
            # todo
            if len(para) != 0:
                paras.append(para)
                para = []
            # paras.append(title_start+title+title_end)
            paras.append(title)
            title = []

        start_str = ""
        # 上一行最后一个字符与last_min_x1的差值小于该行非中文字符串的长度则不分段
        have_han = False
        for char in line:
            if char == " ":
                continue
            if "text" not in char:
                continue
            if not char["text"]:
                continue
            if char["text"][0] == " ":
                start_str += char["text"]
                continue
            if "\u4e00" <= char["text"] <= "\u9fff":
                str_len = char["x0"] - line[0]["x0"]
                have_han = True
                break
            else:
                start_str += char["text"]
        if not have_han:
            str_len = line[-1]["x0"] - line[0]["x0"]

        # TODO 修改，段落粘连
        last_len = (
            page_char_max_x1 - line_chars[i - 1][-1]["x1"]
        )  # TODO 此时的i为0，i-1导致检查到最后一行
        # 如果以特定符号结尾则分段
        if line[-1]["text"] in ["。", "！", "：", ":", "？"]:
            if len(para) > 0:
                # if line_chars[i - 1][-1]["size"] != line[0]["size"] or line_chars[i - 1][-1]["x1"] \
                #         <= page_width * float(LINE_MIN_X1_P) or line_chars[i - 1][-1]["x1"] <= last_min_x1:    ## 上行末和当前行字符大小差别较大，分段
                # 徐志昂20221222修改，目的是修复：
                """
                若预留授予部分在公司2022
                年第三季度报告披露后授予。
                """
                # 上述句子会在2022处进行切句，因为数字的size一般小于文字的size，
                # 原先的逻辑会判断"当前句子的首个文字的大小是不是等于上一个句子的句尾的大小"
                # 因为数字的size一般小于文字的size，如果一行以数字结尾，那么有比较大概率会在数字处进行分句
                if (
                    line[0]["size"]
                    not in set(
                        [
                            line_chars[i - 1][x]["size"]
                            if line_chars[i - 1][x] != " "
                            else ""
                            for x in range(len(line_chars[i - 1]))
                        ]
                    )
                    or line_chars[i - 1][-1]["x1"] <= page_width * float(LINE_MIN_X1_P)
                    or line_chars[i - 1][-1]["x1"] <= last_min_x1
                ):
                    paras.append(para)
                    para = []
            # 以下4行为徐志昂20221212添加，目的是修复：
            # "历史波动率：17.46%、15.87%、17.46%（分别采用上证指数——指数代码：
            # 000001.SH 最近一年、两年、三年的年化波动率）"
            # 上文在指数代码：处进行了错误分句。
            # 如果当前行以特定符号结尾，但是当前还存在括号没有完全匹配，则不分段。
            left_quotationmarks, left_cn_brackets, left_en_brackets = get_symbol_num(
                [line], 0, 0, 0
            )
            if left_en_brackets > 0 or left_cn_brackets > 0 or left_quotationmarks > 0:
                para.append(line)
                continue

            para.append(line)
            paras.append(para)
            flag = False
            para = []
            continue

        if len(para) > 0:
            # left_quotationmarks, left_cn_brackets, left_en_brackets=get_symbol_num(line_chars[i-1],left_quotationmarks, left_cn_brackets, left_en_brackets)
            left_quotationmarks, left_cn_brackets, left_en_brackets = get_symbol_num(
                para, 0, 0, 0
            )
            if (
                left_en_brackets > 0 or left_cn_brackets > 0 or left_quotationmarks > 0
            ):  # 看上一行是否有 "（"，"(","《"   不分段
                para.append(line)
            else:
                # 数字样式   demo：123456.78%   字母：abcdefddd  (\d+[,\.]{0,1})
                # 若以数字开头，获取首部数字
                num_match = re.match(r"(\d+[,\.]{0,1})+", start_str)
                english_match = re.match("[a-zA-Z']+", start_str)
                english_len = 0
                num_len = 0
                if num_match:  # 匹配长数字
                    span = num_match.span()
                    num_len = span[1] - span[0]
                if english_match:  # 匹配长英文
                    span = english_match.span()
                    english_len = span[1] - span[0]

                if line[0]["x0"] <= first_max_x0:  ## 首字靠近左侧
                    if (
                        english_len > START_EN_LENGTH or num_len > START_NUM_LENGTH
                    ):  ## 开头长英文或者数字，不分段
                        para.append(line)
                        continue

                last_len = page_char_max_x1 - line_chars[i - 1][-1]["x1"]
                # if line_chars[i - 1][-1]["size"] != line[0]["size"]:   ## 上行末和当前行字符大小差别较大，分段
                if line[0]["size"] not in set(
                    [
                        line_chars[i - 1][x]["size"]
                        if line_chars[i - 1][x] != " "
                        else ""
                        for x in range(len(line_chars[i - 1]))
                    ]
                ):
                    paras.append(para)
                    para = [line]
                elif line_chars[i - 1][-1]["x1"] <= page_width * float(
                    LINE_MIN_X1_P
                ):  # 判断一行的最短长度，不满足则分段
                    paras.append(para)
                    para = [line]
                elif str_len > last_len and line[0]["text"] not in [
                    "(",
                    "（",
                ]:  ## 行初非中文字符长度大于上行剩余空白长度，不分段
                    para.append(line)
                # TODO：
                elif line_chars[i - 1][-1]["x1"] <= last_min_x1:  ## 行末留空白过长，分段
                    paras.append(para)
                    para = [line]
                elif para[-1][-1]["text"] in [";", "；"]:
                    paras.append(para)
                    para = [line]
                else:
                    para.append(line)
            continue
        para.append(line)

    if len(para) > 0:
        paras.append(para)

    # 遍历到当前页的最后一部分的最后一行,判断是否和下一页拼接起来
    if last_segment_flag:
        # 原来要求页末最后一个字符为中文。
        # 2023-01-06 徐志昂修改，发现存在页末结尾为数字，会导致句子进行切句。
        # 这里增加一个判断逻辑，如果页末为数字且句子长度大于等于5（句子长度大于等于5是为了过滤掉页码），则不会切句。
        if is_title(line_chars[-1], max_size, page_width, min_x0, max_x1):
            flag = False

        elif ("\u4e00" <= line_chars[-1][-1]["text"] <= "\u9fff") or (
            len(line_chars[-1]) >= 5 and "0" <= line_chars[-1][-1]["text"] <= "9"
        ):
            flag = True

    return paras, False, title_flag  # paras, flag, title_flag


def collate_para_chars_old(
    line_chars,
    page_bbox,
    last_segment_flag,
    min_size,
    max_size,
    min_x0,
    max_x1,
    tolerance=PARA_TOLERANCE_PERCENT,
    page_tolerance=PAGE_PARA_TOLERANCE_PERCENT,
    line_char_tolerance=LINE_CHAR_TOLERANCE,
):
    # last_segment_flag：是否是当前页面的最后一部分
    """段落换行输出"""
    title_start = [[{"text": "<title>\n"}]]
    title_end = [[{"text": "\n</title>"}]]
    title_flag = False
    if line_chars == 0:
        return []
    if line_chars == 1:
        return line_chars

    page_width = page_bbox[2] - page_bbox[0]
    tolerance = float(tolerance)
    page_tolerance = float(page_tolerance)
    line_char_tolerance = float(line_char_tolerance)

    # 分段要求：1、一行以常见符号结束；2、是该页最后一行；
    #         3、该行最后一个字符小于该部分中所有chars的x1*阈值
    #         4、该行第一个字符的size和上一行最后一个字符的size差距大于阈值，则上一行分段
    #         4、该行最后一个字符大于该部分中所有chars的x1*阈值但下一行的第一个字符不小于该部分中所有chars的x0*阈值
    # 不分段标准：1、该行最后一个字符大于该部分中所有chars的x1*阈值且下一行的第一个字符小于该部分中所有chars的x0*阈值
    #           2、如果之前的行中存在没有匹配的"《 "或者"（"或者"("，就不分段
    #           3、上一行最后一个字符与last_min_x1的差值小于该行非中文字符串的长度则不分段
    page_char_min_x0 = min([line[0]["x0"] for line in line_chars])
    page_char_max_x1 = max([line[-1]["x1"] for line in line_chars])
    tolerance_width = tolerance * (page_char_max_x1 - page_char_min_x0)
    first_max_x0 = page_char_min_x0 + tolerance_width
    last_min_x1 = max((page_char_max_x1 - tolerance_width), page_tolerance * page_width)

    # 是否换行的标识
    flag = False
    # 左书名号和左括号的数目，初始为0
    left_quotationmarks, left_cn_brackets, left_en_brackets = 0, 0, 0

    paras = []
    para = []
    title = []
    last_len = 0
    str_len = 0
    for i, line in enumerate(line_chars):
        if len(line) == 0 or line is None:
            continue

        if (
            is_title(line, max_size, page_width, min_x0, max_x1)
            and i != len(line_chars) - 1
        ):
            if i == 0:
                title_flag = True
            # 判断当前行与上一行是否属于同一标题
            if len(title) > 0:
                last_bottom = line_chars[i - 1][0]["bottom"]
                cur_top = line[0]["top"]
                if (cur_top - last_bottom) < line[0]["height"] + min_size * float(0.5):
                    title.append(line)
                    continue
            # 当前行若非本页第一行则：当前行与上一行的行间距>当前行字体的height*Demical(2)
            elif i > 0:
                last_bottom = line_chars[i - 1][0]["bottom"]
                cur_top = line[0]["top"]
                if (cur_top - last_bottom) > line[0]["height"] * float(1.2):
                    title.append(line)
                    continue
            else:
                title.append(line)
                continue
        elif len(title) > 0:
            # paras.append(title_start+title+title_end)
            paras.append(title)
            title = []

        # 如果以特定符号结尾则分段
        if line[-1]["text"] in ["。", "！", "：", ":", "？"]:
            para.append(line)
            paras.append(para)
            flag = False
            para = []
            continue

        # 上一行最后一个字符与last_min_x1的差值小于该行非中文字符串的长度则不分段
        have_han = False
        for char in line:
            if char == " ":
                continue
            if char["text"][0] == " ":
                continue
            if "\u4e00" <= char["text"] <= "\u9fff":
                str_len = char["x0"] - line[0]["x0"]
                have_han = True
                break
        if not have_han:
            str_len = line[-1]["x0"] - line[0]["x0"]

        if len(para) > 0:
            # left_quotationmarks, left_cn_brackets, left_en_brackets=get_symbol_num(line_chars[i-1],left_quotationmarks, left_cn_brackets, left_en_brackets)
            left_quotationmarks, left_cn_brackets, left_en_brackets = get_symbol_num(
                para, 0, 0, 0
            )
            if left_en_brackets > 0 or left_cn_brackets > 0 or left_quotationmarks > 0:
                para.append(line)
            else:
                last_len = page_char_max_x1 - line_chars[i - 1][-1]["x1"]
                if line_chars[i - 1][-1]["size"] != line[0]["size"]:
                    paras.append(para)
                    para = [line]

                elif str_len > last_len:
                    para.append(line)
                elif line_chars[i - 1][-1]["x1"] <= last_min_x1:
                    paras.append(para)
                    para = [line]
                else:
                    para.append(line)
            continue
        para.append(line)

    if len(para) > 0:
        paras.append(para)

    # 遍历到当前页的最后一部分的最后一行,判断是否和下一页拼接起来
    if last_segment_flag:
        if ("\u4e00" <= line_chars[-1][-1]["text"] <= "\u9fff") and not is_title(
            line_chars[-1], max_size, page_width, min_x0, max_x1
        ):
            flag = True

    return paras, flag, title_flag


def extract_para_text(para):
    """提取段落"""
    text = ""
    for line in para:
        text += line_chars_to_text(line)
    return text


def extract_para_coordinate(page_num, para, page_height, page_width):
    """
    提取段落的坐标
    :param para:
    :return:
    """
    coordinate = []
    for line in para:
        coordinate.append(
            line_chars_to_coordinate(page_num, line, page_height, page_width)
        )
    return coordinate
