#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#   pass
#
# Author: Li Xiuming, Xu Zhiang
# Last Modified: 2020-10-29
############################################################

import re

# from decimal import Decimal
from logging import exception

from .utils import line_chars_to_text
from .settings import PAGENUM_TOLERANCE, HEADER_TOLERANCE
from .process_chars import extract_para_text
from collections import Counter


# 得到当前页面中所有char的第2小的size
def get_min_page_size(segment_lines):
    min_size = 100
    second_min_size = 100
    max_size = 0
    min_x0 = 100
    max_x1 = 0
    for segment_line in segment_lines:
        if not isinstance(segment_line, list):
            continue
        for line in segment_line:
            if second_min_size > line[0]["size"]:
                second_min_size = line[0]["size"]
                if second_min_size < min_size:
                    min_size, second_min_size = second_min_size, min_size
            if max_size < line[0]["size"]:
                max_size = line[0]["size"]

            if line[0]["x0"] < min_x0:
                min_x0 = line[0]["x0"]
            if line[-1]["x1"] > max_x1:
                # 因为遇到某些情况如果行末出现了像'。'这种标点符号，会导致一行的最右端点的距离x1过于大。
                # 所以遇到了这种标点就需要过滤。
                if line[-1]["text"] == "。":
                    if len(line) > 1 and line[-2] != " " and line[-2]["x1"] > max_x1:
                        max_x1 = line[-2]["x1"]
                else:
                    max_x1 = line[-1]["x1"]
    return min_size, second_min_size, max_size, min_x0, max_x1


def get_possible_header(segment_lines, page_bbox, tolerance=HEADER_TOLERANCE):
    """找到第一行作为可能的页眉，取出第一块第一行的text"""
    tolerance = float(tolerance)

    if len(segment_lines) == 0:  # 空页
        return None
    if not isinstance(segment_lines[0], list):  # 第一块为表格
        return None
    if len(segment_lines[0]) == 0:  # 第一块为空的
        return None
    if len(segment_lines[0][0]) == 0:  # 第一块的第一行为空
        return None
    if segment_lines[0][0][0]["top"] >= float(
        page_bbox[1] + float(0.1) * (page_bbox[3] - page_bbox[1])
    ):
        return None

    # max_size = max(c.get('size', float('0')) for a in segment_lines for b in a for c in b  if c != ' ')
    paragraph_list = list(filter(lambda x: isinstance(x, list), segment_lines))
    max_size = max(
        c.get("size", float("0"))
        for a in paragraph_list
        for b in a
        for c in b
        if c != " "
    )
    # 原先在get_min_page_size中会忽略表格中的文本，进而导致解析过程中出现问题。
    # 2023-01-05徐志昂修改，因为有的页的页码很小，会导致min_size的很小，并进一步导致页眉不能抽取出来。
    # 原先的代码为:
    # min_size,_,_,_,_=get_min_page_size(segment_lines)
    # # 页眉的数目不能过大
    # if segment_lines[0][0][0]["size"]>min_size+float(1):
    #     return None

    # 现在修改后的代码为:
    _, second_min_size, _, _, _ = get_min_page_size(segment_lines)
    # # 页眉的大小不能过大
    if segment_lines[0][0][0]["size"] > second_min_size + float(1):
        return None

    if segment_lines[0][0][0]["size"] >= max_size - float(0.5):
        return None
    line_text = line_chars_to_text(segment_lines[0][0])
    line_text = line_text.strip()
    if line_text == "":
        return None

    return line_text


def get_possible_pagenum(segment_lines, page_bbox, tolerance=PAGENUM_TOLERANCE):
    """找到最后一行作为可能的页码，取出最后一块最后一行的text"""
    tolerance = float(tolerance)

    if len(segment_lines) == 0:  # 空页
        return None
    if not isinstance(segment_lines[-1], list):  # 最后一块为表格
        return None
    if len(segment_lines[-1]) == 0:  # 最后一块为空的
        return None
    if len(segment_lines[-1][-1]) == 0:  # 最后一块的最后一行为空
        return None
    if (
        segment_lines[-1][-1][-1]["bottom"]
        <= float(page_bbox[3] - tolerance * (page_bbox[3] - page_bbox[1])) - 5
    ):
        return None

    line_text = line_chars_to_text(segment_lines[-1][-1])
    line_text = line_text.strip()
    if line_text == "":
        return None

    if (
        re.search("[0-9页/IVXLXCDM]+", line_text) == 0
        or len(line_text.replace(" ", "")) > 9
    ):
        return None
    _, second_min_size, max_size, _, _ = get_min_page_size(segment_lines)

    paragraph_list = list(filter(lambda x: isinstance(x, list), segment_lines))
    max_size = max(
        c.get("size", float("0"))
        for a in paragraph_list
        for b in a
        for c in b
        if c != " "
    )  # max_size = max(char['size'] for char in page.chars)

    if segment_lines[-1][-1][-1]["size"] > second_min_size + float(2):
        return None
    if segment_lines[-1][-1][-1]["size"] >= max_size - float(0.7):
        return None
    # 这里的'^'出现在字符集合模式的第一个字符，表示取反。
    if re.search("[^0-9一二三四五六七八九十页IVXLXCDM第共 -/，]+", line_text):
        return None

    return line_text


def is_headers(possible_headers, page_num):
    """页眉总数应几乎等于页码数目的一半"""
    headers_list = [h for h in possible_headers if h is not None]
    if len(headers_list) > page_num / 2 - 1:
        return True
    # 如果有一个候选页眉出现次数大于5，那么我们认为找到了页眉
    if len(headers_list) > 0 and Counter(headers_list).most_common(1)[0][1] > 5:
        return True
    return False


# def is_pagenums(possible_pagenums):
#     '''去重后页数应几乎等于所有页的数目'''
#
#     pagenums_list = [n for n in possible_pagenums if n is not None]
#     pagenums_set = set(pagenums_list)
#     if len(pagenums_list) == 0:
#         return False
#     else:
#         if len(pagenums_set) < 0.9 * len(pagenums_list):
#             return False
#
#     return False


def is_title(line, max_size, page_width):
    if line[0]["size"] < max_size - float(0.5):
        return False
    mid_position = (line[0]["x0"] + line[-1]["x1"]) / 2
    bias = line[0]["width"] * float(0.5)
    if mid_position < page_width / 2 - bias or mid_position > page_width / 2 + bias:
        return False
    for char in line:
        if char == " ":
            continue
        if char["text"] in [",", "，", "。", "!", "！"]:
            return False
    return True


def get_headers(pages, pages_segment_lines, pages_bbox):
    """
    返回所有的页眉
    Return：
        possible_headers：所有可能的页眉
        most_common_header：出现次数大于总次数一般的页眉，如果不存在为""
    """
    possible_headers = [
        get_possible_header(segment_lines, page_bbox)
        for segment_lines, page_bbox in zip(pages_segment_lines, pages_bbox)
    ]
    # return possible_headers
    is_pages_headers = is_headers(possible_headers, len(pages_segment_lines))
    if is_pages_headers:
        header_count = Counter(
            list(map(lambda x: x.replace(" ", "") if x else None, possible_headers))
        ).most_common(1)
        most_common_header = ""
        # 如果一个页眉的出现次数大于总次数的0.3倍。
        if len(header_count) > 0 and header_count[0][1] >= len(possible_headers) * 0.3:
            most_common_header = header_count[0][0]
        return possible_headers, most_common_header
    else:
        return None, None


def get_pagenums(pages, pages_segment_lines, pages_bbox):
    """返回所有的页码"""
    possible_pagenums = [
        get_possible_pagenum(segment_lines, page_bbox)
        for segment_lines, page_bbox in zip(pages_segment_lines, pages_bbox)
    ]

    return possible_pagenums

    # is_pages_pagenums = is_pagenums(possible_pagenums)
    # if is_pages_pagenums:
    #     return possible_pagenums
    # else:
    #     return None


def get_catalog(pages_segment_lines, strict=False):
    """获取目录"""
    catalog_pagenums = []

    catalog_mark = False
    catalog_symbol = None
    for i, segment_lines in enumerate(pages_segment_lines):
        # 只考虑前5页  todo xiugai
        if (strict and i > 4) or (
            not strict and i > len(pages_segment_lines) * 0.5
        ):  # 限制查找目录范围
            break

        # 只考虑只有一个块且不是表格的页
        if len(segment_lines) != 1 or not isinstance(segment_lines[0], list):
            # 如果上一页是目录，则这一页直接break
            if catalog_mark:
                break
            else:
                continue
        # 只考虑第一块
        for j, line in enumerate(segment_lines[0]):
            # 只考虑前5句
            if j > 4:
                break
            line_text = line_chars_to_text(line).replace(" ", "")
            if re.sub(" +", "", line_text) == "目录":
                catalog_mark = True
                # TODO 目录页面中没有连接符号
                # 查找目录中的标示符号； 查看目录后两行中出现次数最多的字符且出现次数大于5次，否则默认是 " "
                line_text += (
                    line_chars_to_text(segment_lines[0][j + 1]).replace(" ", "")
                    if len(segment_lines[0]) > j + 1
                    else ""
                )
                line_text += (
                    line_chars_to_text(segment_lines[0][j + 2]).replace(" ", "")
                    if len(segment_lines[0]) > j + 2
                    else ""
                )
                catalog_symbol = re.sub("[0-9A-Za-z\u4e00-\u9fa5]", "", line_text)
                # 取出最大个数的符号
                if catalog_symbol:
                    catalog_symbol = max(catalog_symbol, key=catalog_symbol.count)
                else:
                    catalog_mark = False
                break
            elif catalog_mark and catalog_symbol is not None:
                # 上一页是目录，这一页也是; 使用前两行，防止目录跨行
                line_text += (
                    line_chars_to_text(segment_lines[0][j + 1]).replace(" ", "")
                    if len(segment_lines[0]) > j + 1
                    else ""
                )
                if catalog_symbol * 5 in line_text:
                    break
                else:
                    catalog_mark = False
            else:
                catalog_mark = False

        if catalog_mark:
            catalog_pagenums.append(i)

        if catalog_mark and len(segment_lines) > 1:
            break

    return catalog_pagenums


def get_pages_meta(pages, pages_segment_lines, pages_bbox):
    """存储所有页面的页眉，页码，目录"""
    pages_meta = {
        "headers": {},
        "pagenums": {},
        "title": {},
        "catalog": {"catalog_pagenums": [], "catalog_text": ""},
    }

    # 获取页眉header, 和出现次数最多的页眉most_common_header
    headers, most_common_header = get_headers(pages, pages_segment_lines, pages_bbox)
    # 获取页码
    try:
        pagenums = get_pagenums(pages, pages_segment_lines, pages_bbox)
    except Exception as e:
        pagenums = None
    # 获取目录
    catalog_pagenums = get_catalog(pages_segment_lines)

    # 在page_segment_lines中删除掉所有的页眉，页脚。
    catalog_text = []

    rule_based_pagenum = []  # 基于对比全文页码相似度获取的页码
    rule_based_header = []  # 基于对比全文页眉相似度获取的页眉
    candidate_cross_page_table = []

    for i, segment_lines in enumerate(pages_segment_lines):
        if len(segment_lines) == 0:
            continue

        if headers is not None:
            # 徐志昂2022-01-09添加：如果一页开头的句子和出现次数最多的页眉（出现次数大于0.5*总页数）完全相同，那么就认为该句子也是页眉。
            if (
                headers[i] is None
                and len(segment_lines) > 0
                and isinstance(segment_lines[0], list)
                and len(segment_lines[0]) > 0
                and line_chars_to_text(segment_lines[0][0]).replace(" ", "")
                == most_common_header
            ):
                headers[i] = most_common_header
                rule_based_header.append((i, most_common_header))
                segment_lines[0].pop(0)
            elif headers[i] is not None:
                segment_lines[0].pop(0)
        if pagenums is not None:
            if pagenums[i] is not None:
                segment_lines[-1].pop(-1)
            elif (
                i >= 1
                and pagenums[i - 1] is not None
                and pagenums[i] is None
                and len(segment_lines) > 0
                and isinstance(segment_lines[-1], list)
                and len(segment_lines[-1]) > 0
            ):
                try:
                    cur_text = line_chars_to_text(segment_lines[-1][-1])
                    cur_text_without_num = "".join(
                        list(filter(lambda x: "9" < x or x < "0", cur_text))
                    )
                    cur_text_num = "".join(
                        list(filter(lambda x: "0" <= x <= "9", cur_text))
                    )

                    last_text = pagenums[i - 1]
                    last_text_without_num = "".join(
                        list(filter(lambda x: "9" < x or x < "0", last_text))
                    )
                    last_text_num = "".join(
                        list(filter(lambda x: "0" <= x <= "9", last_text))
                    )
                    if (
                        int(cur_text_num) == int(last_text_num) + 1
                        and cur_text_without_num == last_text_without_num
                    ):
                        pagenums[i] = cur_text
                        # TODO 调试信息
                        rule_based_pagenum.append((i, pagenums[i]))
                        # print(f"给第{i}页文档添加了页码{pagenums[i]}")
                        segment_lines[-1].pop(-1)
                except:
                    continue

        # 删除掉页眉页码之后再提取目录
        if i in catalog_pagenums:
            catalog_text.append([line_chars_to_text(line) for line in segment_lines[0]])
            pages_segment_lines[i] = []

    # TODO 调试信息
    # if rule_based_pagenum:
    # print(f"\n基于对比规则，新增加的页码：{rule_based_pagenum}")
    # if rule_based_header:
    # print(f"基于对比规则，新增加的页眉：{rule_based_header}\n")
    pagenum_page = set(list(map(lambda x: x[0], rule_based_pagenum)))
    header_page = set(list(map(lambda x: x[0], rule_based_header)))

    # TODO 调试信息
    for i, segment_lines in enumerate(pages_segment_lines):
        if len(segment_lines) > 0:
            pages_segment_lines[i] = list(
                filter(
                    lambda x: True if not isinstance(x, list) else len(x) > 0,
                    segment_lines,
                )
            )

    for i, segment_lines in enumerate(pages_segment_lines):
        if (
            len(segment_lines) > 0
            and i + 1 < len(pages_segment_lines)
            and len(pages_segment_lines[i + 1]) > 0
            and not isinstance(segment_lines[-1], list)
            and not isinstance(pages_segment_lines[i + 1][0], list)
        ):
            candidate_cross_page_table.append(i)
    # print(f'candidate_cross_page_table:{candidate_cross_page_table}')

    meta_affect_page = [
        page if page in pagenum_page or page + 1 in header_page else ""
        for page in candidate_cross_page_table
    ]
    meta_affect_page = list(filter(lambda x: x != "", meta_affect_page))
    # if meta_affect_page:
    #     print(
    #         f'基于新的去除页眉页脚规则，应该能够多合并{len(meta_affect_page)}个跨页表格，具体页码为{meta_affect_page}\n')

    # 移动页眉、页码、目录到meta信息里
    if headers is not None:
        pages_meta["headers"] = {i: headers[i] for i in range(len(headers))}
    if pagenums is not None:
        pages_meta["pagenums"] = {i: pagenums[i] for i in range(len(pagenums))}
    pages_meta["catalog"]["catalog_pagenums"] = catalog_pagenums

    # add 0811 一份文件中多份目录
    catalog_dict = {}  # 初始化目录 key是目录(start_page, end_page)
    catalog_text_str = ""
    if catalog_pagenums is not None and len(catalog_pagenums) > 0:
        start_p = catalog_pagenums[0]
        one_catalog_text = []
        for i in range(len(catalog_pagenums) - 1):
            one_catalog_text.extend(catalog_text[i])
            catalog_text_str += "\n".join(catalog_text[i])
            if catalog_pagenums[i + 1] - catalog_pagenums[i] != 1:
                catalog_dict[(start_p, catalog_pagenums[i])] = "\n".join(
                    one_catalog_text
                )
                one_catalog_text = []
                start_p = catalog_pagenums[i + 1]
        if catalog_text:
            one_catalog_text.extend(catalog_text[-1])
        catalog_dict[(start_p, catalog_pagenums[-1])] = "\n".join(one_catalog_text)

    pages_meta["catalog"]["catalog_text"] = catalog_text_str
    pages_meta["catalog"]["catalog_dict"] = catalog_dict
    return pages_segment_lines, pages_meta
