#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################################
#
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved
#
# Description:pass
#
# Author: Li Xiuming, Xu Zhiang
# Last Modified: 2020-10-29
############################################################
# import matplotlib.pyplot as plt
import itertools
import json
import os
import pathlib
import time

import cv2
import fitz
import numpy as np

# from MODELALG.utils import pdfplumber as pp
import pdfplumber as pp
from loguru import logger
from pdfminer.pdfdocument import PDFPasswordIncorrect

from MODELALG.utils import common
from MODELALG.utils.hypertxt_utils.add_chapter import MakeChapter
from MODELALG.utils.sdnlp_libs.base.hypertxt import (
    cut_sentences,
    cut_sentences_0804,
    split_list_by_indices,
    split_sentences,
)
from WORKFLOW.OTHER.pdf_txt.v0.utils.process_chars import (
    collate_para_chars,
    extract_para_coordinate,
    extract_para_text,
    new_cluster_chars,
    segment_to_lines,
)
from WORKFLOW.OTHER.pdf_txt.v0.utils.process_meta import (
    get_min_page_size,
    get_pages_meta,
)
from WORKFLOW.OTHER.pdf_txt.v0.utils.process_page import page_segment
from WORKFLOW.OTHER.pdf_txt.v0.utils.process_table import find_tables, parse_table

# import sys
#
# cur_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.join(cur_dir, "../../../")
# sys.path.append(os.path.abspath(root_dir))


class Pdf2Txt:
    def __init__(
        self,
        configs,
        title: str = None,
        cut_function=cut_sentences_0804,
    ):  # cut_sentence_pdf_add_coordinate
        self.configs = configs
        self.title = title
        self.cut_function = cut_function
        self.upload_file_to_oss = common.UploadFileToOss(self.configs["OSS_ADDRESS"])

    def apply(self, filepaths: list):
        hypertxts = []
        for filepath in filepaths:
            hypertxt = self.apply_one(filepath)
            if len(hypertxt) == 0:
                continue
            hypertxts.append(hypertxt)
        return hypertxts

    def apply_one(self, du_hypertxt):
        """
        pdf转为txt
        Args:
            filepath: 需要解析的pdf文件路径, 或者是文件指针
            output_filepath: 写入文件路径，文件名需以.txt结尾
        Returns:
            hypertxt: dict pdf的hypertxt，当output_txt=False时
        """
        filepath = du_hypertxt["file_path"]
        if (
            isinstance(filepath, (str, pathlib.Path))
            and not filepath.endswith("pdf")
            and not filepath.endswith("PDF")
        ):
            logger.warning("Is it really a .pdf file?", filepath)

        starttime = time.time()
        try:
            pdf = pp.open(filepath)
            fitz_pdf = fitz.open(filepath)
        except PDFPasswordIncorrect as psi:
            raise PDFPasswordIncorrect("pdf可能存在加密")
        except Exception as e:
            raise Exception(f"PDF可能不存在, filepath: {filepath}")
        sum_special_chars = sum([1 for a in pdf.chars if "cid" in a["text"]])
        # if sum_special_chars > len(pdf.chars) * 0.25 and sum_special_chars > 1000:
        #     raise ValueError(f"PDF文档存在较多乱码，疑似解析失败")
        extractor = Extract(pdf, fitz_pdf)
        # extractor.extract()：页面中提取处表格，并且将字符按照页-片段-行的逻辑按照坐标关系进行处理
        # 得到表格信息和pdf的页-片段-行的文本信息，其中信息包含：
        # pages_segment_lines(list):pages_segment_lines[pages_index][segment_index][line_index] = line_text
        # pages_meta(dict):包含页眉、页码、目录等信息。
        layout = None
        # layout = extractor.detect_layout()
        extractor.extract(du_hypertxt, layout)
        # 将每一行按照逻辑组合成段落para
        # 根据pages_segment_lines和pages_meta进行跨页的表格合并和跨页的段落合并。
        # text_list: 每（一段文本/表格）的文本信息。
        extractor.extract_text(output_header=True, output_pagenum=True)
        pdf.close()
        end_time = time.time()
        used_time = end_time - starttime
        if used_time > 10.0:
            logger.info("Long time consumed:", used_time)

        # 页眉页脚目录信息  add 20220810
        self.footers, self.headers, self.catalogs = [], [], []
        if extractor.pages_meta is not None:
            for x in extractor.pages_meta["pagenums"].values():
                self.footers.append((x,) if x is not None else ("",))
            for x in extractor.pages_meta["headers"].values():
                self.headers.append(x if x is not None else "")
            self.catalogs = list(
                extractor.pages_meta["catalog"]["catalog_dict"].values()
            )
        hypertxt = self.generate_hypertxt(
            extractor.text_list, extractor.pages_bbox
        )  # section_range随时可以加，目前没有加入
        du_hypertxt["hypertxt"]["sd_new_hypertxt"] = json.dumps(
            hypertxt, indent=4, ensure_ascii=False, default=common.handle_decimal
        )
        return du_hypertxt

    def generate_hypertxt(self, phrases, pages_bbox):
        hypertxt = {
            "metadata": {
                "footers": self.footers,
                "headers": self.headers,
                "catalogs": self.catalogs,
                "pages_bbox": np.array(pages_bbox, dtype=int).tolist(),
            },
            "chapters": {},
            "context": [],
        }
        if self.title is not None:
            hypertxt["context"].append(
                {
                    "text": self.title,
                    "type": "title",
                    "sid": 0,
                    "pid": 0,
                    "page_idx": 0,
                    "text_box": None,
                    "metadata": {},
                }
            )

        context = []
        pid = 0
        sid = 0
        coordinates = []
        for phrase in phrases:
            page_bbox = hypertxt["metadata"]["pages_bbox"][phrase["page_idx"] - 1]
            if len(phrase["text"]) == 0:
                continue
            if phrase["type"] == "paragraph":
                sentences, cut_indices, final_indices = self.cut_function(
                    phrase["text"]
                )

                sentences_bboxes = []
                if "coordinate" in phrase.keys():
                    coordinate_list = list(
                        itertools.chain(*[item for item in phrase["coordinate"]])
                    )
                    # assert len(coordinate_list) == sum(len(text) for sentence in sentences for text in sentence)
                    coordinates = split_list_by_indices(
                        coordinate_list, cut_indices, final_indices
                    )
                    coordinates = [item for item in coordinates if len(item) > 0]
                    sentences = [
                        text
                        for sentence in sentences
                        for text in sentence
                        if len(text) > 0
                    ]

                    for coordinate in coordinates:
                        boxes = [i[1] for i in coordinate]
                        boxes = np.array(boxes)
                        boxes[:, [0, 2]] *= page_bbox[2]
                        boxes[:, [1, 3]] *= page_bbox[3]
                        boxes[:, [1, 3]] = page_bbox[3] - boxes[:, [1, 3]]
                        box = [
                            int(min(boxes[:, [0, 2]].flatten())),
                            int(min(boxes[:, [1, 3]].flatten())),
                            int(max(boxes[:, [0, 2]].flatten())),
                            int(max(boxes[:, [1, 3]].flatten())),
                        ]
                        sentences_bboxes.append(box)
                    for idx, sentence in enumerate(sentences):
                        if idx not in coordinates or len(sentence) != len(
                            coordinates[idx]
                        ):
                            logger.warning(f"坐标可能出错！pdf2txt")

                pid += 1
                for i, sentence in enumerate(sentences):
                    sid += 1

                    rec = {
                        "text": sentence.replace("<***۞Enter۞***>", ""),
                        "type": "text",
                        "pid": pid,
                        "sid": sid,
                        "page_idx": phrase["page_idx"],
                        "text_box": sentences_bboxes[i] if coordinates else None,
                        "source": (
                            coordinates[i]
                            if False and coordinates and i < len(coordinates)
                            else ""
                        ),
                        "metadata": {
                            "section_range": (
                                phrase["page_range"] if "page_range" in phrase else ""
                            )
                        },
                    }
                    context.append(rec)
            elif phrase["type"] == "table":
                for idx_1, elem in enumerate(phrase["text"]):
                    phrase["text"][idx_1][1] = elem[1].replace("<***۞Enter۞***>", " \n")
                pid += 1
                sid += 1
                rec = {
                    "text": phrase["text"],
                    "type": "table",
                    "pid": pid,
                    "sid": sid,
                    "cid": None,
                    "page_idx": phrase["page_idx"],
                    "text_box": None,
                    "metadata": {"section_range": phrase["page_range"]},
                }
                context.append(rec)
            elif phrase["type"] == "picture":
                pic = cv2.imdecode(
                    np.frombuffer(phrase["text"]["stream"].rawdata, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if pic is not None:
                    (
                        response_code,
                        response_message,
                        response_down_url,
                    ) = self.upload_file_to_oss.upload_image(pic)
                else:
                    continue
                pid += 1
                sid += 1
                if response_message == "Done":
                    rec = {
                        "text": "",
                        "down_url": response_down_url,
                        "type": "picture",
                        "sid": sid,
                        "pid": pid,
                        "cid": None,
                        "page_idx": phrase["page_idx"],
                        "text_box": None,
                        "metadata": {},
                    }
                    logger.info("layout数据上传成功！！！")
                else:
                    rec = {
                        "text": "",
                        "down_url": "Failed",
                        "type": "picture",
                        "sid": sid,
                        "pid": pid,
                        "cid": None,
                        "page_idx": phrase["page_idx"],
                        "text_box": None,
                        "metadata": {},
                    }
                    logger.info("layout数据上传失败！！！")
                context.append(rec)

        hypertxt["context"].extend(context)

        maker = MakeChapter()
        hypertxt = maker.apply(hypertxt)
        return hypertxt


class Extract:
    def __init__(self, pdf, fitz_pdf):
        self.pdf = pdf
        self.fitz_pdf = fitz_pdf
        self.text_list = None
        self.text_section_list = None
        if len(pdf.pages) == 0:
            raise ValueError("页数为0")

    def calculate_gap_width(self, segments, page_width):
        """检测是否为双栏格式，并返回空白区域的宽度"""

        # 创建一个水平直方图，使用较小的bin来检测空白
        bin_width = 5
        num_bins = int(page_width / bin_width)
        histogram = [0] * num_bins

        # 填充直方图
        for segment in segments:
            for char in segment:
                bin_index = int(char["x0"] / bin_width)
                if bin_index < num_bins:
                    histogram[bin_index] += 1

        # 计算页面左半部分和右半部分的字符数量
        middle = int(num_bins / 2)
        left_side_chars = sum(histogram[:middle])
        right_side_chars = sum(histogram[middle:])

        # 设定阈值，要求页面两侧都有一定数量的字符
        threshold = 10

        # 寻找中间区域的水平空白
        search_range = int(0.2 * num_bins)
        gap_width = 0
        consecutive_empty_bins = 0
        if left_side_chars > threshold and right_side_chars > threshold:
            for i in range(middle - search_range, middle + search_range):
                if histogram[i] == 0:
                    consecutive_empty_bins += 1
                else:
                    if consecutive_empty_bins >= 3:
                        gap_width = consecutive_empty_bins * bin_width
                        break
                    consecutive_empty_bins = 0

        return gap_width

    def find_double_column_pages(self, gap_widths, tolerance=2):
        """找出双栏页面的索引"""
        if len(gap_widths) < 2:
            return False, []

        # 计算平均值和标准差
        mean_width = np.mean(gap_widths)
        std_dev = np.std(gap_widths)

        # 找出在范围内的页面索引
        pages_within_range = [
            index
            for index, gw in enumerate(gap_widths)
            if mean_width - tolerance * std_dev
            <= gw
            <= mean_width + tolerance * std_dev
        ]

        # 如果列表中至少80%的值在范围内，我们认为gap宽度是一致的
        is_double_column = len(pages_within_range) / len(gap_widths) >= 0.8

        return is_double_column, pages_within_range

    def detect_layout(self):
        """判断是不是双栏pdf"""
        gap_widths = []
        for page in self.pdf.pages:
            tables = find_tables(page)
            _, segments = page_segment(page, tables)

            page_width = page.bbox[2] - page.bbox[0]
            try:
                gap_width = self.calculate_gap_width(segments, page_width)
            except Exception as e:
                gap_width = 0

            if gap_width > 0:
                gap_widths.append(gap_width)
        if len(gap_widths) < len(self.pdf.pages) * 0.5 or len(gap_widths) <= 3:
            is_double_column = False
            double_column_pages = []
        else:
            is_double_column, double_column_pages = self.find_double_column_pages(
                gap_widths
            )
            # print(f"检测到的空白间距:{gap_widths}")
        print(
            f"检测到PDF双栏文件:{is_double_column}, 一共有{len(double_column_pages)}页"
        )
        layout = {
            "is_double_column": is_double_column,
            "double_column_pages": double_column_pages,
        }
        return layout

    def extract(self, du_hypertxt, layout):
        """页面中提取表格，并且将字符按照页-片段-行的逻辑按照坐标关系进行处理"""
        self.pages_meta = None
        self.pages_segment_lines = []
        self.page_idx_list = []
        self.pages_bbox = []

        if layout is None:
            double_column_pages = None
            is_double_column = None
        else:
            is_double_column = layout["is_double_column"]
            double_column_pages = set(layout["double_column_pages"])

        for page_index, page in enumerate(self.pdf.pages):
            # 使用原生的pdfplumber中find_tables方法
            tables = find_tables(du_hypertxt, page, self.fitz_pdf[page_index])

            # page_segment:按照表格的上下边界，把页面划分为多个片段segment, segment包含片段中所有的字符。returns: segments(list)
            _, segments = page_segment(page, tables)

            # 如果是双栏页面，并且当前页是双栏格式，则拆分成两个单独的页面进行处理
            if (
                is_double_column
                and double_column_pages
                and page_index in double_column_pages
            ):
                left_segments, right_segments = self.split_segments_for_double_column(
                    segments, page.bbox[2] - page.bbox[0]
                )
                left_segment_lines = segment_to_lines(left_segments)
                right_segment_lines = segment_to_lines(right_segments)

                # 将左栏和右栏作为两个单独的页面添加到结果列表中
                half_width = (page.bbox[2] - page.bbox[0]) / 2
                left_bbox = (
                    page.bbox[0],
                    page.bbox[1],
                    page.bbox[0] + half_width,
                    page.bbox[3],
                )
                right_bbox = (
                    page.bbox[0] + half_width,
                    page.bbox[1],
                    page.bbox[2],
                    page.bbox[3],
                )

                self.pages_segment_lines.append(left_segment_lines)
                self.pages_bbox.append(left_bbox)
                self.page_idx_list.append(page_index + 1)
                self.pages_segment_lines.append(right_segment_lines)
                self.pages_bbox.append(right_bbox)
                self.page_idx_list.append(page_index + 1)
            else:
                # 使用原来的segment_to_lines
                segment_lines = segment_to_lines(segments)
                self.pages_segment_lines.append(segment_lines)
                self.pages_bbox.append(page.bbox)
                self.page_idx_list.append(page_index + 1)

        # 从pages_segment_lines中分离页眉、页码、目录信息到pages_meta，分离后pages_segment_lines不包含page_meta信息。
        self.pages_segment_lines, self.pages_meta = get_pages_meta(
            self.pdf.pages, self.pages_segment_lines, self.pages_bbox
        )

        # 添加layout
        # for idx, segment_lines in enumerate(self.pages_segment_lines):
        for idx, image in enumerate(self.pdf.images):
            self.pages_segment_lines[image["page_number"] - 1].append(image)

    # 在类的其他部分添加此辅助方法
    def split_segments_for_double_column(self, segments, page_width):
        left_column_segments = []
        right_column_segments = []

        # 分离左栏和右栏的字符
        for segment in segments:
            left_chars = [char for char in segment if char["x0"] < page_width / 2]
            right_chars = [char for char in segment if char["x0"] >= page_width / 2]

            if left_chars:
                left_column_segments.append(left_chars)
            if right_chars:
                right_column_segments.append(right_chars)

        return left_column_segments, right_column_segments

    def extract_text(
        self,
        output_header=True,
        output_pagenum=True,
        output_catalog=True,
        output_title=True,
    ):
        """
        1. 聚合成段落文本
        2. 根据pages_segment_lines和pages_meta进行跨页的表格合并和跨页的段落合并
        """
        text_list = []
        merge_flag = False  # 是否合并的标志
        last_table = []
        is_out_catalog = False
        text_section_list = []  # 每一段落的页码索引 [] # [0,0]第0 页； [0,1]第0～1页
        table_res = []
        table_section_begin = -1
        # TODO

        for i, (segment_lines, page_bbox, page_idx) in enumerate(
            zip(self.pages_segment_lines, self.pages_bbox, self.page_idx_list)
        ):
            # 这里的i可以认为是从0开始的页码
            page_height = page_bbox[3] - page_bbox[1]
            page_width = page_bbox[2] - page_bbox[0]
            # print(page_height, page_width)

            if (
                i > 0
                and output_pagenum
                and self.pages_meta is not None
                and self.pages_meta["pagenums"].get(i - 1, None) is not None
            ):
                text_section_list.append([i, i])

            # 页眉
            if (
                output_header
                and self.pages_meta is not None
                and self.pages_meta["headers"].get(i, None) is not None
            ):
                text_section_list.append([i + 1, i + 1])

            # # 目录
            # if output_catalog and self.pages_meta is not None and i in self.pages_meta["catalog"][
            #     "catalog_pagenums"] and self.pages_meta["catalog"]["catalog_text"] != "":
            #     text_section_list.append([i + 1, i + 1])
            #     is_out_catalog = True
            #     output_catalog = False
            #
            # # 若当前页面是目录，并且已经输出，则跳过
            # if is_out_catalog and i in self.pages_meta["catalog"]["catalog_pagenums"]:
            #     continue

            # 正文
            min_size, second_min_size, max_size, min_x0, max_x1 = get_min_page_size(
                segment_lines
            )

            # 删除每个segment_line出现的空白段落
            segment_lines = list(
                filter(
                    lambda x: not (isinstance(x, list) and len(x) == 0), segment_lines
                )
            )

            for n, obj in enumerate(segment_lines):
                if isinstance(obj, dict) and obj["object_type"] == "image":
                    text_section_list.append([None, None])
                    text_list.append(
                        {
                            "type": "picture",
                            "text": obj,
                            "page_idx": page_idx,
                            "page_range": [None, None],
                        }
                    )
                    continue
                if not isinstance(obj, list):
                    cur_table = obj
                    # if last_table == [] 输出当前表格解析结果。
                    # if last_table != [] 说明表格需要向上合并，输出表格合并解析结果。
                    last_segment_lines = list(
                        filter(
                            lambda x: not (isinstance(x, list) and len(x) == 0),
                            self.pages_segment_lines[i - 1],
                        )
                    )
                    last_ori_table = (
                        last_segment_lines[-1]
                        if len(last_segment_lines) > 0 and last_table != []
                        else None
                    )
                    table_res = parse_table(
                        cur_table,
                        last_table,
                        last_ori_table,
                    )
                    merge_flag = False

                    # 表格可能向下合并的条件：表格在当前页面的最后一部分, 且当前页面不是最后一页。
                    if (
                        n == len(segment_lines) - 1
                        and i != len(self.pages_segment_lines) - 1
                    ):
                        if not last_table:
                            table_section_begin = (
                                i + 1
                            )  # table_section_begin记录表格合并的起点
                        last_table = table_res

                    # 如果表格不能向下合并, 当前表格已经是个独立的表格，输出当前表格。
                    else:
                        cur_table_section = (
                            [table_section_begin, i + 1]
                            if last_table
                            else [i + 1, i + 1]
                        )
                        text_section_list.append(cur_table_section)
                        text_list.append(
                            {
                                "type": "table",
                                "text": table_res,
                                "page_idx": page_idx,
                                "page_range": cur_table_section,
                            }
                        )
                        last_table = []
                else:
                    # 如果在处理表格合并过程中, 遇到了文本, 输出表格
                    if last_table:
                        cur_table_section = [table_section_begin, i]
                        text_section_list.append(cur_table_section)
                        text_list.append(
                            {
                                "type": "table",
                                "text": table_res,
                                "page_idx": page_idx,
                                "page_range": cur_table_section,
                            }
                        )
                        last_table = []

                    last_segment_flag = False  # last_segment_flag：当前部分是否是当前页面最后一部分的标志
                    # 徐志昂2023-01-09修改：每一页开始解析的时候，都令title_flag = False，不然的话，上一页的title_flag也会引入到下一页，导致错误。
                    title_flag = False

                    if (
                        n == len(segment_lines) - 1
                    ):  # 当前页面的最后一行，最后一个segment_line
                        last_segment_flag = True
                        paras, flag, title_flag = collate_para_chars(
                            obj,
                            page_bbox,
                            last_segment_flag,
                            min_size,
                            max_size,
                            min_x0,
                            max_x1,
                        )
                    else:
                        paras, _, _ = collate_para_chars(
                            obj,
                            page_bbox,
                            last_segment_flag,
                            min_size,
                            max_size,
                            min_x0,
                            max_x1,
                        )
                        flag = False
                    if n == 0 and merge_flag and not title_flag:
                        # 合并当前页和上一页
                        text_list[-1]["text"] = "".join(
                            [text_list[-1]["text"], extract_para_text(paras[0])]
                        )
                        text_list[-1]["coordinate"] = text_list[-1][
                            "coordinate"
                        ] + extract_para_coordinate(
                            i + 1, paras[0], page_height, page_width
                        )
                        text_list[-1]["page_range"] = [
                            text_list[-1]["page_range"][0],
                            i + 1,
                        ]
                        text_section_list[-1] = [i, i + 1]

                        paras.pop(0)

                    merge_flag = flag
                    # 分析每一段落的文本信息和文本跨度信息
                    for para in paras:
                        text_list.append(
                            {
                                "type": "paragraph",
                                "text": extract_para_text(para),
                                "coordinate": extract_para_coordinate(
                                    page_num=i + 1,
                                    para=para,
                                    page_height=page_height,
                                    page_width=page_width,
                                ),
                                "page_idx": page_idx,
                                "page_range": [i + 1, i + 1],
                            }
                        )
                        text_section_list.append([i + 1, i + 1])
        # text_list只存储可能是章节的文本，不包含页眉、页脚、目录
        self.text_list = text_list
        self.text_section_list = text_section_list
