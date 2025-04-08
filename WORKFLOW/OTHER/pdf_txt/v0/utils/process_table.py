#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import re
import time
from decimal import Decimal

import cv2
import Levenshtein
import numpy as np
import torch
from loguru import logger
from PIL import Image

from MODELALG.utils import common
from MODELALG.utils.common import Log
from WORKFLOW.DET.TableDet.v1.detector_table import Detector as Detector_table

# from WORKFLOW.OTHER.TableStruct.v4.table_cell_postprocess import merge_line_cell
from WORKFLOW.OTHER.OCR.v0 import utils
from WORKFLOW.OTHER.OCR.v0.AuxiliaryMeans.table import (
    StraightenCells,
    TableStructure,
    TextToCellBorder,
    TextToCellBorderless,
)
from WORKFLOW.OTHER.OCR.v0.utils import CvModels

from .settings import TABLE_SETTINGS
from .utils import format_cell_text

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


# logger = Log(__name__).get_logger()


class Table(object):
    def __init__(self, page, bbox, cur_table, cells):
        self.page = page
        self.bbox = tuple(bbox)
        self.cur_table = cur_table
        self.cells = cells
        self.cv = True


def use_cv_struct_table(du_hypertxt, img, boxes, page_index):
    totaltime = time.time()
    du_hypertxt["data"] = copy.deepcopy(utils.data_template)
    cv_models = utils.CvModels(du_hypertxt["configs"])

    # 只保留表格位置的图像，剩余部分全是空白
    masked_img = np.ones(img.shape, dtype=np.uint8) * 255
    for box in boxes:
        masked_img[box[1] : box[3], box[0] : box[2]] = copy.deepcopy(
            img[box[1] : box[3], box[0] : box[2]]
        )
    du_hypertxt["data"]["ori_img"] = masked_img

    # 文本检测工作
    starttime = time.time()
    du_hypertxt = utils.joint_textdetector_to_du(
        du_hypertxt,
        cv_models.text_detector(
            utils.joint_du_to_textdetector(du_hypertxt, loop_flag=False)
        ),
    )
    logger.info(
        " ···-> 文本检测成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 文本方向分类工作
    starttime = time.time()
    if du_hypertxt["configs"]["PROCESSES_CONTROL"]["OCRModule"]["use_cls_flag"]:
        du_hypertxt = utils.joint_textclassifier_to_du(
            du_hypertxt,
            cv_models.text_classifier(utils.joint_du_to_textclassifier(du_hypertxt)),
        )
    else:
        du_hypertxt["data"]["chr_info"]["chrs_cls"] = [0] * len(
            du_hypertxt["data"]["chr_info"]["chrs_img"]
        )
    logger.info(
        " ···-> 文本方向分类成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 表格检测工作
    du_hypertxt["data"]["table_info"]["tables_box"] = boxes
    du_hypertxt["data"]["table_info"]["tables_img"] = [
        img[box[1] : box[3], box[0] : box[2]] for box in boxes
    ]

    # 表格分类工作
    starttime = time.time()
    if (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["TableModule"]["Classifier_table"][
            "cls_type"
        ]
        == 0
    ):
        du_hypertxt["data"]["table_info"]["tables_cls"] = [0] * len(
            du_hypertxt["data"]["table_info"]["tables_img"]
        )
    elif (
        du_hypertxt["configs"]["PROCESSES_CONTROL"]["TableModule"]["Classifier_table"][
            "cls_type"
        ]
        == 1
    ):
        du_hypertxt["data"]["table_info"]["tables_cls"] = [1] * len(
            du_hypertxt["data"]["table_info"]["tables_img"]
        )
    else:
        du_hypertxt = utils.joint_tableclassifier_to_du(
            du_hypertxt,
            cv_models.table_classifier(utils.joint_du_to_tableclassifier(du_hypertxt)),
        )
    logger.info(
        " ···-> 表格分类成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 有线表格单元格检测+无线表格chrs_box赋值
    starttime = time.time()
    du_hypertxt = utils.joint_tablecelldetector_to_du(
        du_hypertxt,
        cv_models.table_cell_detector(utils.joint_du_to_tablecelldetector(du_hypertxt)),
    )
    logger.info(
        " ···-> 有线表格单元格检测+无线表格chrs_box赋值成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 有线表格规整出单元格，并用得到的单元格把文本裁断
    starttime = time.time()
    du_hypertxt = StraightenCells()(du_hypertxt)
    logger.info(
        " ···-> 有线表格规整出单元格，并用得到的单元格把文本裁断成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 文本识别工作
    starttime = time.time()
    du_hypertxt = utils.joint_textrecognizer_to_du(
        du_hypertxt,
        cv_models.text_recognizer(utils.joint_du_to_textrecognizer(du_hypertxt)),
    )
    logger.info(
        " ···-> 文本识别成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 使用pdf工具的结果修复ocr识别的结果
    starttime = time.time()
    logger.info(
        " ···-> 使用pdf工具的结果修复ocr识别的结果开始({} ·-> 第{}页)···->".format(
            du_hypertxt["filename"], page_index
        )
    )
    if du_hypertxt["configs"]["PROCESSES_CONTROL"]["PagePreprocessModule"][
        "use_pdf_tool_text_repair"
    ]["use_flag"]:
        du_hypertxt = utils.use_pdf_tool_text_repair(img, page_index, du_hypertxt)
    logger.info(
        " ···-> 使用pdf工具的结果修复ocr识别的结果成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 有线表格文本入单元格+无线表格规整出单元格+无线表格文本入单元格
    starttime = time.time()
    du_hypertxt = TextToCell()(du_hypertxt)
    logger.info(
        " ···-> 有线表格文本入单元格+无线表格规整出单元格+无线表格文本入单元格成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    # 表格结构化输出
    starttime = time.time()
    du_hypertxt = TableStructure()(du_hypertxt)
    logger.info(
        " ···-> 表格结构化成功({}s)({} ·-> 第{}页)！".format(
            str(round(time.time() - starttime, 2)), du_hypertxt["filename"], page_index
        )
    )

    logger.info(
        " ···-> cv处理表格总耗时({}s)({} ·-> 第{}页)！\n\n\n".format(
            str(round(time.time() - totaltime, 2)), du_hypertxt["filename"], page_index
        )
    )
    return du_hypertxt


def find_tables(
    du_hypertxt, page, fitz_page, page_index, table_settings=TABLE_SETTINGS
):
    """提取表格，待修改"""
    # starttime = time.time()
    pp_tables = page.find_tables(table_settings)  # pdfplumber提取表格
    return pp_tables

    pp_tables_boxes = [list(map(lambda x: int(x), t.bbox)) for t in pp_tables]

    # 图像方法提取表格 -》start
    dpi = du_hypertxt["configs"]["PROCESSES_CONTROL"]["PagePreprocessModule"][
        "pdf2img_dpi"
    ]
    # pil_image = page.to_image(resolution=dpi)  # 设置想要的分辨率，注意这里图片是放大过的，为了检测更好
    # img = np.array(pil_image.original)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pix = fitz_page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # logger.info(
    #     " ···-> ****************************************************耗时({}s)！\n\n\n".format(
    #         str(round(time.time() - starttime, 2))
    #     )
    # )
    table_boxes_pic, _, _ = CvModels(du_hypertxt["configs"]).table_detector(
        [img.copy()]
    )[0]

    if len(table_boxes_pic) != 0:
        arg_idx = np.argsort(np.array(table_boxes_pic)[:, 1])
        table_boxes_pic = np.array(table_boxes_pic)[arg_idx].tolist()
        # cones = np.array(cones)[arg_idx].tolist()
        # clses = np.array(clses)[arg_idx].tolist()
    if len(table_boxes_pic) == 0:
        return pp_tables
    radio = img.shape[0] / int(page.bbox[-1])
    table_boxes_page = list(
        np.array(
            np.array(copy.deepcopy(table_boxes_pic)) / radio,
            dtype=int,
        )
    )  # 还原到和page一个尺寸

    if len(pp_tables) > 0:  # 去重-如果两边都检测到了，保留pp检测到的
        iou_matrix = common.cal_iou_parallel(
            table_boxes_page,
            pp_tables_boxes,
            cal_type=-1,
        )
        retain_idx = list(np.all(np.array(iou_matrix < 0.7), axis=1))
        table_boxes_page = list(np.array(table_boxes_page)[retain_idx])
        table_boxes_pic = list(np.array(table_boxes_pic)[retain_idx])

    du_hypertxt = use_cv_struct_table(du_hypertxt, img, table_boxes_pic, page_index)
    cv_tables = []
    for idx, box in enumerate(du_hypertxt["data"]["table_info"]["tables_box"]):
        box = np.array(np.array(box) / radio, dtype=int).tolist()
        cur_table = list(
            du_hypertxt["data"]["table_info"]["tables_structure"][idx].values()
        )
        table_cells = np.array(
            copy.deepcopy(du_hypertxt["data"]["table_info"]["tables_cells"][idx])
        )
        table_cells[:, [0, 2]] += box[0]
        table_cells[:, [1, 3]] += box[1]
        table_cells = table_cells / radio
        table_cells = [tuple([float(int(num)) for num in row]) for row in table_cells]
        cv_tables.append(Table(page, table_boxes_page[idx], cur_table, table_cells))

    # cv_tables = []
    # for idx in range(len(table_boxes_pic)):
    #     box_pic, box_page = table_boxes_pic[idx], table_boxes_page[idx]
    #     (
    #         _,
    #         table_out_dict,
    #     ) = text_sys.table_struct.processes_tabele_line_noline(
    #         "None",
    #         copy.deepcopy(img),
    #         text_sys,
    #         configs,
    #         manual_assign_boxes=[box_pic],
    #         out_dir=None,
    #     )
    #     if len(table_out_dict) == 0:
    #         continue
    #     cur_table = [i[1] for i in table_out_dict[0][2].items()]
    #     cells = np.array(table_out_dict[0][1])
    #     cells[:, [0, 2]] += box_pic[0]
    #     cells[:, [1, 3]] += box_pic[1]
    #     cells = cells / radio
    #     cells = [tuple([float(int(num)) for num in row]) for row in cells]
    #
    #     cv_tables.append(Table(page, box_page, cur_table, cells))
    # 图像方法提取表格 -》end
    return pp_tables + cv_tables


def get_coordinates(sen):
    [a, b, c, d] = sen.strip().split("|")[0].strip().split(",")
    return [int(a), int(b), int(c), int(d)]


def parse_table(
    table,
    last_table,
    last_ori_table,
    v_offsets_sorted=None,
    h_offsets_sorted=None,
):
    """提取表格文本"""
    if hasattr(table, "cv"):
        # pil_image = table.page.to_image(resolution=200)  # 设置想要的分辨率，注意这里图片是放大过的，为了检测更好
        # img = np.array(pil_image.original)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # radio = img.shape[0] / int(table.page.bbox[-1])
        # table_box = list(
        #     np.array(np.array(list(table.bbox), dtype=int) * radio, dtype=int)
        # )
        # (
        #     _,
        #     table_out_dict,
        # ) = text_sys.table_struct.processes_tabele_line_noline(
        #     "None",
        #     copy.deepcopy(img),
        #     text_sys,
        #     configs,
        #     manual_assign_boxes=[table_box],
        #     out_dir=None,
        # )
        # cur_table = [i[1] for i in table_out_dict[0][2].items()]
        if v_offsets_sorted is None:
            v_offsets_sorted = sorted(
                set([c[1] for c in table.cells]).union(set(c[3] for c in table.cells))
            )
        if h_offsets_sorted is None:
            h_offsets_sorted = sorted(
                set([c[0] for c in table.cells]).union(set(c[2] for c in table.cells))
            )
        cur_table = table.cur_table
    else:
        cell_texts = table.extract()
        rows = table.rows
        assert len(rows) == len(cell_texts)
        cur_table = []

        if v_offsets_sorted is None:
            v_offsets_sorted = sorted(
                set([c[1] for c in table.cells]).union(set(c[3] for c in table.cells))
            )
        if h_offsets_sorted is None:
            h_offsets_sorted = sorted(
                set([c[0] for c in table.cells]).union(set(c[2] for c in table.cells))
            )

        for i, row in enumerate(rows):
            assert len(row.cells) == len(cell_texts[i])
            for j, (cell, cell_text) in enumerate(zip(row.cells, cell_texts[i])):
                if cell is None:
                    continue
                x0 = v_offsets_sorted.index(cell[1])
                y0 = h_offsets_sorted.index(cell[0])
                x1 = v_offsets_sorted.index(cell[3])
                y1 = h_offsets_sorted.index(cell[2])
                cell_bbox = [x0, y0, x1, y1]
                cur_table.append([cell_bbox, format_cell_text(cell_text)])

    cur_table = restore_border1(cur_table)
    if (not last_table) or (not last_ori_table):
        return cur_table
    else:
        return union_table(
            last_table, cur_table, last_ori_table, table, h_offsets_sorted
        )


def union_table(table1, table2, ori_table1, ori_table2, h_offsets_sorted):
    # ori_table指的是 包含坐标信息的table
    # 进行表格合并, 合并后，不影响原先的两个表格解析, 即使表格不能合并，合并后也不会造成不良影响。
    # 合并条件：如果第一个表格的最后一行列数 和 第二个表格第一行的列数相同，则进行合并。
    # 例如设单元格坐标表示方式为(左边界, 右边界)
    # first_table_row  = (0,2)(2,4)(4,7)(7,8)(9,11)
    # second_table_row  = (0,3)(3,5)(5,6)(6,8)(9,10)
    # 合并后： row = (0,3)(3,5)(5,8)(8,10)(10,12)
    end_x1 = max(cell[0][2] for cell in table1)  # end_x1是第一个表格最后一行的横坐标
    down_row = [
        cell[0] for cell in table1 if cell[0][2] == end_x1
    ]  # 第一页最下边的表格最后一行的坐标。
    up_row = [
        cell[0] for cell in table2 if cell[0][0] == 0
    ]  # 第二页最上边的表格第一行的坐标。

    table1, table2 = drop_duplicated_header(table1, table2, end_x1)
    if not table2:
        return table1

    if len(down_row) == len(up_row):
        # 合并条件：如果第一个表格的最后一行列数 和 第二个表格第一行的列数相同，则进行合并。
        # 处理两个表格的左右坐标
        table1, table2 = align_columns_by_rule(table1, down_row, table2, up_row)

    # 跨页表格列数不相同，基于坐标对齐表格列
    elif not isinstance(ori_table1, list) and not isinstance(ori_table2, list):
        # 如果两个表格都是table类型。
        table1, table2 = align_columns_by_coord(
            table1, table2, ori_table1, h_offsets_sorted
        )
        # 补齐表格2的边框
        table2 = restore_border1(table2)
        # 删除重复标题行
        table1, table2 = drop_duplicated_header(table1, table2, end_x1)

    # 针对（第一个表格第一列最后一行单元格, 第二个表格第一列第一行单元格）的特殊合并策略
    table1, table2, is_merged, merged_idx = merge_blank_cells(table1, table2, end_x1)
    # 针对（第一个表格最后一行单元格, 第二个表格第一行单元格）的通用合并策略
    table1 = detect_one_row(table1, table2, end_x1, is_merged, merged_idx)
    # 利用表格序号信息修复跨页合并单元格
    # table1 = union_by_id(table1, table2, end_x1)

    return table1


def union_by_id(table1, table2, end_x1):
    """
    如果第二个表格首行为空，但是第一个表格第一列存在连续的序号，第二个表格存在连续的序号，且序号在跨页处也是连续的。
    说明第二个表格第一行需要合并到第一个表格。
    :param table1:
    :param table2:
    :param end_x1:
    :return:
    """
    cell_merge_flag = False
    # 获取上一个表格最后一行第一列的文本
    text_table1 = list(filter(lambda x: x[0][2] == end_x1 and x[0][1] == 0, table1))[0][
        1
    ]
    # 获取当前表格第一行第一列的文本
    cell_table2 = list(filter(lambda x: x[0][0] == 0, table2))[0]
    text_table2 = cell_table2[1]
    if len(text_table2) == 0:
        text_table2 = list(
            filter(lambda x: x[0][0] == cell_table2[0][2] and x[0][1] == 0, table2)
        )[0][1]

    # 检查两个单元格的序号是不是数字，且是否相连
    id_table1 = int(text_table1)
    id_table2 = int(text_table2)

    if int(id_table1) + 1 == int(id_table2):
        cell_merge_flag = True


def align_columns_by_rule(table1, down_row, table2, up_row):
    down_row = sorted(down_row, key=lambda coord: coord[1])
    up_row = sorted(up_row, key=lambda coord: coord[1])

    for index, (up_cell, down_cell) in enumerate(zip(up_row, down_row)):
        bias = up_cell[3] - down_cell[3]
        if bias == 0:
            continue
        target = min(up_cell[3], down_cell[3])
        target_table = table1 if bias > 0 else table2
        bias = abs(bias)
        for j, coord in enumerate(target_table):
            if coord[0][1] >= target:
                target_table[j][0][1] += bias
            if coord[0][3] >= target:
                target_table[j][0][3] += bias

    return table1, table2


def align_columns_by_coord(table1, table2, ori_table1, h_offsets_sorted):
    last_table_h_offsets = sorted(
        set([c[0] for c in ori_table1.cells]).union(
            set([c[2] for c in ori_table1.cells])
        )
    )
    cur_table_h_offsets = sorted(set(h_offsets_sorted))
    offsets_tolerance = float(6)

    for idx, cur_offset in enumerate(cur_table_h_offsets):
        for last_offset in last_table_h_offsets:
            if abs(cur_offset - last_offset) <= offsets_tolerance:
                cur_table_h_offsets[idx] = last_offset
                break

    last_table_h_offsets = set(last_table_h_offsets)
    cur_table_h_offsets = set(cur_table_h_offsets)
    if (
        max(map(lambda x: x[0][3], table1)) == len(last_table_h_offsets) - 1
        and len(last_table_h_offsets & cur_table_h_offsets) != 0
    ):
        total_h_offsets = sorted(last_table_h_offsets | cur_table_h_offsets)

        for table, offset in (
            (table1, last_table_h_offsets),
            (table2, cur_table_h_offsets),
        ):
            idx = sorted([total_h_offsets.index(x) for x in offset])
            coord_map = {a: b for a, b in enumerate(idx)}
            for i, _ in enumerate(table):
                table[i][0][1] = coord_map.get(table[i][0][1], 0)
                table[i][0][3] = coord_map.get(table[i][0][3], 0)

    return table1, table2


def drop_duplicated_header(table1, table2, end_x1):
    table1_xlength, table2_xlength = table1[0][0][2], table2[0][0][2]
    first_row_text_table1 = list(
        map(lambda x: x[1], filter(lambda x: x[0][2] <= table1_xlength, table1))
    )
    first_row_text_table2 = list(
        map(lambda x: x[1], filter(lambda x: x[0][2] <= table2_xlength, table2))
    )
    last_row_text_table1 = list(
        map(lambda x: x[1], filter(lambda x: x[0][2] == end_x1, table1))
    )
    last_row_idx_table1 = []
    for idx, cell in enumerate(table1):
        if cell[0][2] == end_x1:
            last_row_idx_table1.append(idx)

    if first_row_text_table1 == first_row_text_table2:
        table2 = list(
            filter(lambda x: x[0][2] > table2_xlength, table2)
        )  # 删除第二个表格的表头
        # with open('header.txt', 'a', encoding='utf-8') as f:
        #     for header in first_row_text_table2:
        #         f.write(header + '\n')

        # print(f"删除了重复表头:表头宽度为{table2_xlength}, 表头内容为{first_row_text_table2}")
        for cell in table2:  # 其余行的x坐标减少table2_xlength(表头宽度)
            cell[0][0] -= table2_xlength
            cell[0][2] -= table2_xlength
        first_row_text_table2 = list(
            map(lambda x: x[1], filter(lambda x: x[0][2] <= table2_xlength, table2))
        )

    table1_not_finish, table2_need_head = False, False
    if len(last_row_idx_table1) == len(first_row_text_table2):
        for text1, text2 in list(zip(last_row_text_table1, first_row_text_table2)):
            (
                left_quotationmarks1,
                left_cn_brackets1,
                left_en_brackets1,
            ) = get_symbol_num_text(text1)
            (
                left_quotationmarks2,
                left_cn_brackets2,
                left_en_brackets2,
            ) = get_symbol_num_text(text2)
            table2_need_head = (
                (left_quotationmarks2 + left_cn_brackets2 + left_en_brackets2) < 0
                and left_quotationmarks2 <= 0
                and left_cn_brackets2 <= 0
                and left_en_brackets2 <= 0
                and (left_quotationmarks1 + left_quotationmarks2) == 0
                and (left_cn_brackets1 + left_cn_brackets2) == 0
                and (left_en_brackets1 + left_en_brackets2) == 0
                or (
                    5 > len(text2) > 2
                    and (
                        (text2[0] == "（" and text2[-1] == "）")
                        or (text2[0] == "(" and text2[-1] == ")")
                    )
                )
            )
            if table2_need_head:
                break
        table1_not_finish = any(
            [
                len(text) >= 3 and text[-1] in (",", "，", "/", "、", "(", "（", "《")
                for text in last_row_text_table1
            ]
        )
        if table2_need_head or table1_not_finish:
            for i in range(len(last_row_idx_table1)):
                # print(
                #     f'process_table:括号换行合并：{table1[last_row_idx_table1[i]][1]}    +    {first_row_text_table2[i]}')
                table1[last_row_idx_table1[i]][1] += first_row_text_table2[i]
            # print('\n')
            table2 = list(
                filter(lambda x: x[0][0] != 0, table2)
            )  # 删除第二个表格第一行
            for cell in table2:  # 其余行的x坐标减少1
                cell[0][0] -= 1
                cell[0][2] -= 1

    return table1, table2


def merge_blank_cells(table1: list, table2: list, end_x1) -> (list, list, int):
    """
    表格跨页的时候，一个合并单元格往往被分为两部分（合并单元格，空白单元格）
    在合并表格的时候，这个地方需要进行特殊处理，把空白单元格合并成原来的合并单元格。
    按照出现频率考虑，暂且只处理表格第一列出现合并单元格的情况
    :param end_x1:
    :param table1:
    :param table2:
    :return:
    """
    if not table2:
        return table1, table2, False, -1

    idx = -1
    is_merged = False

    # 获取第一个表格最后一行，第一个单元格
    merged_cell = ""
    for i, cell in enumerate(table1):
        if cell[0][2] == end_x1:
            merged_cell = cell
            idx = i
            break
    # 获取第二个表格第一行第一个单元格
    blank_cell = table2[0]

    if (
        blank_cell[0][0] == 0
        and (blank_cell[0][2] > 1 or merged_cell[0][0] < end_x1 - 1)
        and merged_cell[0][1] == blank_cell[0][1]
        and merged_cell[0][3] == blank_cell[0][3]
    ):
        if blank_cell[1] == "" or merged_cell[1] == "":
            # 说明出现了合并单元格，并且单元格内容为空白。
            # print(f'跨页单元格合并{merged_cell[1]}     +       {blank_cell[1]}')
            table1[idx][1] += blank_cell[1]
            table1[idx][0][2] += blank_cell[0][2]
            table2.pop(0)  # table2删除这个单元格
            is_merged = True
        else:
            up_merged_cell = []
            down_blank_cell = []
            for cell in table1:
                if (
                    cell[0][2] == merged_cell[0][0]
                    and cell[0][1] == merged_cell[0][1]
                    and cell[0][3] == merged_cell[0][3]
                ):
                    up_merged_cell = cell
                    break

            for cell in table2:
                if (
                    cell[0][0] == blank_cell[0][2]
                    and cell[0][1] == blank_cell[0][1]
                    and cell[0][3] == blank_cell[0][3]
                ):
                    down_blank_cell = cell
                    break
            merged_text = merged_cell[1] + blank_cell[1]
            if up_merged_cell:
                edit_distance_after_merge = Levenshtein.distance(
                    merged_text, up_merged_cell[1]
                )
                if edit_distance_after_merge < len(
                    merged_text
                ) * 0.3 and edit_distance_after_merge < Levenshtein.distance(
                    merged_cell[1], up_merged_cell[1]
                ):
                    # print(f'跨页单元格合并_结合文本编辑距离{merged_cell[1]}     +       {blank_cell[1]}')
                    table1[idx][1] += blank_cell[1]
                    table1[idx][0][2] += blank_cell[0][2]
                    table2.pop(0)  # table2删除这个单元格
                    is_merged = True
            elif down_blank_cell:
                edit_distance_after_merge = Levenshtein.distance(
                    merged_text, down_blank_cell[1]
                )
                if edit_distance_after_merge < len(
                    merged_text
                ) * 0.3 and edit_distance_after_merge < Levenshtein.distance(
                    blank_cell[1], down_blank_cell[1]
                ):
                    # print(f'跨页单元格合并_结合文本编辑距离{merged_cell[1]}     +       {blank_cell[1]}')
                    table1[idx][1] += blank_cell[1]
                    table1[idx][0][2] += blank_cell[0][2]
                    table2.pop(0)  # table2删除这个单元格
                    is_merged = True

    return table1, table2, is_merged, idx


def detect_one_row(table1, table2, end_x1, is_merged, merged_idx):
    # 判断两个跨页表格，在跨页处的两行单元格能否合并成一行单元格。
    # table1：第一页页末的表格
    # table2：第二页页首的表格
    # end_x1：第一页页末表格的最大的行坐标。

    table1_text, table2_text = [], []
    raw_text = []
    raw_text2 = []
    first_row_table2_x1 = []

    for cell in table1:
        if cell[0][2] == end_x1 and cell[0][0] <= end_x1 - 1:
            table1_text.append((cell[0][1], cell[1]))
        elif cell[0][2] == end_x1 - 1 and cell[0][0] <= end_x1 - 2:
            raw_text.append((cell[0][1], cell[1]))
    for cell in table2:
        if cell[0][0] == 0 and cell[0][2] >= 1:
            first_row_table2_x1.append((cell[0][1], cell[0][2]))
            table2_text.append((cell[0][1], cell[1]))
        elif cell[0][0] <= 1 and cell[0][2] >= 2:
            raw_text2.append((cell[0][1], cell[1]))

    # for text_name in ('table1_text', 'raw_text', 'first_row_table2_x1', 'table2_text', 'raw_text2'):
    table1_text = list(map(lambda x: x[1], sorted(table1_text, key=lambda x: x[0])))
    raw_text = list(map(lambda x: x[1], sorted(raw_text, key=lambda x: x[0])))
    first_row_table2_x1 = list(
        map(lambda x: x[1], sorted(first_row_table2_x1, key=lambda x: x[0]))
    )
    table2_text = list(map(lambda x: x[1], sorted(table2_text, key=lambda x: x[0])))
    raw_text2 = list(map(lambda x: x[1], sorted(raw_text2, key=lambda x: x[0])))

    # 如果列数不相同，直接将第二个表格拼接到第一个表格上。
    if len(table1_text) != len(table2_text):
        for cell in table2:
            cell[0][0] += end_x1
            cell[0][2] += end_x1
        table1.extend(table2)
        return table1

    table_text = list(map(lambda x: "".join(x), list(zip(table1_text, table2_text))))
    # 因为只判断中文内容的编辑距离，仅仅选择中文字符，对非中文的字符进行过滤。

    table1_text_han = [
        "".join(list(filter(lambda x: "9" < x or x < "0", text)))
        for text in table1_text
    ]
    table2_text_han = [
        "".join(list(filter(lambda x: "9" < x or x < "0", text)))
        for text in table2_text
    ]
    table_text_han = [
        "".join(list(filter(lambda x: "9" < x or x < "0", text))) for text in table_text
    ]
    raw_text_han = [
        "".join(list(filter(lambda x: "9" < x or x < "0", text))) for text in raw_text
    ]
    raw_text2_han = [
        "".join(list(filter(lambda x: "9" < x or x < "0", text))) for text in raw_text2
    ]

    cell_merge_flag = False
    if len(raw_text) == len(table_text) and len(table1_text) == len(table2_text):
        for i in range(len(table_text)):
            # 如果跨页的两个单元格能够合并，且合并后的语句长度大于10，且语句中文内容之间的差异比小于0.3, 则认为两个单元格能够进行合并
            distance_before_merge0 = Levenshtein.distance(
                table1_text_han[i], table2_text_han[i]
            )
            distance_before_merge1 = Levenshtein.distance(
                table1_text_han[i], raw_text_han[i]
            )
            distance_before_merge2 = Levenshtein.distance(
                table2_text_han[i], raw_text_han[i]
            )
            distance_after_merge = Levenshtein.distance(
                table_text_han[i], raw_text_han[i]
            )
            if (
                len(table_text_han[i]) >= 8
                and distance_after_merge
                < min(distance_before_merge1, distance_before_merge2)
                and (
                    (
                        distance_after_merge < len(table_text_han[i]) * 0.3
                        and distance_before_merge0 > len(table_text_han[i]) * 0.18
                    )
                    or (
                        distance_after_merge < len(table_text_han[i]) * 0.1
                        and distance_before_merge0 > len(table_text_han[i]) * 0.12
                    )
                )
            ):
                cell_merge_flag = True
                break
    if (
        not cell_merge_flag
        and len(raw_text2) == len(table_text)
        and len(table1_text) == len(table2_text)
    ):
        for i in range(len(table_text)):
            # 如果跨页的两个单元格能够合并，且合并后的语句长度大于10，且语句中文内容之间的差异比小于0.3, 则认为两个单元格能够进行合并
            distance_before_merge0 = Levenshtein.distance(
                table1_text_han[i], table2_text_han[i]
            )
            distance_before_merge1 = Levenshtein.distance(
                table1_text_han[i], raw_text2_han[i]
            )
            distance_before_merge2 = Levenshtein.distance(
                table2_text_han[i], raw_text2_han[i]
            )
            distance_after_merge = Levenshtein.distance(
                table_text_han[i], raw_text2_han[i]
            )
            if (
                len(table_text_han[i]) >= 8
                and distance_after_merge
                < min(distance_before_merge1, distance_before_merge2)
                and (
                    (
                        distance_after_merge < len(table_text_han[i]) * 0.3
                        and distance_before_merge0 > len(table_text_han[i]) * 0.18
                    )
                    or (
                        distance_after_merge < len(table_text_han[i]) * 0.1
                        and distance_before_merge0 > len(table_text_han[i]) * 0.12
                    )
                )
            ):
                cell_merge_flag = True
                break

    if not cell_merge_flag:
        idx = [1 if text != "" else 0 for text in table2_text]
        if sum(idx) == 1 and idx[0] == 0 and len(table2_text[idx.index(1)]) <= 4:
            cell_merge_flag = True
            # print(f'规则：一行中只有一个单元格存在内容，并且内容较短: \n{table1_text[idx.index(1)]}   +   {table2_text[idx.index(1)]} ')
        else:
            try:
                id1 = int(re.sub(r"[^0-9]", "", table1_text[0]))
                id2 = int(re.sub(r"[^0-9]", "", raw_text2[0]))
                if id1 + 1 == id2 and table2_text[0] == "":
                    cell_merge_flag = True
                    # print(f'\n因为跨页序号{table1_text[0]}  +  {raw_text2[0]}连续，合并单元格：')
            except Exception:
                cell_merge_flag = False

    if cell_merge_flag:
        table_idxs = []
        # 把第二个表格的第一行内容拼接到第一个的表格的末尾
        table_text_idx = 0
        for idx, cell in enumerate(table1):
            if cell[0][2] == end_x1:
                table_idxs.append((idx, cell[0][1]))

        table_idxs = sorted(table_idxs, key=lambda x: x[1])

        for table_idx in table_idxs:
            if table_text_idx < len(table_text):
                table1[table_idx[0]][1] = table_text[table_text_idx]
                table1[table_idx[0]][0][2] += first_row_table2_x1[table_text_idx] - 1
                # print(f"文本编辑距离拼接:{table1_text[table_text_idx]}  +   {table2_text[table_text_idx]}")
                table_text_idx += 1
        # print('\n')

        for cell in table2:
            if cell[0][0] != 0:
                cell[0][0] = cell[0][0] + end_x1 - 1
                cell[0][2] = cell[0][2] + end_x1 - 1
                table1.append(cell)
        if is_merged:
            table1[merged_idx][0][2] -= 1
    else:
        # 处理第二个表格的上下坐标。
        for cell in table2:
            cell[0][0] += end_x1
            cell[0][2] += end_x1
        table1.extend(table2)

    return table1


# TODO： 直接重写一个最小编辑距离计算函数，不使用Levenshtein.distance()，免得还需要安装一个库。
# 2023-01-06 发现自己写的最小编辑距离函数，相比标准库的最小编辑距离函数速度慢大约20倍。
# 写完之后测试发现速度比Levenshtein库慢20倍左右。

# def min_edit_distance(s1, s2):
#     dp = [[0 for i in range(len(s1))] for j in range(len(s2))]
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             if i == 0 or j == 0:
#                 dp[i][j] = i+j
#             else:
#                 fx = 0 if s1[i-1] == s2[j-1] else 1
#                 dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1 , dp[i-1][j-1]+fx)
#     return dp[-1][-1]


def get_symbol_num_text(text):
    quotationmarks_diff = text.count("《") - text.count("》")
    cn_brackets_diff = text.count("（") - text.count("）")
    en_brackets_diff = text.count("(") - text.count(")")
    return quotationmarks_diff, cn_brackets_diff, en_brackets_diff


def restore_border2(cur_table):
    # 获取max_x1, max_y1
    min_x0, max_x1, min_y0, max_y1 = 0, 0, 0, 0
    for cell in cur_table:
        max_x1 = max(max_x1, cell[0][2])
        max_y1 = max(max_y1, cell[0][3])

    # 根据min_x0, max_x1, min_y0, max_y1补全单元格
    tmp_table = copy.deepcopy(cur_table)
    # for cell in tmp_table:
    #     if


def restore_border1(cur_table):
    # 按照边，补足cur_table

    tmp_table = copy.deepcopy(cur_table)
    add_list = []
    max_y1 = max(cell[0][2] for cell in cur_table)
    max_x1 = max(cell[0][3] for cell in cur_table)
    row_visible = [0 for i in range(max_y1)]
    for i, cell in enumerate(tmp_table):
        flag = True
        for j in range(cell[0][0], cell[0][2]):
            if row_visible[j] == 1:
                flag = False
                break

        if flag:
            for j in range(cell[0][0], cell[0][2]):
                row_visible[j] = 1
            if cell[0][1] != 0:
                if len(add_list) > 0:
                    coord = add_list[-1][1][0]
                    # 如果上一个添加的单元格和现在要添加的单元格能够合并，则进行合并。
                    if coord[2] == cell[0][0] and coord[3] == cell[0][1]:
                        add_list[-1][1] = [[coord[0], 0, cell[0][2], coord[3]], ""]
                    else:
                        add_list.append(
                            [i, [[cell[0][0], 0, cell[0][2], cell[0][1]], ""]]
                        )
                else:
                    add_list.append([i, [[cell[0][0], 0, cell[0][2], cell[0][1]], ""]])

    for idx, item in enumerate(add_list):
        cur_table.insert(item[0] + idx, item[1])
        # print(f'修补左侧单元格{item[1]}')

    cur_table = sorted(cur_table, key=lambda x: (x[0][0], -x[0][1]))
    tmp_table = copy.deepcopy(cur_table)
    add_list = []
    row_visible = [0 for i in range(max_y1)]
    for i, cell in enumerate(tmp_table):
        flag = True
        for j in range(cell[0][0], cell[0][2]):
            if row_visible[j] == 1:
                flag = False
                break

        if flag:
            for j in range(cell[0][0], cell[0][2]):
                row_visible[j] = 1
            if cell[0][3] != max_x1:
                if len(add_list) > 0:
                    coord = add_list[-1][1][0]
                    # 如果上一个添加的单元格和现在要添加的单元格能够合并，则进行合并。
                    if coord[2] == cell[0][0] and coord[1] == cell[0][1]:
                        add_list[-1][1] = [[coord[0], coord[3], cell[0][2], max_x1], ""]
                    else:
                        add_list.append(
                            [i, [[cell[0][0], cell[0][3], cell[0][2], max_x1], ""]]
                        )
                else:
                    add_list.append(
                        [i, [[cell[0][0], cell[0][3], cell[0][2], max_x1], ""]]
                    )

    for idx, item in enumerate(add_list):
        cur_table.insert(item[0] + idx, item[1])
        # print(f'修补右侧单元格{item[1]}')

    cur_table = sorted(cur_table, key=lambda x: (x[0][0], x[0][1]))

    return cur_table
