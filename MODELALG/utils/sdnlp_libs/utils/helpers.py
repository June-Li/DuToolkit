#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from loguru import logger


def recover_hypertxt_from_table(hypertxt):
    """
    将表格信息转换为文本信息，可以处理合并单元格情况
    """
    total_cell_num = 0
    new_hypertxt = copy.deepcopy(hypertxt)
    for idx, context in enumerate(hypertxt["context"]):
        try:
            if context["type"] == "table":
                max_x = max(cell[0][2] for cell in context["text"])
                max_y = max(cell[0][3] for cell in context["text"])
                new_table = [[""] * max_y for _ in range(max_x)]
                for cell in context["text"]:
                    for i in range(cell[0][0], cell[0][2]):
                        for j in range(cell[0][1], cell[0][3]):
                            new_table[i][j] = cell[1]

                for i in range(max_x):
                    new_row = "\t".join(new_table[i])
                    new_hypertxt["context"].insert(
                        idx + total_cell_num,
                        {"text": new_row, "type": "text", "pid": 1, "sid": 1, "cid": context["cid"],
                         "metadata": context['metadata']
                         }
                    )
                    total_cell_num += 1
        except Exception as e:
            logger.exception(e)
            continue
    return new_hypertxt
