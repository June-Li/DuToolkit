import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
import json
import re

import numpy as np

from MODELALG.utils import common
from MODELALG.utils.common import Log
from WORKFLOW.OTHER.llm_api.v0.llm_processor import get_answer

logger = Log(__name__).get_logger()


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

    # llm生成章节
    if len(title_list) != 0:
        prompt_ori = str(
            open(
                root_dir
                + du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"]["chapter_llm"][
                    "prompt_file_path"
                ],
                "r",
                encoding="utf-8",
            ).read()
        )

        # 大模型生成章节
        failed_mesage = "None"
        for i in range(2):
            if failed_mesage == "None":
                prompt = prompt_ori + "\n" + str(title_list)
            elif failed_mesage == "list":
                prompt = prompt_ori + (
                    "\n"
                    + "再次强调：输出的一定得是一个列表，因为我要用eval将字符串转换成列表。\n"
                    + failed_mesage
                )
            result = get_answer(
                prompt,
                "text",
                url=du_hypertxt["configs"]["API_BASE"],
                model=du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"][
                    "chapter_llm"
                ]["model"],
                max_retries=du_hypertxt["configs"]["PROCESSES_CONTROL"]["Chapter"][
                    "chapter_llm"
                ]["max_retries"],
            )

            try:
                result = eval(result)
                break
            except:
                failed_mesage = "list"
                if i == 1:
                    raise Exception("无法将大模型返回结果转为list，生成章节失败")

        # 修正result，每个元素都有两个元素
        for idx_0, elem in enumerate(result):
            if len(elem) != 2:
                result[idx_0] = [None, ""]

        # 如果result长度和title_list长度不一致，则进行匹配
        if len(result) != len(title_list):
            row_idx_filter, col_idx_filter = common.find_best_matches(
                title_list, np.array(result)[:, 1].tolist()
            )
            result_fixed = [[None, ""] for i in range(len(title_list))]
            for idx_0, _ in enumerate(row_idx_filter):
                result_fixed[row_idx_filter[idx_0]] = [
                    result[col_idx_filter[idx_0]][0],
                    title_list[row_idx_filter[idx_0]],
                ]
            result = result_fixed

        # 将章节信息加入到sd_hypertxt
        for idx_0, elem in enumerate(result):
            if elem[0] is not None:
                sd_hypertxt["context"][title_id_list[idx_0]]["cid"] = elem[0]
    else:
        result = []

    # 生成sid、pid、cid
    sid, pid, cid = 1, 1, 0
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
    sd_hypertxt["chapters"] = dict(result)
    du_hypertxt["hypertxt"]["sd_hypertxt"] = json.dumps(
        sd_hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64
    )
    return du_hypertxt
