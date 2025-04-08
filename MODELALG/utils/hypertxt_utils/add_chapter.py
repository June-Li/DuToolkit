# -*- coding:utf-8 -*-
############################################################
#
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#   为解析后的pdf文件添加章节信息，只能处理带单个编号和单个字母的章节信息
#   适用于同时满足以下条件的情况：
#   1、一级标题样式唯一且编号连续
#
#   以后：
#   加入字体判断信息
#
# Author: Li Xiuming, Shi Wenbao, Xu Zhiang
# Last Modified: 2022-09-05
############################################################

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../")
sys.path.append(os.path.abspath(root_dir))

import logging

import numpy as np
from Levenshtein import (
    distance as levenshtein_distance,  # pip install python-Levenshtein
)
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.ERROR)
import cn2an
import regex as re
from treelib import Tree

from MODELALG.utils import common
from WORKFLOW.OTHER.llm_api.v0 import llm_processor

from .chapter_settings import (
    CHAPTER_COMMON_PATTERNS,
    CHAPTER_CONFIG,
    CHAPTER_NUM_PATTERNS,
    CUSTOMIZED_PATTERNS,
    customized_text_post,
)


class Chapter:
    def __init__(
        self,
        text,
        context_id=None,
        rule_id=None,
        num_type=None,
        num=None,
        hierarchy="[DEFAULT]",
        **kwargs,
    ):
        self.context_id = context_id
        self.text = text
        self.rule_id = rule_id
        self.hierarchy = hierarchy
        self.num_type = num_type
        self.num = num
        for k, v in kwargs.items():
            self.__setattr__(v, k)

    def parse_text(self):
        """处理标题文本"""
        text = self.text
        for post in customized_text_post:
            if re.search(post["pattern"], text):
                text = re.sub(post["sub_pattern"], "", text)
                break
        self.text = text


class MakeChapter:
    def __init__(self, configs, catalog=None):
        self.chapter_rules = self.conform_rules(catalog=catalog)
        self.chapter_tree = Tree()
        self.chapter_tree.create_node(
            identifier="0", tag="1", data=Chapter(text="[CHAPTER_ROOT]")
        )
        self.configs = configs

    def apply(self, ori_hypertxt, is_print=False):
        # 获取章节
        use_type = "pattern"
        if use_type == "pattern":
            chapters = self.get_chapter_only_by_pattern(ori_hypertxt["context"])
        elif use_type == "llm":
            try:
                chapters = self.get_chapter_only_by_llm(
                    ori_hypertxt["context"], self.configs
                )
            except Exception as e:
                chapters = self.get_chapter_only_by_pattern(ori_hypertxt["context"])
        else:
            raise ValueError(f"use_type must be 'pattern' or 'llm', but got {use_type}")

        if len(chapters) == 0:
            ori_hypertxt["chapters"] = {}
            return ori_hypertxt

        # 建树
        self.build_tree_only_by_pattern(chapters)
        self.add_tag()

        # 打印
        if is_print:
            self.print_tree(data_property="text")
            if "hierarchy" in CHAPTER_CONFIG:
                self.print_tree(
                    data_property="text",
                    filter=lambda x: self.chapter_tree.depth(x)
                    < CHAPTER_CONFIG["hierarchy"],
                )
            else:
                self.print_tree(data_property="text")

        # 过滤
        cid2chapter_list = []
        for nid in self.chapter_tree.expand_tree(mode=Tree.DEPTH, sorting=False):
            if "hierarchy" in CHAPTER_CONFIG:
                if (
                    self.chapter_tree.depth(self.chapter_tree.get_node(nid))
                    >= CHAPTER_CONFIG["hierarchy"]
                ):
                    continue
            cid2chapter_list.append(
                (self.chapter_tree[nid].tag, self.chapter_tree[nid].data)
            )
        if len(cid2chapter_list) == 0:
            ori_hypertxt["chapters"] = {}
            return ori_hypertxt

        if "max_len" in CHAPTER_CONFIG:
            cid2chapter_list = [
                (cid, v)
                for cid, v in cid2chapter_list
                if len(cid.split(".")) <= CHAPTER_CONFIG["max_len"]
            ]
        cid2chapter_list = sorted(
            cid2chapter_list,
            key=lambda x: (
                getattr(x[1], "context_id") if getattr(x[1], "context_id") else 0
            ),
        )

        # 再次遍历赋予每个段落cid
        for context_idx, context in enumerate(ori_hypertxt["context"]):
            new_cid = None
            for i, (cid, data) in enumerate(cid2chapter_list):
                if data.text == "[CHAPTER_ROOT]":
                    continue
                if context_idx < data.context_id:
                    new_cid = cid2chapter_list[i - 1][0]
                    break
                elif (
                    i == len(cid2chapter_list) - 1
                    or data.context_id
                    <= context_idx
                    < cid2chapter_list[i + 1][1].context_id
                ):
                    new_cid = cid
                    break
            assert new_cid is not None
            ori_hypertxt["context"][context_idx]["cid"] = new_cid

        # 对cid2chapter的文本进行处理
        cid2chapter = {}
        for cid, chapter in cid2chapter_list:
            chapter.parse_text()
            cid2chapter[cid] = chapter.text
        ori_hypertxt["chapters"] = cid2chapter
        # print("cid2chapter", json.dumps(cid2chapter, ensure_ascii=False, indent=4))
        return ori_hypertxt

    def print_tree(self, data_property=None, filter=None):
        self.chapter_tree.show(data_property=data_property, filter=filter)

    def add_tag(self):
        def preorder(node):
            if not node:
                return
            ptag = node.tag
            # self.chapter_tree.children(node.identifier).sort(key=lambda x: x.data.context_id)
            children = self.chapter_tree.children(node.identifier)
            children = sorted(children, key=lambda x: x.data.context_id)
            for i, cur_node in enumerate(children):
                cur_node.tag = ptag + "." + str(i + 1)
                preorder(cur_node)

        preorder(self.chapter_tree.get_node(self.chapter_tree.root))

    def conform_rules(self, catalog=None):
        # 目前只用到了规则
        chapter_patterns = CHAPTER_COMMON_PATTERNS
        if catalog is not None:
            # 获取目录相关的章节信息，目前只用了目录的文本，以后可以用到目录上字体样式
            chapter_patterns.extend([c["text"] for c in catalog])
        chapter_patterns.extend(CUSTOMIZED_PATTERNS)

        chapter_rules = [
            {"pattern": {"[DEFAULT]": p} if isinstance(p, str) else p}
            for p in chapter_patterns
        ]  # 兼容定制的和通用的规则
        return chapter_rules

    def build_tree_only_by_pattern(self, chapters):
        self.chapter_tree.create_node(identifier="1", parent="0", data=chapters[0])
        for idx, chapter in enumerate(chapters[1:]):
            nid = idx + 2
            if chapter.hierarchy == "first":
                self.chapter_tree.create_node(
                    identifier=str(nid), parent="0", data=chapter
                )
            elif chapter.hierarchy == "second":
                for child_id in self.chapter_tree.children(0)[::-1]:
                    data = self.chapter_tree.get_node(child_id).data
                    if data.rule_id == chapter.rule_id and data.hierarchy == "first":
                        self.chapter_tree.create_node(
                            identifier=str(nid), parent=child_id, data=chapter
                        )
                        break
            else:
                found = False
                parent = self.chapter_tree.parent(str(nid - 1))
                while parent is not None:
                    parent_nid = parent.identifier
                    silbing = None
                    for node in self.chapter_tree.children(parent_nid)[::-1]:
                        if node.data.hierarchy == "[DEFAULT]":
                            silbing = node
                            break

                    if silbing is not None:
                        if (
                            silbing.data.rule_id == chapter.rule_id
                            and silbing.data.num_type == chapter.num_type
                            and chapter.num - silbing.data.num in [1, 2]
                        ):
                            self.chapter_tree.create_node(
                                identifier=str(nid), parent=parent_nid, data=chapter
                            )
                            found = True
                            break
                    parent = self.chapter_tree.parent(parent_nid)

                if not found:
                    pid = (
                        "0"
                        if self.chapter_tree.depth(str(nid - 1)) == 1
                        and silbing is None
                        else str(nid - 1)
                    )
                    self.chapter_tree.create_node(
                        identifier=str(nid), parent=pid, data=chapter
                    )

    def get_chapter_only_by_pattern(self, contexts):
        """获取能匹配到章节正则的所有段落"""
        chapters = []
        for context_id, context in enumerate(contexts):
            if context["type"] != "text":
                continue
            text = context["text"]

            # 对于每个规则
            flag = False
            for rule_id, rule in enumerate(self.chapter_rules):
                # 对于每个规则里的正则
                for hierarchy, pat in rule["pattern"].items():
                    if re.search(pat, text):
                        num, num_type = 0, "[STR]"  # 默认值
                        num_rs = re.findall(pat, text)
                        # 找到编号和编号类型
                        if len(num_rs) == 1:
                            for (
                                num_type_key,
                                num_type_pattern,
                            ) in CHAPTER_NUM_PATTERNS.items():
                                if re.search("^" + num_type_pattern + "$", num_rs[0]):
                                    num_type = num_type_key
                                    if num_type_key == "chn":
                                        num = cn2an.cn2an(num_rs[0])
                                    elif num_type_key == "num":
                                        num = int(num_rs[0])
                                    elif num_type_key in ["uletter", "lletter"]:
                                        num = ord(
                                            num_rs[0]
                                        )  # 如果是单个字符，用ascii码代替
                                    break

                        if hierarchy != "[DEFAULT]" or (
                            hierarchy == "[DEFAULT]" and num != 0
                        ):
                            chapter = Chapter(
                                text, context_id, rule_id, num_type, num, hierarchy
                            )
                            chapters.append(chapter)
                            flag = True
                            break

                if flag:
                    break
        return chapters

    def get_chapter_only_by_llm(self, contexts, configs):
        # 规整出文本列表和相对contexts的索引 -> start
        text_list = []
        context_idx_list = []
        for idx_0, elem in enumerate(contexts):
            if elem["type"] != "text":
                continue
            text_list.append(elem["text"])
            context_idx_list.append(idx_0)
        # 规整出文本列表和相对contexts的索引 -> end

        # 构建chapter -> start
        chapters = []
        step = 50
        for idx_0 in range(0, len(text_list), step):
            text_fragment_list = text_list[idx_0 : idx_0 + step]
            text_fragment = "\n".join(text_fragment_list)
            context_idx_fragment_list = context_idx_list[idx_0 : idx_0 + step]
            Q = (
                open(root_dir + "/prompts/chapter_prompt_v0.txt", "r").read()
                + text_fragment
            )
            A = llm_processor.get_answer(
                question=Q,
                question_type="text",
                task_description="你是ChatGPT, 一个由OpenAI训练的大型语言模型, 你旨在回答并解决人们的任何问题，并且可以使用多种语言与人交流。",
            )
            c_list = eval(A)
            is_chapter_idx_list = self.find_best_matches(text_fragment_list, c_list)[0]

            for idx_1 in is_chapter_idx_list:
                num_type = (
                    lambda s: (
                        "num"
                        if any(c.isdigit() for c in s[:6])
                        else (
                            "chn"
                            if any(c in "一二三四五六七八九十" for c in s[:6])
                            else "lletter"
                        )
                    )
                )(text_fragment_list[idx_1])
                to_arabic_text = re.sub(
                    r"\d+|[零一二三四五六七八九拾佰仟万]+",
                    self.to_arabic,
                    text_fragment_list[idx_1],
                )
                num = next(
                    (
                        match.group()
                        for match in re.finditer(r"\d+", to_arabic_text)
                        if match
                    ),
                    None,
                )
                chapter = Chapter(
                    text_fragment_list[idx_1],
                    context_idx_fragment_list[idx_1],
                    -1,
                    num_type,
                    num,
                    hierarchy="[DEFAULT]",
                )
                chapters.append(chapter)
        # 构建chapter -> end

        return chapters

    def find_best_matches(self, A, B, threshold=0.65):
        """寻找A和B之间最佳匹配"""
        if not A or not B:
            return {}

        cost_matrix = self.calculate_similarity_matrix(A, B)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # matches = {}
        # for row, col in zip(row_ind, col_ind):
        #     similarity = 1 - cost_matrix[row, col]
        #     if similarity >= threshold:  # 只考虑相似度大于等于阈值的配对
        #         matches[A[row]] = B[col]

        # return matches

        row_idx_filter = []
        col_idx_filter = []
        for row, col in zip(row_ind, col_ind):
            similarity = 1 - cost_matrix[row, col]
            if similarity >= threshold:  # 只考虑相似度大于等于阈值的配对
                row_idx_filter.append(row)
                col_idx_filter.append(col)
        return row_idx_filter, col_idx_filter

    def calculate_similarity_matrix(self, A, B):
        """计算A和B之间所有组合的相似度矩阵"""
        matrix = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                # 使用编辑距离计算相似度，转换为百分比形式，并转换为成本
                dist = levenshtein_distance(a, b)
                max_len = max(len(a), len(b))
                if max_len == 0:
                    similarity = 100  # 如果两个字符串都是空，则认为完全匹配
                else:
                    similarity = (1 - dist / max_len) * 100
                cost = self.similarity_to_cost(similarity)
                matrix[i, j] = cost  # 不再使用np.inf，而是保留高成本
        return matrix

    def similarity_to_cost(self, similarity):
        """将相似度转换为成本"""
        return 1 - similarity / 100

    def to_arabic(self, match):
        text = match.group()
        if text.isdigit():  # 如果已经是阿拉伯数字，则保持原样
            return text
        try:
            return str(cn2an.cn2an(text, "smart"))
        except ValueError:
            return text


if __name__ == "__main__":
    import json

    # # type, text是必要字段
    # contexts = [
    #     {"type": "text", "text": "我们", "metadata": {}},
    #     {"type": "text", "text": "第一章 我们", "metadata": {}},
    #     {"type": "text", "text": "调查人员声明", "metadata": {}},
    #     {"type": "text", "text": "1. 你们", "metadata": {}},
    #     {"type": "text", "text": "2. 他们", "metadata": {}},
    #     {"type": "text", "text": "第二章 你们", "metadata": {}},
    #     {"type": 'table', "text": {"text": [[1,2,3], "单元格文本"]}, "metadata": {}},
    #     {"type": 'text', "text": "文本1", "metadata": {}},
    #     {"type": "text", "text": "1、段落测试", "metadata": {}},
    #     {"type": 'text', "text": "文本2", "metadata": {}},
    #     {"type": "text", "text": "实地调查", "metadata": {}},
    #     {"type": 'text', "text": "文本3", "metadata": {}}
    # ]
    # # 可选
    # # catalog = [
    # #     {"text": "目录"},
    # #     {"text": "第一章 XXX"},
    # #     {"text": "第二章 XXX"},
    # # ]
    # hypertxt = {"context": contexts}

    with open(
        "/home/lixm/mycodes/dev_lib/授信调查报告.1_81829F803CE865E9A628B1529801E5DA.txt",
        "r",
    ) as f:
        hypertxt = json.load(f)

    maker = MakeChapter()
    hypertxt = maker.apply(hypertxt)
    print(
        json.dumps(hypertxt, ensure_ascii=False, indent=4, default=common.convert_int64)
    )
