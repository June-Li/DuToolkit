# -*- coding:utf-8 -*-
############################################################
#
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#   章节抽取配置
#
# Author: Li Xiuming, Shi Wenbao, Xu Zhiang
# Last Modified: 2022-09-05
############################################################

# ==========================================定制化
# 定制化的配置（示例：宁波银行）
customized_patterns = [{"first": "^调查人员声明|^实地调查|^外围信息采集及其他方式|^营销背景"}]

customized_text_post = [
    {
        "pattern": ".+",
        "sub_pattern": " *单位.*?(?:人迷你|美元)",  # 授信调查报告pdf中出现了单位和表格在同一行的情况
        # ，如"二、银行总授信情况 单位：万元 √/ 万美元 □"
    }
]

# 抽取配置
CHAPTER_CONFIG = {
    "max_len": 100,  # 章节文本的最大长度
    "hierarchy": 6,  # 限制章节的层级
    "chapter_text_size_gap": 1.0,  # 仅在无法判断章节层级关系时使用
}

# ==========================================公告通用
notice_patterns = [
    {
        "first": "^(?:特别|重要|重大)(?:风险|内容|事项)提示|^声明|^释义|^附件[一二三四五六七八九十]{1,3}",
    }
]

CUSTOMIZED_PATTERNS = notice_patterns + customized_patterns

# ==========================================通用
CHAPTER_NUM_PATTERNS = {
    "chn": "[一二三四五六七八九十百零]",
    "num": "[0-9]",
    "uletter": "[A-Z]",
    "lletter": "[a-z]",
}

CHAPTER_COMMON_PATTERNS = [
    r"^目录",
    r"^第(.{1,3})节",  # 第一节
    r"^第(.{1,3})章",  # 第一章
    r"^第(.{1,3})条",  # 第一条
    r"^第(.{1,3})部分",  # 第一部分
    r"^(.{1,3})、",  # 一、
    r"^\(.{1,3}\)",  # (一)
    r"^（(.{1,3})）",  # （一）
    r"^([0-9]{1,2})、",  # 1、
    r"^\(([0-9]{1,2})\)",  # (1)
    r"^（([0-9]{1,2})）",  # （1）
    r"^([0-9]{1,2})\)",  # 1)
    r"^([0-9]{1,2})）",  # 1）
    r"^([0-9]{1,2})．",  # 1．中文点
    r"^([0-9]{1,2}). ",  # 1．英文点+空格
    r"^([0-9]{1,2})\.",  # 1. 英文点无空格
]
