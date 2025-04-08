#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from decimal import Decimal

# 徐志昂2023-01-13修改intersection_x_tolerance为2.1
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 5,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    # "keep_blank_chars": False,
    "text_tolerance": 3,
    "text_x_tolerance": None,
    "text_y_tolerance": None,
    "intersection_tolerance": 3,
    "intersection_x_tolerance": float(5),
    "intersection_y_tolerance": float(5),
}

X_TOLERANCE = 3
PARA_TOLERANCE_PERCENT = 0.05  # 分段参考的句子长度比例
PAGE_PARA_TOLERANCE_PERCENT = 0.8  # 页面分段右边界的最大比例
PAGENUM_TOLERANCE = 0.1  # 在页面末尾10%以后
HEADER_TOLERANCE = 0.1  # 在页面末尾10%以后
LINE_CHAR_TOLERANCE = 3  # 段间距
START_EN_LENGTH = 8  # 行前的英文长度，大于该值，不分段
START_NUM_LENGTH = 7  # 行前的数字长度，大于该值，不分段
LINE_MIN_X1_P = 0.5  # 一行长度的最小值比例系数  page_width * LINE_MIN_X1_P 可以获取行最短长度

# 定义章节编号的样式类型
PATTERNS = [
    r"(目录)",
    r"([一二三四五六七八九十]{1,3})、",  # 一、
    r"[\(]{1}([一二三四五六七八九十]{1,3})[\)]{0,1}",  # (一)
    r"[（]{1}([一二三四五六七八九十]{1,3})[）]{0,1}",  # （一）
    r"第([一二三四五六七八九十]{1,3})节",  # 第一节
    r"第([一二三四五六七八九十]{1,3})章",  # 第一节
    r"第([一二三四五六七八九十]{1,3})条",  # 第一节
    r"(\d{1,2})、",  # 1、
    r"\((\d{1,2})\)",  # (1)
    r"（(\d{1,2})）",  # （1）
    r"(\d{1,2})．",  # 1．中文点
    r"(\d{1,2}). ",  # 1．中文点
    r"(调查人员声明)",
    r"(实地调查)",
    r"(外围信息采集及其他方式)",
    r"(营销背景)",
    r"附件([一二三四五六七八九十]{1,3})",
    r"(特别风?险?提示)",
    r"(重要内?容?提示)",
    r"(声明|释义)",
    r"(重大风险提示|重大事项提示)",
]
# 定义关键信息章节在样式中索引，关键信息章节会被提升到最高层级
KEYINFO_PATTERNS_IDXS = [
    PATTERNS.index(r"(特别风?险?提示)"),
    PATTERNS.index(r"(重要内?容?提示)"),
    PATTERNS.index(r"(重大风险提示|重大事项提示)"),
    PATTERNS.index(r"(声明|释义)"),
    PATTERNS.index(r"(调查人员声明)"),
    PATTERNS.index(r"(实地调查)"),
    PATTERNS.index(r"(外围信息采集及其他方式)"),
    PATTERNS.index(r"(营销背景)"),
]
# 章节抽取配置
CHAPTER_CONFIG = {
    "use_catalog": True,  # 使用使用目录信息协助抽取
    "chapter_max_len": 100,  # 章节文本的最大长度
    "layers": 3,  # 限制章节的层级
    "chapter_patterns": PATTERNS,  # 编号样式
    "keyinfo_patterns_idxs": KEYINFO_PATTERNS_IDXS,  # 单独编号
    "chapter_text_size_gap": 1.0,  # 仅在无法判断章节层级关系时使用
}

# 新添加的配置信息，因为抽取到的一些章节文本包含额外的信息，
# 二、银行总授信情况 单位：万元 √/ 万美元 □
# 这些章节文本就需要我们进行过滤。
NEED_DELETE_CONFIG = [
    r" 单位.*美元 □",  # 授信调查报告pdf中出现了单位和表格在同一行的情况
]
# 即开头不能包含的信息。
# 如果章节开头包含这些信息，则不认为是一个新章节。
FORBIDDEN_START_CONFIG = [
    r"4、威胁",
    r"3、机遇",
    r"2、劣势",
    r"1、优势",
]
