#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2022 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#
# Author: Li Xiuming
# Last Modified: 2022-03-15
############################################################

import math


def edit_distance(word1, word2):
    """
    计算编辑距离

    Args:
        word1 (str): 字符串
        word2 (str): 字符串

    Returns:
        distance (int): 编辑距离
    """

    m = len(word1)
    n = len(word2)
    dp = [[float("inf") for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for i in range(n + 1):
        dp[0][i] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1

    distance = dp[m][n]
    return distance


def find_lcs(s1, s2):
    """
    查找两个字符串之间的最长公共子序列

    Args:
        s1 (str): 字符串
        s2 (str): 字符串
    Returns:
        cs (str): s1和s2的最长公共子序列
        length (int): s1和s2的长度
    """
    res = [[0 for i in range(len(s1) + 1)] for j in range(len(s2) + 1)]
    idxs = []
    for i in range(1, len(s2) + 1):
        for j in range(1, len(s1) + 1):
            if s2[i - 1] == s1[j - 1]:
                res[i][j] = res[i - 1][j - 1] + 1
                idxs.append(j - 1)
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    idxs = sorted(list(set(idxs)))
    cs = "".join([s1[idx] for idx in idxs])
    length = res[-1][-1]
    return cs, length


def compute_softmax(scores):
    if len(scores) == 0:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def split_windows(context, window_size, overlap=0.25):
    if len(context) <= window_size:
        return [context], [0]
    contexts = []
    start = 0
    context_starts = []
    while start < len(context):
        context_starts.append(start)
        contexts.append(context[start : start + window_size])
        start = start + int((1 - overlap) * window_size)
    return contexts, context_starts


class dic2class:
    def __init__(self, dic):
        for k, v in dic.items():
            self.__setattr__(k, v)


def remove_list_duplicate(ls, key=""):
    """去重后保持顺序一致"""

    i = 0
    while i < len(ls):
        if key:
            if ls[i][key] in [item[key] for item in ls[:i]]:
                ls.pop(i)
                continue
        else:
            if ls[i] in ls[:i]:
                ls.pop(i)
                continue
        i += 1
    return ls
