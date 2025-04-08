#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2022 SenseDeal AI, Inc. All Rights Reserved
#
# Description: 前缀树
#
# Author: Li Xiuming
# Last Modified: 2022-03-15
############################################################

import collections


class TrieNode:
    def __init__(self):
        # 子节点。{value: TrieNode}
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self, use_single=True):
        # 创建空的根节点
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        """插入一个单词序列"""
        current = self.root # 当前节点
        deep = 0
        for letter in word: # 遍历单词的每一个字符
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            # 没找到
            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, word, space=""):
        """
        Args:
            word: 需要匹配的词
        Return:
            返回匹配的词, 如果存在多字词，则会筛去单字词
        """
        word = [c for c in word]

        matched = []
        while len(word) > self.min_len:
            if self.search(word):
                matched.insert(0, space.join(word[:])) # 短的词总是在最前面
            del word[-1]

        if len(matched) > 1 and len(matched[0]) == 1: # filter single character word
            matched = matched[1:]

        return matched

