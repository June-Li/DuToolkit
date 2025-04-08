#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################
#                                                           
# Copyright (C) 2020 SenseDeal AI, Inc. All Rights Reserved 
#    
# Description:
#   pass
#
# Author: Li Xiuming
# Last Modified: 2020-10-29                                                 
############################################################

import re
import copy


class fuzzy_set():
    def __init__(self, elements, tolerance=2):
        self.elements = set(elements)
        self.tolerance = tolerance
        self.sorted_list = sorted(list(self.elements))

    def add(self, e):
        distance = min(list(map(lambda x: abs(x - e), self.elements)))
        if distance > self.tolerance:
            self.elements.add(e)
            self.sorted_list = sorted(list(self.elements))

    def union(self, other):
        res = copy.deepcopy(self.elements)
        for e in other.elements:
            distance = min(list(map(lambda x: abs(x - e), self.elements)))
            if distance > self.tolerance:
                res.add(e)
        return fuzzy_set(res, self.tolerance)

    def intersection(self, other):
        res = set()
        for e in other.elements:
            distance = min(list(map(lambda x: abs(x - e), self.elements)))
            if distance <= self.tolerance:
                res.add(e)
        return fuzzy_set(res, self.tolerance)

    def index(self, e):
        distances = list(map(lambda x: abs(x - e), self.sorted_list))
        if min(distances) <= self.tolerance:
            index = distances.index(min(distances))
        else:
            index = -1
        return index


def is_chinese(char):
    if char["text"] >= "\u4e00" and char["text"] <= "\u9fff":
        return True
    return False


def format_cell_text(text):
    return text.replace("\n", "")


def is_roman_num(text):
    roman_letter = ["I", "V", "X", "L", "X", "C", "D", "M"]
    for c in text:
        if c not in roman_letter:
            return False
    return True


def is_digit_num(text):
    if re.findall("^[0-9]+$", text.strip()) == 0:
        return False
    return True


def line_chars_to_text(line):
    return "".join([char["text"] for char in line if char != " "])


def line_chars_to_coordinate(page_num, line, page_height, page_width):
    return [
        [
            str(page_num),
            [
                round(float(char['x0']/page_width), 3),
                round(float(char['y0']/page_height), 3),
                round(float(char['x1']/page_width), 3),
                round(float(char['y1']/page_height), 3)
            ]
        ] for char in line if char != " "
    ]
