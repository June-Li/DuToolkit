import re

from MODELALG.utils.hypertxt_utils.chapter_settings import CHAPTER_CONFIG


def catalog2list(catalog_str):
    """从目录中截取章节字符串，并返回章节编号样式"""
    chapter_list = catalog_str.split("\n")
    catalog_symbol = max(catalog_str, key=catalog_str.count)  # 目录标示符
    chapter_list_clean = []

    # 从目录字符串中拿出，分割出章节的字符；考虑目录跨行问题
    i = 0
    while i < len(chapter_list):
        chapter_str = chapter_list[i].replace(" ", "")
        if chapter_str != "目录":  # 字符是目录则跳过
            index = chapter_str.find(catalog_symbol)
            if index == -1:  # 当前行未找到目录标示符，则连接下一行
                if (
                    len(chapter_str) < 8
                ):  # 判断目录行字符串长度，小于5则可能是页码，数字等，直接跳过
                    i += 1
                    continue
                i += 1
                chapter_str += (
                    chapter_list[i].replace(" ", "")
                    if len(chapter_list) > i
                    else chapter_str
                )
            index = chapter_str.find(catalog_symbol)
            if (
                index > -1
                and len(chapter_str) > index + 1
                and chapter_str[index + 1] == catalog_symbol
            ):
                chapter_list_clean.append(chapter_str[0:index])
        i += 1

    # 匹配章节的编码样式
    chapter_pattern_idxes = []
    for chapter_str in chapter_list_clean:
        is_matched = False
        for i, pattern in enumerate(CHAPTER_CONFIG["chapter_patterns"]):
            if re.match(pattern, chapter_str):
                chapter_pattern_idxes.append(i)
                is_matched = True
                break
        if not is_matched:
            chapter_pattern_idxes.append(-1)  # 默认中不存在该pattern

    return chapter_list_clean, chapter_pattern_idxes
