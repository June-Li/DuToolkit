# -*- coding: utf-8 -*-

############################################################
#
# Copyright (C) 2021 SenseDeal AI, Inc. All Rights Reserved
#
# Description:
#     测试代码
#
# Author: Li Xiuming
# Last Modified: 2022-03-15
############################################################
import copy
import json
import collections
import itertools
import regex as re


def cut_sentences(para, max_len=510):
    """
    输入一段话，输出分句结果，分句的依据
    1. 按照句号、感叹号、问号、中英文省略号结尾，添加标志符\n
    2. 如果分句之后，单行长度超过max_len，则在句子里面查询分号。
    2.1 如果有分号，则按照分号分句。
    2.2 如果没有分号，则按照max_len分句。
    :param para:
    :param max_len:
    :return:
    """
    para = re.sub(r'([。!！？\?])([^”’\[])', r"\1\n\2", para)  # 单字符断句符# 捕获句号、感叹号、问号，后面不是双引号或者单引号
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。!！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    res = []
    for line in para.split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        # 如果单行过长，用分号进一步分句
        if len(line) > max_len:
            for subline in re.sub('([:：；;])', r"\1\n", line).split("\n"):
                subline = subline.strip()
                if len(subline) <= max_len:
                    res.append(subline)
                # 用分号分句单行依然过长，则继续分句
                # 则先找到限制长度下最右边的逗号，将其与前面部分分出去，然后继续分后面部分
                # 如果找不到逗号，则将前限制长度的部分分出去，然后继续分后面部分
                else:
                    start = 0
                    while start < len(subline):
                        index = subline.rfind("，", start, start + max_len)
                        if index >= 0:
                            res.append(subline[start: index + 1])
                            start = index + 1
                        else:
                            res.append(subline[start: start + max_len])
                            start = start + max_len
        else:
            res.append(line)
    return [r for r in res if len(r) > 0], None


def find_cut_indices(para, max_len=510):
    # 定义用于分割句子的结束标点
    end_punctuations_single = {'。', '!', '！', '？', '?'}
    end_punctuations_multi = {'……', '......'}
    end_punctuation_followers = {'”', '’', '）', ')'}

    # 分割句子的起始点
    indices = [0]

    i = 0
    while i < len(para):
        next_char = para[min(i + 1, len(para) - 1)]  # 下一个字符

        # 根据单字符结束标点进行分句
        if para[i] in end_punctuations_single and next_char not in end_punctuation_followers:
            indices.append(i + 1)
        # 根据多字符结束标点进行分句
        elif i + 5 < len(para) and para[
                                   i:i + 6] in end_punctuations_multi and next_char not in end_punctuation_followers:
            indices.append(i + 6)

        i += 1

    return indices


def split_sentences(para, indices):
    """
    根据分割点切割句子

    :param para: 输入的段落
    :param indices: 分割点列表
    :return: 返回一个列表，包含了所有的句子
    """
    # 如果indices为空，返回整个段落作为一个句子
    if not indices:
        return [para]

    sentences = [para[i:j] for i, j in zip(indices, indices[1:] + [None])]
    return sentences


def split_list_by_indices(input_list, indices, final_indices):
    """
    根据给定的断句下标列表切割输入列表

    :param input_list: 输入的列表
    :param indices: 初始的断句下标列表
    :param final_indices: 进一步分割后的断句下标列表
    :return: 返回一个列表，包含了所有的子列表
    """
    sublists = []
    for i, (start, end) in enumerate(zip(indices, indices[1:] + [None])):
        sublist = input_list[start:end]
        final_indice = final_indices[i]
        for j, (s, e) in enumerate(zip(final_indice, final_indice[1:] + [None])):
            sublists.append(sublist[s:e])
    return sublists


# def split_long_sentence(sentence, max_len):
#     """
#     进一步切割过长的句子
#
#     :param sentence: 输入的句子
#     :param max_len: 句子的最大长度
#     :return: 返回一个列表，包含了所有的子句
#     """
#     split_punctuations = {':', '：', ';', '；'}
#     if len(sentence) <= max_len:  # 句子长度小于max_len，无需分割
#         return [sentence]
#
#     # 句子长度大于max_len，需要进行分割
#     new_indices = [0]
#     for i, char in enumerate(sentence):
#         if char in split_punctuations:  # 根据分隔符进行分句
#             new_indices.append(i + 1)
#     new_sentences = [sentence[i:j] for i, j in zip(new_indices, new_indices[1:] + [None])]
#
#     # 根据max_len进行最后的分句
#     final_sentences = []
#     for new_sentence in new_sentences:
#         while len(new_sentence) > max_len:
#             final_sentences.append(new_sentence[:max_len])
#             new_sentence = new_sentence[max_len:]
#         final_sentences.append(new_sentence)
#
#     return final_sentences

def split_long_sentence(sentence, max_len):
    split_punctuations = {':', '：', ';', '；'}
    indices = [0]

    # 如果句子长度大于max_len，进行分割
    if len(sentence) > max_len:
        for i, char in enumerate(sentence):
            if char in split_punctuations:  # 根据分隔符进行分句
                indices.append(i + 1)
        sentences = [sentence[i:j] for i, j in zip(indices, indices[1:] + [None])]

        # 根据max_len进行最后的分割，并记录分割下标
        final_sentences = []
        final_indices = []
        for idx, new_sentence in zip(indices, sentences):
            while len(new_sentence) > max_len:
                final_sentences.append(new_sentence[:max_len])
                final_indices.append(idx)
                new_sentence = new_sentence[max_len:]
                idx += max_len  # 更新当前子句的起始索引
            final_sentences.append(new_sentence)
            final_indices.append(idx)
    else:
        # 句子长度小于max_len，无需分割
        final_sentences = [sentence]
        final_indices = [0]

    return final_sentences, final_indices


def cut_sentences_0804(para, max_len=510):
    """
    综合以上函数进行分句

    :param para: 输入的段落
    :param max_len: 句子的最大长度
    :return: 返回分割点列表和句子列表
    """
    indices = find_cut_indices(para)  # 找到所有的分割点
    sentences = split_sentences(para, indices)  # 根据分割点切割句子
    final_sentences = []
    final_indices = []
    for sentence in sentences:
        final_sentence, final_indice = split_long_sentence(sentence, max_len)
        if len(final_sentence) > 0:
            final_sentences.append(final_sentence)  # [[text1, text2], ]
            final_indices.append(final_indice)
    return final_sentences, indices, final_indices  # 返回分割点列表和句子列表


class Text:
    def __init__(self, text, type, doc_start=None, doc_end=None, related_id=None, related_ori_idx=None, **kwargs):
        """
        Args:
            related_id: 不同场景下text对应的id
            related_ori_idx: 对应原始hypertxt的context下的索引
        """
        self.text = text
        self.type = type
        self.doc_start = doc_start
        self.doc_end = doc_end

        if related_id is not None and Hypertxt.granularity_related_dict[self.type]:
            self.__setattr__(Hypertxt.granularity_related_dict[self.type], related_id)

        # 存放原始的hypertxt对应的idx
        if related_ori_idx is not None:
            self.related_ori_idx = related_ori_idx

        for k, v in kwargs.items():
            self.__setattr__(k, v)


class Cell:
    def __init__(self, coord, cell_value):
        self.coord = tuple(coord)
        self.value = cell_value
        self.x0, self.y0, self.x1, self.y1 = coord

    @staticmethod
    def is_merge_cell(x0, y0, x1, y1):
        """判断、拆分合并单元格"""
        is_merge = False
        splits = []

        x_diff = x1 - x0
        y_diff = y1 - y0
        if x_diff != 1:
            is_merge = True
            i = 0
            while x0 + i + 1 <= x1:
                new_x0 = x0 + i
                new_x1 = x0 + i + 1

                if y_diff != 1:
                    j = 0
                    while y0 + j + 1 <= y1:
                        splits.append((new_x0, y0 + j, new_x1, y0 + j + 1))
                        j += 1
                else:
                    splits.append((new_x0, y0, new_x1, y1))
                i += 1
        else:
            if y_diff != 1:
                is_merge = True
                j = 0
                while y0 + j + 1 <= y1:
                    splits.append((x0, y0 + j, x1, y0 + j + 1))
                    j += 1

        return is_merge, splits


class Table:
    def __init__(self, table_cell_list):
        self.ori_cell_list = table_cell_list
        self.cells = {}
        self.merge_cells = collections.defaultdict(list)  # 存放拆分后的单元格伪坐标
        for coord, cell_value in table_cell_list:
            if isinstance(cell_value, dict):
                raise ValueError("暂不支持提取嵌套表格")
            else:
                self.cells[tuple(coord)] = Cell(coord, cell_value)

            is_merge, splits = self.cells[tuple(coord)].is_merge_cell(*coord)
            if is_merge:
                self.merge_cells[tuple(coord)].extend(splits)

        # 计算坐标极值
        x0s, y0s, x1s, y1s = zip(*self.cells.keys())
        self.max_x0, self.max_y0, self.max_x1, self.max_y1 = max(x0s), max(y0s), max(x1s), max(y1s)
        self.min_x0, self.min_y0, self.min_x1, self.min_y1 = min(x0s), min(y0s), min(x1s), min(y1s)

        # 存放伪坐标对应的原始坐标
        self.reversed_merge_cells = {}
        for coord, splits in self.merge_cells.items():
            for s in splits:
                self.reversed_merge_cells[s] = coord

        self.candidates = self.get_candidates()
        self.diaphragm_cells = self.get_diaphragm_cell()

    def __len__(self):
        return len(self.cells)

    def get_one_row(self, x0, split_merge=True):
        """获取某一行单元格"""
        row = []
        for coord, cell in self.cells.items():
            if split_merge and coord in self.merge_cells:
                for coord1 in self.merge_cells[coord]:
                    if coord1[0] == x0:
                        row.append((coord, cell))
            else:
                if coord[0] == x0:
                    row.append((coord, cell))

        row.sort(key=lambda x: x[0][1])  # 按y1排序
        return row

    def get_one_col(self, y0, split_merge=True):
        """获取某一列单元格"""
        col = []
        for coord, cell in self.cells.items():
            if split_merge and coord in self.merge_cells:
                for coord1 in self.merge_cells[coord]:
                    if coord1[1] == y0:
                        col.append((coord, cell))
            else:
                if coord[1] == y0:
                    col.append((coord, cell))

        col.sort(key=lambda x: x[0][0])  # 按x0排序
        return col

    def get_candidates(self):
        """预查所有单元格对应的向右、向下的单元格"""
        candidates = {k: {"right": [], "down": []} for k in self.cells}
        for coord in candidates:
            x0, y0, x1, y1 = coord
            for new_x0 in range(x1, self.max_x0 + 1):
                for new_x1 in range(new_x0 + 1, self.max_x1 + 1):
                    new_coord = (new_x0, y0, new_x1, y1)
                    # 判断构造的单元格是否存在原始单元格里
                    if new_coord in self.cells:
                        candidates[coord]["down"].append(new_coord)
                    # 如果是伪单元格，且原始左边界或右边界相等，则也是一样
                    elif new_coord in self.reversed_merge_cells:
                        if self.reversed_merge_cells[new_coord][1] == y0 or self.reversed_merge_cells[new_coord][3] == y1:
                            candidates[coord]["down"].append(self.reversed_merge_cells[new_coord])
                        else:
                            break

            for new_y0 in range(y1, self.max_y0 + 1):
                for new_y1 in range(new_y0 + 1, self.max_y1 + 1):
                    new_coord = (x0, new_y0, x1, new_y1)
                    if new_coord in self.cells:
                        candidates[coord]["right"].append(new_coord)
                    elif new_coord in self.reversed_merge_cells:
                        if self.reversed_merge_cells[new_coord][0] == x0 or self.reversed_merge_cells[new_coord][2] == x1:
                            candidates[coord]["right"].append(self.reversed_merge_cells[new_coord])
                        else:
                            break

        for coord in candidates:
            candidates[coord]["right"] = sorted(list(set(candidates[coord]["right"])))
            candidates[coord]["down"] = sorted(list(set(candidates[coord]["down"])))

            for i in range(1, len(candidates[coord]["right"])):
                # 不连续则截断
                if candidates[coord]["right"][i][1] != candidates[coord]["right"][i - 1][3]:
                    candidates[coord]["right"] = candidates[coord]["right"][:i]
                    break
            for i in range(1, len(candidates[coord]["down"])):
                if candidates[coord]["down"][i][0] != candidates[coord]["down"][i - 1][2]:
                    candidates[coord]["down"] = candidates[coord]["down"][:i]
                    break

        return candidates

    def get_diaphragm_cell(self):
        """获取横隔单元格"""
        diaphragm_cells = [cell.coord for cell in self.cells.values() if
                           cell.y0 == self.min_y0 and cell.y1 == self.max_y1]
        return diaphragm_cells

    def get_cross_coords(self, coord1, coord2):
        """获取交叉单元格"""

        # 默认1为左
        if coord1[1] > coord2[1]:
            coord1, coord2 = coord2, coord1

        # 完全同一行或同一列的排除
        if coord1[0] == coord2[0] or coord1[2] == coord2[2]:
            return []
        if coord1[1] == coord2[1] or coord1[3] == coord2[3]:
            return []

        # 候选交叉单元格
        if coord1[0] > coord2[0]:
            x0, y0, x1, y1 = coord1[0], coord2[1], coord1[2], coord2[3]
        else:
            x0, y0, x1, y1 = coord2[0], coord1[1], coord2[2], coord1[3]
        is_merge, splits = Cell.is_merge_cell(x0, y0, x1, y1)
        if not is_merge:
            splits.extend([(x0, y0, x1, y1)])

        # 判断纵向是否有横隔单元格
        cross_coords = []
        for coord in splits:
            x0, y0, x1, y1 = coord
            flag = False
            if coord[0] > coord2[0]:
                i = x0
                while i > coord2[0]:
                    coord_last = (i, y0, i + 1, y1)
                    if coord_last in self.diaphragm_cells or (coord_last in self.reversed_merge_cells \
                                                              and self.reversed_merge_cells[
                                                                  coord_last] in self.diaphragm_cells):
                        flag = True
                        break
                    i -= 1
            elif coord[0] < coord2[0]:
                i = x0
                while i < coord2[0]:
                    coord_last = (i, y0, i + 1, y1)
                    if coord_last in self.diaphragm_cells or (coord_last in self.reversed_merge_cells \
                                                              and self.reversed_merge_cells[
                                                                  coord_last] in self.diaphragm_cells):
                        flag = True
                        break
                    i += 1
            if not flag:
                cross_coords.append(coord)

        # 筛选候选交叉单元格
        coord_checked = []
        for coord in cross_coords:
            if coord in self.cells:
                coord_checked.append(coord)
            elif coord in self.reversed_merge_cells:
                coord_checked.append(self.reversed_merge_cells[coord])
        cross_coords = list(set(coord_checked))

        return cross_coords


class Hypertxt:
    granularity_related_dict = {
        "title": "title_id",
        "sentence": "sid",
        "paragraph": "pid",
        "segment": "seg_id",
        "generated_segment": "gen_seg_id",
        "table": "table_id",
        "document": "doc_id"
    }

    def __init__(self, ori_hypertxt=None, id=None):
        self.id = id
        if ori_hypertxt is not None:
            self.ori_hypertxt = ori_hypertxt
            # Text(text=Table(text.text), type="table", related_id=table_id, related_ori_idx=[idx])
            self.texts = [Text(**r) for r in self.ori_hypertxt["context"]]

            self.titles = collections.OrderedDict()
            self.sentences = collections.OrderedDict()
            self.paragraphs = collections.OrderedDict()
            self.segments = collections.OrderedDict()
            self.generated_segments = collections.OrderedDict()
            self.tables = collections.OrderedDict()
            self.documents = collections.OrderedDict()

            self.transform()

            self.nested_tables = None
            if 'nested_tables' in self.ori_hypertxt["metadata"].keys():
                self.nested_tables = {key: Text(text=Table(table['text']),
                                                type="table",
                                                table_id=int(key.split('|')[0]),
                                                related_ori_idx=self.tables[int(key.split('|')[0])].related_ori_idx)
                                      for key, table in self.ori_hypertxt["metadata"]['nested_tables'].items()}

    def transform(self):
        paragraph_idxs = []
        segment_idxs = []
        document_idxs = []
        cur_pid = 0
        for idx, text in enumerate(self.texts):
            # 标题
            if text.type == "title":
                self.titles[0] = Text(text=text.text, type="title", related_id=0, related_ori_idx=[idx])
                continue

            # 表格
            elif text.type == "table":
                table_id = len(self.tables)
                self.tables[table_id] = Text(text=Table(text.text), type="table", related_id=table_id,
                                             related_ori_idx=[idx])

                # 组织表格前片段
                if len(segment_idxs) > 0:
                    seg_id = len(self.segments)
                    seg_text = []
                    related_ori_idx = []
                    for idxs in segment_idxs:
                        related_ori_idx.extend(idxs)
                        seg_text.append("".join([self.texts[idx].text for idx in idxs]))
                    seg_text = "\n".join(seg_text)
                    self.segments[seg_id] = Text(text=seg_text, type="segment", related_id=seg_id,
                                                 related_ori_idx=related_ori_idx)

                segment_idxs = []

            # 段落
            elif text.type == "text":
                self.sentences[text.sid] = Text(text=text.text, type="sentence", related_id=text.sid,
                                                related_ori_idx=[idx])
                if text.pid != cur_pid:
                    segment_idxs.append([idx])
                else:
                    segment_idxs[-1].append(idx)

                if text.pid != cur_pid:
                    document_idxs.append([idx])
                else:
                    document_idxs[-1].append(idx)

                if text.pid != cur_pid and len(paragraph_idxs) > 0:
                    # 整合上一段
                    paragraph_text = "".join([self.texts[idx].text for idx in paragraph_idxs])
                    self.paragraphs[cur_pid] = Text(text=paragraph_text, type="paragraph", related_id=cur_pid,
                                                    related_ori_idx=paragraph_idxs)
                    cur_pid = text.pid
                    paragraph_idxs = [idx]
                else:
                    cur_pid = text.pid
                    paragraph_idxs.append(idx)

        # 组织最后的片段
        if len(paragraph_idxs) > 0:
            paragraph_text = "".join([self.texts[idx].text for idx in paragraph_idxs])
            self.paragraphs[cur_pid] = Text(text=paragraph_text, type="paragraph", related_id=cur_pid,
                                            related_ori_idx=paragraph_idxs)

        if len(segment_idxs) > 0:
            seg_id = len(self.segments)
            seg_text = []
            related_ori_idx = []
            for idxs in segment_idxs:
                related_ori_idx.extend(idxs)
                seg_text.append("".join([self.texts[idx].text for idx in idxs]))
            seg_text = "\n".join(seg_text)
            self.segments[seg_id] = Text(text=seg_text, type="segment", related_id=seg_id,
                                         related_ori_idx=related_ori_idx)

        if len(document_idxs) > 0:
            doc_text = []
            related_ori_idx = []
            for idxs in document_idxs:
                related_ori_idx.extend(idxs)
                doc_text.append("".join([self.texts[idx].text for idx in idxs]))
            doc_text = "\n".join(doc_text)
            self.documents[0] = Text(text=doc_text, type="document", related_id=0, related_ori_idx=related_ori_idx)

    @staticmethod
    def generate(txt=None, title=None, filepath=None, output_filepath=None, cut_fn=cut_sentences,
                 line2cid=None, cid2chapter=None, meta_info=None, line2section=None):
        if txt is None and filepath is not None:
            with open(filepath, "r") as f:
                txt = f.read()

        hypertxt = {"metadata": meta_info if meta_info is not None else {},
                    "chapters": cid2chapter if cid2chapter is not None else {},
                    "context": []}
        if title is not None:
            hypertxt["context"].append({"text": title, "type": title, "sid": 0, "pid": 0})

        if txt is None or txt == "":
            hypertxt["context"] = []
        else:
            lines = txt.split("\n")
            context = []
            table = []
            sid = 1
            pid = 1
            for j, line in enumerate(lines):
                cid = line2cid[j] if line2cid is not None else ""
                section_range = line2section[j] if line2section is not None else []
                line = line.strip()
                # 跳过空行
                if len(line) == 0:
                    continue

                # 找到表格，表格自成一段
                if line == "<table>":
                    table = [line]
                elif line == "</table>":
                    table.append(line)
                    table_cell_list = []
                    for cell in table[1:-1]:
                        coord_values = cell.split("|")
                        coord = coord_values[0]
                        cell_value = "|".join(coord_values[1:])

                        coord = coord.split(",")
                        coord = map(float, coord)
                        x0, y0, x1, y1 = map(int, coord)

                        table_cell_list.append(((x0, y0, x1, y1), cell_value))

                    context.append({"text": table_cell_list, "type": "table", "sid": sid, "pid": pid,
                                    "cid": cid, "metadata": {"section_range": section_range}})
                    pid += 1
                    sid += 1
                    table = []
                else:
                    if table:
                        table.append(line)
                    else:
                        sents, _ = cut_fn(line)
                        context.extend([{"text": sent, "type": "text", "sid": sid + i, "pid": pid, "cid": cid,
                                         "metadata": {"section_range": section_range}} for i, sent in enumerate(sents)])
                        sid += len(sents)
                        pid += 1
            hypertxt["context"].extend(context)

        if output_filepath is not None:
            with open(output_filepath, "w") as f:
                json.dump(hypertxt, f, ensure_ascii=False, indent=4)

        return hypertxt

    @staticmethod
    def add_title_to_ori(ori_hypertxt, title, force=False):

        title_idx = None
        for idx, context in enumerate(ori_hypertxt["context"]):
            if context["sid"] == 0:
                title_idx = idx
                break

        if len(ori_hypertxt["context"]) > 0:
            if title_idx is None:
                ori_hypertxt["context"] = [{"text": title, "type": "title", "sid": 0, "pid": 0}] + ori_hypertxt[
                    "context"]
            elif title_idx is not None or force:
                ori_hypertxt["context"][title_idx]["text"] = title
        else:
            ori_hypertxt["context"].append({"text": title, "type": "title", "sid": 0, "pid": 0})

        return ori_hypertxt

    def get_text(self, key_type, key):
        if key_type == "title_id":
            return self.titles[key]
        elif key_type == "sid":
            return self.sentences[key]
        elif key_type == "pid":
            return self.paragraphs[key]
        elif key_type == "table_id":
            return self.tables[key]
        elif key_type == "seg_id":
            return self.segments[key]
        elif key_type == "gen_seg_id":
            return self.generated_segments[key]
        elif key_type == "doc_id":
            return self.documents[key]
        else:
            return None

    def get_sids(self, key_type, key, text=None):
        if text is None:
            text = self.get_text(key_type, key)
            if text is None:
                return None

        sids = []
        for ori_idx in text.related_ori_idx:
            sids.append(self.texts[ori_idx].sid)

        if len(sids) > 0:
            return sids
        else:
            return None

    def find_gen_seg(self, text, related_ori_idx=None):
        assert isinstance(text, str)
        for k, v in self.generated_segments.items():
            if v.text == text:
                return k

        gen_seg_id = len(self.generated_segments)
        self.generated_segments[gen_seg_id] = Text(text=text, type="generated_segment", related_id=gen_seg_id,
                                                   related_ori_idx=related_ori_idx)
        return gen_seg_id

    def filter(self, granularity="sentences", with_gen_seg=False):
        if granularity == "title":
            return list(self.titles.values())
        elif granularity == "sentence":
            return list(self.sentences.values())
        elif granularity == "table":
            return list(self.tables.values())
        elif granularity == "paragraph":
            return list(self.paragraphs.values())
        elif granularity == "segment":
            if with_gen_seg:
                return itertools.chain(list(self.segments.values()), list(self.generated_segments.values()))
            else:
                return list(self.segments.values())
        elif granularity == "document":
            return list(self.documents.values())

    def get_scope(self, scope, exclude_scope=None):
        """
        根据id范围获取对应的记录。

        Args:
             scope: {"sid": [], "pid": []....}
        """

        exclude_sids, exclude_pids, exclude_seg_ids, \
        exclude_gen_seg_ids, exclude_table_ids, exclude_doc_ids, exclude_title_ids = None, None, None, None, None, None, None
        if exclude_scope is not None:
            exclude_sids = exclude_scope.get("sid", None)
            exclude_pids = exclude_scope.get("pid", None)
            exclude_seg_ids = exclude_scope.get("seg_id", None)
            exclude_gen_seg_ids = exclude_scope.get("gen_seg_id", None)
            exclude_table_ids = exclude_scope.get("table_id", None)
            exclude_doc_ids = exclude_scope.get("doc_id", None)
            exclude_title_ids = exclude_scope.get("title_id", None)

        records = []
        sids = scope.get("sid", None)
        if sids:
            for sid in sids:
                if exclude_sids is not None and sid in exclude_sids:
                    continue
                if sid not in self.sentences:
                    continue
                records.append(self.sentences[sid])

        pids = scope.get("pid", None)
        if pids:
            for pid in pids:
                if exclude_pids is not None and pid in exclude_pids:
                    continue
                if pid not in self.paragraphs:
                    continue
                records.append(self.paragraphs[pid])

        seg_ids = scope.get("seg_id", None)
        if seg_ids:
            for seg_id in seg_ids:
                if exclude_seg_ids is not None and seg_id in exclude_seg_ids:
                    continue
                if seg_id not in self.segments:
                    continue
                records.append(self.segments[seg_id])

        gen_seg_ids = scope.get("gen_seg_id", None)
        if gen_seg_ids:
            for gen_seg_id in gen_seg_ids:
                if exclude_gen_seg_ids is not None and gen_seg_id in exclude_gen_seg_ids:
                    continue
                if gen_seg_id not in self.generated_segments:
                    continue
                records.append(self.generated_segments[gen_seg_id])

        table_ids = scope.get("table_id", None)
        if table_ids:
            for table_id in table_ids:
                if exclude_table_ids is not None and table_id in exclude_table_ids:
                    continue
                if table_id not in self.tables:
                    continue
                records.append(self.tables[table_id])

        doc_ids = scope.get("doc_id", None)
        if doc_ids:
            for doc_id in doc_ids:
                if exclude_doc_ids is not None and doc_id in exclude_doc_ids:
                    continue
                if doc_id not in self.documents:
                    continue
                records.append(self.documents[doc_id])

        title_ids = scope.get("title_id", None)
        if title_ids:
            for title_id in title_ids:
                if exclude_title_ids is not None and title_id in exclude_title_ids:
                    continue
                if title_id not in self.titles:
                    continue
                records.append(self.titles[title_id])

        return records

    def get_scopes(self, scopes, exclude_scopes=None):
        if len(scopes) == 0:
            return []

        merge_scope = copy.deepcopy(scopes[0])
        for scope in scopes[1:]:
            for key in scope:
                if key not in merge_scope:
                    merge_scope[key] = []
                merge_scope[key].extend(scope[key])

        for key in merge_scope:
            merge_scope[key] = sorted(list(set(merge_scope[key])))

        merge_exclude_scope = None
        if exclude_scopes is not None:
            merge_exclude_scope = copy.deepcopy(exclude_scopes[0])
            for scope in exclude_scopes[1:]:
                for key in scope:
                    merge_exclude_scope[key].append(scope[key])

            for key in merge_exclude_scope:
                merge_exclude_scope[key] = sorted(list(set(merge_exclude_scope[key])))

        records = self.get_scope(merge_scope, merge_exclude_scope)
        return records
