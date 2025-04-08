import collections
import sys

from . import utils
from operator import itemgetter
import itertools

DEFAULT_SNAP_TOLERANCE = 3
DEFAULT_JOIN_TOLERANCE = 3
DEFAULT_MIN_WORDS_VERTICAL = 3
DEFAULT_MIN_WORDS_HORIZONTAL = 1


def snap_edges(edges, tolerance=DEFAULT_SNAP_TOLERANCE):
    """
    Given a list of edges, snap any within `tolerance` pixels of one another
    to their positional average.
    """
    v, h = [list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")]

    snap = utils.snap_objects
    snapped = snap(v, "x0", tolerance) + snap(h, "top", tolerance)
    return snapped


def join_edge_group(edges, orientation, tolerance=DEFAULT_JOIN_TOLERANCE):
    """
    Given a list of edges along the same infinite line, join those that
    are within `tolerance` pixels of one another.
    """
    if orientation == "h":
        min_prop, max_prop = "x0", "x1"
    elif orientation == "v":
        min_prop, max_prop = "top", "bottom"
    else:
        raise ValueError("Orientation must be 'v' or 'h'")

    sorted_edges = list(sorted(edges, key=itemgetter(min_prop)))
    joined = [sorted_edges[0]]
    for e in sorted_edges[1:]:
        last = joined[-1]
        if e[min_prop] <= (last[max_prop] + tolerance):
            if e[max_prop] > last[max_prop]:
                # Extend current edge to new extremity
                joined[-1] = utils.resize_object(last, max_prop, e[max_prop])
        else:
            # Edge is separate from previous edges
            joined.append(e)

    return joined


def merge_edges(edges, snap_tolerance, join_tolerance):
    """
    Using the `snap_edges` and `join_edge_group` methods above,
    merge a list of edges into a more "seamless" list.
    """

    def get_group(edge):
        if edge["orientation"] == "h":
            return ("h", edge["top"])
        else:
            return ("v", edge["x0"])

    if snap_tolerance > 0:
        edges = snap_edges(edges, snap_tolerance)

    if join_tolerance > 0:
        _sorted = sorted(edges, key=get_group)
        edge_groups = itertools.groupby(_sorted, key=get_group)
        edge_gen = (
            join_edge_group(items, k[0], join_tolerance) for k, items in edge_groups
        )
        edges = list(itertools.chain(*edge_gen))
    return edges


def words_to_edges_h(words, word_threshold=DEFAULT_MIN_WORDS_HORIZONTAL):
    """
    Find (imaginary) horizontal lines that connect the tops
    of at least `word_threshold` words.
    """
    by_top = utils.cluster_objects(words, "top", 1)
    large_clusters = filter(lambda x: len(x) >= word_threshold, by_top)
    rects = list(map(utils.objects_to_rect, large_clusters))
    if len(rects) == 0:
        return []
    min_x0 = min(map(itemgetter("x0"), rects))
    max_x1 = max(map(itemgetter("x1"), rects))
    max_bottom = max(map(itemgetter("bottom"), rects))
    edges = [
                {
                    "x0": min_x0,
                    "x1": max_x1,
                    "top": r["top"],
                    "bottom": r["top"],
                    "width": max_x1 - min_x0,
                    "orientation": "h",
                }
                for r in rects
            ] + [
                {
                    "x0": min_x0,
                    "x1": max_x1,
                    "top": max_bottom,
                    "bottom": max_bottom,
                    "width": max_x1 - min_x0,
                    "orientation": "h",
                }
            ]

    return edges


def words_to_edges_v(words, word_threshold=DEFAULT_MIN_WORDS_VERTICAL):
    """
    Find (imaginary) vertical lines that connect the left, right, or
    center of at least `word_threshold` words.
    """
    # Find words that share the same left, right, or centerpoints
    by_x0 = utils.cluster_objects(words, "x0", 1)
    by_x1 = utils.cluster_objects(words, "x1", 1)
    by_center = utils.cluster_objects(words, lambda x: (x["x0"] + x["x1"]) / 2, 1)
    clusters = by_x0 + by_x1 + by_center

    # Find the points that align with the most words
    sorted_clusters = sorted(clusters, key=lambda x: -len(x))
    large_clusters = filter(lambda x: len(x) >= word_threshold, sorted_clusters)

    # For each of those points, find the bboxes fitting all matching words
    bboxes = list(map(utils.objects_to_bbox, large_clusters))

    # Iterate through those bboxes, condensing overlapping bboxes
    condensed_bboxes = []
    for bbox in bboxes:
        overlap = False
        for c in condensed_bboxes:
            if utils.get_bbox_overlap(bbox, c):
                overlap = True
                break
        if not overlap:
            condensed_bboxes.append(bbox)

    if len(condensed_bboxes) == 0:
        return []

    condensed_rects = map(utils.bbox_to_rect, condensed_bboxes)
    sorted_rects = list(sorted(condensed_rects, key=itemgetter("x0")))

    max_x1 = max(map(itemgetter("x1"), sorted_rects))
    min_top = min(map(itemgetter("top"), sorted_rects))
    max_bottom = max(map(itemgetter("bottom"), sorted_rects))

    # Describe all the left-hand edges of each text cluster
    edges = [
                {
                    "x0": b["x0"],
                    "x1": b["x0"],
                    "top": min_top,
                    "bottom": max_bottom,
                    "height": max_bottom - min_top,
                    "orientation": "v",
                }
                for b in sorted_rects
            ] + [
                {
                    "x0": max_x1,
                    "x1": max_x1,
                    "top": min_top,
                    "bottom": max_bottom,
                    "height": max_bottom - min_top,
                    "orientation": "v",
                }
            ]

    return edges


def edges_to_intersections(edges, x_tolerance=1, y_tolerance=1):
    """
    Given a list of edges, return the points at which they intersect
    within `tolerance` pixels.
    """
    intersections = {}
    v_edges, h_edges = [
        list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")
    ]
    for v in sorted(v_edges, key=itemgetter("x0", "top")):
        for h in sorted(h_edges, key=itemgetter("top", "x0")):
            if (
                    (v["top"] <= (h["top"] + y_tolerance))
                    and (v["bottom"] >= (h["top"] - y_tolerance))
                    and (v["x0"] >= (h["x0"] - x_tolerance))
                    and (v["x0"] <= (h["x1"] + x_tolerance))
            ):
                vertex = (v["x0"], h["top"])
                if vertex not in intersections:
                    intersections[vertex] = {"v": [], "h": []}
                intersections[vertex]["v"].append(v)
                intersections[vertex]["h"].append(h)
    return intersections


def intersections_to_cells(intersections):
    """
    Given a list of points (`intersections`), return all rectangular "cells"
    that those points describe.

    `intersections` should be a dictionary with (x0, top) tuples as keys,
    and a list of edge objects as values. The edge objects should correspond
    to the edges that touch the intersection.
    """

    def edge_connects(p1, p2):
        def edges_to_set(edges):
            return set(map(utils.obj_to_bbox, edges))

        if p1[0] == p2[0]:
            common = edges_to_set(intersections[p1]["v"]).intersection(
                edges_to_set(intersections[p2]["v"])
            )
            if len(common):
                return True

        if p1[1] == p2[1]:
            common = edges_to_set(intersections[p1]["h"]).intersection(
                edges_to_set(intersections[p2]["h"])
            )
            if len(common):
                return True
        return False

    points = list(sorted(intersections.keys()))
    n_points = len(points)

    def find_smallest_cell(points, i):
        if i == n_points - 1:
            return None
        pt = points[i]
        rest = points[i + 1:]
        # Get all the points directly below and directly right
        below = [x for x in rest if x[0] == pt[0]]
        right = [x for x in rest if x[1] == pt[1]]
        for below_pt in below:
            if not edge_connects(pt, below_pt):
                continue

            for right_pt in right:
                if not edge_connects(pt, right_pt):
                    continue

                bottom_right = (right_pt[0], below_pt[1])

                if (
                        (bottom_right in intersections)
                        and edge_connects(bottom_right, right_pt)
                        and edge_connects(bottom_right, below_pt)
                ):
                    return (pt[0], pt[1], bottom_right[0], bottom_right[1])

    cell_gen = (find_smallest_cell(points, i) for i in range(len(points)))
    return list(filter(None, cell_gen))


def cells_to_tables(cells):
    """
    Given a list of bounding boxes (`cells`), return a list of tables that
    hold those cells most simply (and contiguously).
    """

    def bbox_to_corners(bbox):
        x0, top, x1, bottom = bbox
        return list(itertools.product((x0, x1), (top, bottom)))

    cells = [
        {"available": True, "bbox": bbox, "corners": bbox_to_corners(bbox)}
        for bbox in cells
    ]

    # Iterate through the cells found above, and assign them
    # to contiguous tables

    def init_new_table():
        return {"corners": set([]), "cells": []}

    def assign_cell(cell, table):
        table["corners"] = table["corners"].union(set(cell["corners"]))
        table["cells"].append(cell["bbox"])
        cell["available"] = False

    n_cells = len(cells)
    n_assigned = 0
    tables = []
    current_table = init_new_table()
    while True:
        initial_cell_count = len(current_table["cells"])
        for i, cell in enumerate(filter(itemgetter("available"), cells)):
            if len(current_table["cells"]) == 0:
                assign_cell(cell, current_table)
                n_assigned += 1
            else:
                corner_count = sum(
                    c in current_table["corners"] for c in cell["corners"]
                )
                if corner_count > 0 and cell["available"]:
                    assign_cell(cell, current_table)
                    n_assigned += 1
        if n_assigned == n_cells:
            break
        if len(current_table["cells"]) == initial_cell_count:
            tables.append(current_table)
            current_table = init_new_table()

    if len(current_table["cells"]):
        tables.append(current_table)

    _sorted = sorted(tables, key=lambda t: min(t["corners"]))
    filtered = [t["cells"] for t in _sorted if len(t["cells"]) > 1]
    return filtered


class CellGroup(object):
    def __init__(self, cells):
        self.cells = cells
        self.bbox = (
            min(map(itemgetter(0), filter(None, cells))),
            min(map(itemgetter(1), filter(None, cells))),
            max(map(itemgetter(2), filter(None, cells))),
            max(map(itemgetter(3), filter(None, cells))),
        )


class Row(CellGroup):
    pass


class Table(object):
    def __init__(self, page, cells):
        self.page = page
        self.cells = cells
        self.bbox = (
            min(map(itemgetter(0), cells)),
            min(map(itemgetter(1), cells)),
            max(map(itemgetter(2), cells)),
            max(map(itemgetter(3), cells)),
        )

    @property
    def rows(self):
        _sorted = sorted(self.cells, key=itemgetter(1, 0))
        xs = list(sorted(set(map(itemgetter(0), self.cells))))
        rows = []
        for y, row_cells in itertools.groupby(_sorted, itemgetter(1)):
            xdict = dict((cell[0], cell) for cell in row_cells)
            row = Row([xdict.get(x) for x in xs])
            rows.append(row)
        return rows

    def extract(
            self,
            x_tolerance=utils.DEFAULT_X_TOLERANCE,
            y_tolerance=utils.DEFAULT_Y_TOLERANCE,
    ):

        chars = self.page.chars
        # 徐志昂2022-03-15添加如下一行，因为表格中出现了诸如(cid:13)这样的字符。
        chars = list(filter(lambda x: len(x['text']) < 3, chars))
        table_arr = []

        def char_in_bbox(char, bbox):
            v_mid = (char["top"] + char["bottom"]) / 2
            h_mid = (char["x0"] + char["x1"]) / 2
            x0, top, x1, bottom = bbox
            return (
                    (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
            )

        for row in self.rows:
            arr = []
            row_chars = [char for char in chars if char_in_bbox(char, row.bbox)]

            for cell in row.cells:
                if cell is None:
                    cell_text = None
                else:
                    cell_chars = [
                        char for char in row_chars if char_in_bbox(char, cell)
                    ]

                    if len(cell_chars):
                        cell_text = utils.extract_text(
                            cell_chars, x_tolerance=x_tolerance, y_tolerance=y_tolerance
                        ).strip()
                    else:
                        cell_text = ""
                arr.append(cell_text)
            table_arr.append(arr)

        return table_arr


TABLE_STRATEGIES = ["lines", "lines_strict", "text", "explicit"]
DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
    "join_tolerance": DEFAULT_JOIN_TOLERANCE,
    "edge_min_length": 3,
    "min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
    "min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
    "keep_blank_chars": False,
    "text_tolerance": 3.5,
    "text_x_tolerance": None,
    "text_y_tolerance": None,
    "intersection_tolerance": 3,
    "intersection_x_tolerance": None,
    "intersection_y_tolerance": None,
}


class TableFinder(object):
    """
    Given a PDF page, find plausible table structures.

    Largely borrowed from Anssi Nurminen's master's thesis:
    http://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

    ... and inspired by Tabula:
    https://github.com/tabulapdf/tabula-extractor/issues/16
    """

    def __init__(self, page, settings={}):
        for k in settings.keys():
            if k not in DEFAULT_TABLE_SETTINGS:
                raise ValueError(f"Unrecognized table setting: '{k}'")
        self.page = page
        self.settings = dict(DEFAULT_TABLE_SETTINGS)
        self.settings.update(settings)
        for var, fallback in [
            ("text_x_tolerance", "text_tolerance"),
            ("text_y_tolerance", "text_tolerance"),
            ("intersection_x_tolerance", "intersection_tolerance"),
            ("intersection_y_tolerance", "intersection_tolerance"),
        ]:
            if self.settings[var] is None:
                self.settings.update({var: self.settings[fallback]})

        self.edges = self.get_edges()

        if self.edges:
            self.add_virtual_edges2()
            self.edges = merge_edges(
                self.edges,
                snap_tolerance=settings["snap_tolerance"],
                join_tolerance=settings["join_tolerance"],
            )

        self.intersections = edges_to_intersections(
            self.edges,
            self.settings["intersection_x_tolerance"],
            self.settings["intersection_y_tolerance"],
        )

        self.cells = intersections_to_cells(self.intersections)
        self.tables = [Table(self.page, t) for t in cells_to_tables(self.cells)]

    def add_virtual_edges(self):
        x_tolerance = self.settings["text_x_tolerance"],
        y_tolerance = self.settings["text_y_tolerance"],
        edges = self.edges
        x_min = min(edge['x0'] for edge in edges)
        y_min = min(edge['y0'] for edge in edges)
        h_edges = [edge for edge in edges if edge['orientation'] == 'h']

        x_min_edges = [edge for edge in edges if abs(edge['x0'] - x_min) < x_tolerance[0]]
        if any(edge['orientation'] == 'h' for edge in x_min_edges) and not any(
                edge['orientation'] == 'v' for edge in x_min_edges):
            if len(x_min_edges) >= 2:
                y_min = min(edge['y0'] for edge in x_min_edges)
                y_max = max(edge['y1'] for edge in x_min_edges)
                self.edges.append(self.add_edges(x_min, y_min, x_min, y_max, self.edges[0]['page_number']))

        edges.sort(key=lambda x: x['x1'], reverse=True)
        cur_max = 0
        cur_num = 0
        find_x_max = False
        for edge in edges:
            if cur_num == 0:
                cur_max = edge['x1']
                cur_num = 1
            else:
                if cur_max - edge['x1'] <= x_tolerance[0]:
                    cur_num += 1
                    if cur_num == 3:
                        find_x_max = True
                        break
                else:
                    cur_max = edge['x1']
                    cur_num = 1
        if find_x_max:
            x_max_edges = [edge for edge in edges if abs(edge['x1'] - cur_max) < x_tolerance[0]]
            if any(edge['orientation'] == 'h' for edge in x_max_edges) and not any(
                    edge['orientation'] == 'v' for edge in x_max_edges):
                if len(x_max_edges) >= 2:
                    y_min = min(edge['y0'] for edge in x_max_edges)
                    y_max = max(edge['y1'] for edge in x_max_edges)
                    self.edges.append(self.add_edges(cur_max, y_min, cur_max, y_max, self.edges[0]['page_number']))

    def add_virtual_edges2(self):
        # 针对于有些表格缺少左右两条边的情况，增加左右两条边
        # 困难的地方就在于，此时程序中还没有表格的概念，如果条件比较宽松，就容易识别出不存在的表格。
        x_tolerance = self.settings["text_x_tolerance"],
        y_tolerance = self.settings["text_y_tolerance"],
        edges = self.edges
        # 计算出所有文本块，以及文本块的坐标
        words = self.page.extract_words(
            x_tolerance=self.settings["text_x_tolerance"],
            y_tolerance=self.settings["text_y_tolerance"],
            # keep_blank_chars=self.settings["keep_blank_chars"],
            keep_blank_chars=self.settings["keep_blank_chars"],
        )
        # 获取所有的横向边和纵向边
        h_edges = [edge for edge in edges if edge['orientation'] == 'h']
        v_edges = [edge for edge in edges if edge['orientation'] == 'v']

        lacked_left_edges = set()
        lacked_right_edges = set()  # x: (y_begin, y_end)

        edge_set = set()
        need_add_edge = collections.defaultdict(list)
        edge_infs = []
        added_tolerance = 1.5 * x_tolerance[0]

        # 对于每个每个文本块去寻找上下左右四个方向上距离最近的四条边
        for word in words:
            # 使用dete_edge找到包围word的上下左右方向的四条边的index，如果没有找到边，边的index == -1。
            (lIndex, rIndex, tIndex, bIndex), edge_count = self.detect_edge(word, h_edges, v_edges, x_tolerance,
                                                                            y_tolerance)
            # 通过边索引找到具体的边
            (left_edge, right_edge) = [v_edges[index] if index != -1 else None for index in [lIndex, rIndex]]
            (top_edge, bottom_edge) = [h_edges[index] if index != -1 else None for index in [tIndex, bIndex]]
            # 这里规则写的可以强一些，避免误伤
            if edge_count == 4 and top_edge['y0'] == top_edge['y1'] and bottom_edge['y0'] == bottom_edge['y1']:
                bad_edge = calculate_score(left_edge, right_edge, top_edge, bottom_edge, added_tolerance)
                if bad_edge != '':
                    if bad_edge == 'top':
                        # print(f"抛弃 {word['text']} 的上边线")
                        top_edge = None
                        tIndex = -1
                        edge_count -= 1
                    if bad_edge == 'bottom':
                        # print(f"抛弃 {word['text']} 的下边线")
                        bottom_edge = None
                        bIndex = -1
                        edge_count -= 1
                    if bad_edge == 'left':
                        # print(f"抛弃 {word['text']} 的左边线")
                        left_edge = None
                        lIndex = -1
                        edge_count -= 1
                    if bad_edge == 'right':
                        # print(f"抛弃 {word['text']} 的右边线")
                        right_edge = None
                        rIndex = -1
                        edge_count -= 1

                # if bottom_edge['x0'] - left_edge['x0'] > added_tolerance or top_edge['x0'] - left_edge['x0'] > added_tolerance:
                #     print(f"抛弃 {word['text']} 的左边线,{left_edge['y0']},{bottom_edge['y0']},{right_edge['y1']},{top_edge['y1']}")
                #     left_edge = None
                #     lIndex = -1
                #     edge_count -= 1
                # if right_edge['x1'] - bottom_edge['x1'] > added_tolerance or right_edge['x1'] - top_edge['x1'] > 1.5*added_tolerance:
                #     print(f"抛弃 {word['text']} 的右边线,{right_edge['y0']},{bottom_edge['y0']},{right_edge['y1']},{top_edge['y1']}")
                #     right_edge = None
                #     rIndex = -1
                #     edge_count -= 1
                #
                # if edge_count == 4 and (top_edge['y0'] - left_edge['y1'] > added_tolerance or top_edge['y0'] - right_edge['y1'] > added_tolerance):
                #     print(f"抛弃 {word['text']} 的上边线")
                #     top_edge = None
                #     tIndex = -1
                #     edge_count -= 1
                # # 如果左边的上下坐标相比上下边的坐标相差较大，那么左边直接抛弃，使用生成方案。
                # if left_edge['y0'] - bottom_edge['y0'] > y_tolerance[0] or top_edge['y0'] - left_edge['y1'] > y_tolerance[0]:
                #     print(f"抛弃 {word['text']} 的左边线,{left_edge['y0']},{bottom_edge['y0']},{right_edge['y1']},{top_edge['y1']}")
                #     left_edge = None
                #     lIndex = -1
                #     edge_count -= 1
                # if right_edge['y0'] - bottom_edge['y0'] > y_tolerance[0] or top_edge['y0'] - right_edge['y1'] > y_tolerance[0]:
                #     print(f"抛弃 {word['text']} 的右边线,{right_edge['y0']},{bottom_edge['y0']},{right_edge['y1']},{top_edge['y1']}")
                #     right_edge = None
                #     rIndex = -1
                #     edge_count -= 1

            if edge_count >= 2:
                if (lIndex, rIndex, tIndex, bIndex) in edge_set:
                    continue
                else:
                    edge_set.add((lIndex, rIndex, tIndex, bIndex))
                    # 88.224 507.220(609.34,624,94)
                    # 507.580,(566.710,593.500)
                # 补齐左右两条边
                # 如果现在缺少左边的那一条边，但是上下方向存在边，加入到need_add_edge中,
                # 如果两条边的X坐标差值 <= x_tolerance[0]，则认为两条边的X坐标相同

                if top_edge is not None and bottom_edge is not None and \
                        left_edge is None and abs(top_edge['x0'] - bottom_edge['x0']) <= added_tolerance:
                    # 如果只有上下两条边，并且间距较大，那么直接舍弃。
                    if edge_count == 2 and top_edge['y0'] - bottom_edge['y0'] >= 20:
                        continue
                    find_x0 = -1
                    for x0 in lacked_left_edges:
                        # if abs(x0 - top_edge['x0']) <= x_tolerance[0]:
                        if abs(x0 - top_edge['x0']) <= added_tolerance:
                            find_x0 = x0
                            break
                    if find_x0 == -1:
                        find_x0 = top_edge['x0']
                        lacked_left_edges.add(find_x0)
                    need_add_edge[find_x0].append((top_edge['y0'], bottom_edge['y0'], edge_count, len(edge_infs)))
                    edge_infs.append(
                        f"第{self.edges[0]['page_number']}页的{word['text']}增加了左侧一条线(x:{find_x0},{top_edge['y0']}:{bottom_edge['y0']})")

                # 如果现在缺少右边的那一条边
                if top_edge is not None and bottom_edge is not None and \
                        right_edge is None and abs(top_edge['x1'] - bottom_edge['x1']) <= added_tolerance:
                    # 如果只有上下两条边，并且间距较大，那么直接舍弃。
                    if edge_count == 2 and top_edge['y0'] - bottom_edge['y0'] >= 20:
                        continue
                    find_x1 = -1
                    for x1 in lacked_right_edges:
                        # if abs(x1 - top_edge['x1']) <= x_tolerance[0]:
                        if abs(x1 - top_edge['x1']) <= added_tolerance:
                            find_x1 = x1
                            break
                    if find_x1 == -1:
                        find_x1 = top_edge['x1']
                        lacked_right_edges.add(find_x1)
                    need_add_edge[find_x1].append((top_edge['y1'], bottom_edge['y1'], edge_count, len(edge_infs)))
                    edge_infs.append(
                        f"第{self.edges[0]['page_number']}页的{word['text']}增加了右侧一条线(x:{find_x1},{top_edge['y0']}:{bottom_edge['y0']})")

                if edge_count == 3 and top_edge is None and abs(left_edge['y1'] - right_edge['y1']) <= y_tolerance[0]:
                    self.edges.append(
                        self.add_edges(left_edge['x0'], left_edge['y1'], right_edge['x0'], left_edge['y1'],
                                       self.page.page_number))
                    # print(f'第{self.page.page_number}页 {word}:增加上面的一条边')

                if edge_count == 3 and bottom_edge is None and abs(left_edge['y0'] - right_edge['y0']) <= y_tolerance[
                    0]:
                    self.edges.append(
                        self.add_edges(left_edge['x0'], left_edge['y0'], right_edge['x0'], left_edge['y0'],
                                       self.page.page_number))
                    # print(f'第{self.page.page_number}页 {word}:增加下面的一条边')
        # print(edge_infs)
        # 聚合具有相同X坐标，连续Y坐标的边为更长的边
        # 例如X坐标都为1，Y坐标为（1, 3),（2.9, 5）,(5.1,7)
        # 聚合后的Y的坐标为(1, 7)
        for key, value in need_add_edge.items():
            value.sort(key=lambda x: x[0], reverse=True)
            d = []
            last_x = -1
            last_y = -1
            last_edge_count = -1
            logger_index = []
            for (x, y, edge_count, index) in value:
                if last_y == -1:
                    last_x = x
                    last_y = y
                    last_edge_count = edge_count
                    logger_index.append(index)
                else:
                    if abs(last_y - x) <= x_tolerance[0]:
                        last_y = y
                        last_edge_count = 3 if edge_count == 3 else last_edge_count
                        logger_index.append(index)
                    else:
                        if last_edge_count == 3:
                            d.append((last_x, last_y))
                            last_x = x
                            last_y = y
                            last_edge_count = edge_count
                            # for index in logger_index:
                            #     print(edge_infs[index])
                            logger_index = []
            if last_x != -1 and last_edge_count == 3:
                d.append((last_x, last_y))
                # for index in logger_index:
                #     print(edge_infs[index])
                logger_index = []

            need_add_edge[key] = d
        for key, value in need_add_edge.items():
            for y1, y0 in value:
                self.edges.append(self.add_edges(key, y0, key, y1, self.page.page_number))

    # 593->641
    # 609->624
    # 611->622
    def detect_edge(self, word, h_edges, v_edges, x_tolerance, y_tolerance):
        x_tolerance = x_tolerance[0]
        y_tolerance = y_tolerance[0]

        distance = {'left': sys.maxsize, 'right': sys.maxsize, 'top': sys.maxsize, 'bottom': sys.maxsize}
        res_index = {'left': -1, 'right': -1, 'top': -1, 'bottom': -1}
        word_bottom = utils.decimalize(self.page.layout.y1) - word['bottom']
        word_top = utils.decimalize(self.page.layout.y1) - word['top']
        for index, edge in enumerate(h_edges):
            if min(edge['x0'], edge['x1']) <= word['x0'] + x_tolerance and max(edge['x0'], edge['x1']) >= word['x1'] - x_tolerance:
                sub = word_bottom - max(edge['y0'], edge['y1'])
                if -x_tolerance < sub < distance['bottom']:
                    distance['bottom'] = sub
                    res_index['bottom'] = index
                else:
                    sub = word_top - min(edge['y0'], edge['y1'])
                    if sub < x_tolerance and abs(sub) < distance['top']:
                        distance['top'] = abs(sub)
                        res_index['top'] = index

        for index, edge in enumerate(v_edges):
            if min(edge['y0'], edge['y1']) <= word_bottom + y_tolerance and max(edge['y0'],
                                                                                edge['y1']) >= word_top - y_tolerance:
                sub = word['x0'] - max(edge['x0'], edge['x1'])
                if -y_tolerance < sub < distance['left']:
                    distance['left'] = sub
                    res_index['left'] = index
                else:
                    sub = word['x1'] - min(edge['x0'], edge['x1'])
                    if sub < y_tolerance and abs(sub) < distance['right']:
                        distance['right'] = abs(sub)
                        res_index['right'] = index

        directions = ['left', 'right', 'top', 'bottom']
        edge_count = 4
        for direction in directions:
            if res_index[direction] == -1:
                edge_count -= 1
        return [res_index[direction] for direction in directions], edge_count

    def add_edges(self, x0, y0, x1, y1, page_number):
        edge = {}
        edge.update({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "width": abs(x1 - x0),
            "height": abs(y1 - y0),
            "pts": [(x0, y0), (x1, y1)],
            "linewidth": utils.decimalize(0),
            "orientation": 'h' if y0 == y1 else 'v',
            "top": self.page.bbox[3] - max(y0, y1),
            "bottom": self.page.bbox[3] - min(y0, y1),
            'doctop': self.page.bbox[3] - max(y0, y1),
            "object_type": 'rect_edge',
            "page_number": page_number
        })
        return edge

    def get_edges(self):
        # v_base = self.page.edges
        settings = self.settings
        for name in ["vertical", "horizontal"]:
            strategy = settings[name + "_strategy"]
            if strategy not in TABLE_STRATEGIES:
                raise ValueError(
                    f'{name}_strategy must be one of {{{",".join(TABLE_STRATEGIES)}}}'
                )
            if strategy == "explicit":
                if len(settings["explicit_" + name + "_lines"]) < 2:
                    raise ValueError(
                        f"If {strategy}_strategy == 'explicit', explicit_{name}_lines "
                        f"must be specified as a list/tuple of two or more "
                        f"floats/ints."
                    )

        v_strat = settings["vertical_strategy"]
        h_strat = settings["horizontal_strategy"]

        if v_strat == "text" or h_strat == "text":
            words = self.page.extract_words(
                x_tolerance=settings["text_x_tolerance"],
                y_tolerance=settings["text_y_tolerance"],
                keep_blank_chars=settings["keep_blank_chars"],
            )

        v_explicit = []
        for desc in settings["explicit_vertical_lines"]:
            if isinstance(desc, dict):
                for e in utils.obj_to_edges(desc):
                    if e["orientation"] == "v":
                        v_explicit.append(e)
            else:
                v_explicit.append(
                    {
                        "x0": utils.decimalize(desc),
                        "x1": utils.decimalize(desc),
                        "top": self.page.bbox[1],
                        "bottom": self.page.bbox[3],
                        "height": self.page.bbox[3] - self.page.bbox[1],
                        "orientation": "v",
                    }
                )
        if v_strat == "lines":

            orientation = "v"
            edge_type = None
            min_length = 1

            if orientation not in ("v", "h", None):
                raise ValueError("Orientation must be 'v' or 'h'")

            def test(e):
                dim = "height" if e["orientation"] == "v" else "width"
                orient_correct = e["orientation"] == orientation
                return orient_correct and (e[dim] >= min_length)

            v_base = list(filter(test, self.page.edges))
            # v_base = utils.filter_edges(self.page.edges, "v")

        elif v_strat == "lines_strict":
            v_base = utils.filter_edges(self.page.edges, "v", edge_type="line")
        elif v_strat == "text":
            v_base = words_to_edges_v(
                words, word_threshold=settings["min_words_vertical"]
            )

        elif v_strat == "explicit":
            v_base = []

        v = v_base + v_explicit


        h_explicit = []
        for desc in settings["explicit_horizontal_lines"]:
            if isinstance(desc, dict):
                for e in utils.obj_to_edges(desc):
                    if e["orientation"] == "h":
                        h_explicit.append(e)
            else:
                h_explicit.append(
                    {
                        "x0": self.page.bbox[0],
                        "x1": self.page.bbox[2],
                        "width": self.page.bbox[2] - self.page.bbox[0],
                        "top": utils.decimalize(desc),
                        "bottom": utils.decimalize(desc),
                        "orientation": "h",
                    }
                )

        if h_strat == "lines":
            h_base = utils.filter_edges(self.page.edges, "h")
        elif h_strat == "lines_strict":
            h_base = utils.filter_edges(self.page.edges, "h", edge_type="line")
        elif h_strat == "text":
            h_base = words_to_edges_h(
                words, word_threshold=settings["min_words_horizontal"]
            )
        elif h_strat == "explicit":
            h_base = []

        h = h_base + h_explicit

        edges = list(v) + list(h)

        if len(edges) != 0 and 'non_stroking_color' in edges[0].keys():
            # non_stroking_color_set = set([edge['non_stroking_color'] for edge in edges])
            edges = list(filter(
                lambda x: x['non_stroking_color'] not in [0.651, 0.749, 0.827, 0.753, 0.843, 0.847, 0.857, 0.851, 0.745,
                                                          0.898, 0.949, 1], edges))
            non_stroking_color_set = set([edge['non_stroking_color'] if not isinstance(edge['non_stroking_color'],
                                                                                       list) else tuple(
                edge['non_stroking_color']) for edge in edges])
            # if (num - len(edges)) != 0:
            #     print(f"edges过滤掉了{num - len(edges)}个")
            if len(non_stroking_color_set) > 1 and None not in non_stroking_color_set:
                pass
                # print(f"在{self.page.page_number}页，需要关注的edges:{non_stroking_color_set}")
            # edges = list(filter(lambda x: x['non_stroking_color'] != 1, edges))
        # print(f'page:{self.page}, {set(edge["non_stroking_color"] for edge in edges)}')
        if settings["snap_tolerance"] > 0 or settings["join_tolerance"] > 0:
            # TODO:
            edges2 = merge_edges(
                edges,
                snap_tolerance=settings["snap_tolerance"],
                join_tolerance=settings["join_tolerance"],
            )
            edges = merge_edges(
                edges,
                # snap_tolerance=settings["snap_tolerance"],
                snap_tolerance=5,
                join_tolerance=settings["join_tolerance"],
            )
            if len(edges2) > len(edges):
                pass
                # print(f'第{self.page.page_number}页修改了merge_edges参数后，边的变化{len(edges2)}->{len(edges)}')

        return utils.filter_edges(edges, min_length=settings["edge_min_length"])


def calculate_score(left_edge, right_edge, top_edge, bottom_edge, added_tolerance):
    left_score, right_score, top_score, bottom_score = 0, 0, 0, 0
    if bottom_edge['x0'] - left_edge['x0'] > added_tolerance:  # bottom x0大了不好
        bottom_score -= 1
        left_score -= 1

    if left_edge['y0'] - bottom_edge['y0'] > added_tolerance:  # left y0 大了不好
        bottom_score -= 1
        left_score -= 1

    if right_edge['x1'] - bottom_edge['x1'] > added_tolerance:  # right
        bottom_score -= 1
        right_score -= 1

    if right_edge['y0'] - bottom_edge['y0'] > added_tolerance:
        bottom_score -= 1
        right_score -= 1

    if top_edge['y1'] - left_edge['y1'] > added_tolerance:  # top y1 大了不好
        top_score -= 1
        left_score -= 1

    if top_edge['x0'] - left_edge['x0'] > added_tolerance:  # top x0 大了不好
        top_score -= 1
        left_score -= 1

    if top_edge['y1'] - right_edge['y1'] > added_tolerance:  # top y1 大了不好
        top_score -= 1
        right_score -= 1

    if right_edge['x1'] - top_edge['x1'] > added_tolerance:  # right x1 大了不好
        top_score -= 1
        right_score -= 1

    sorted_scores = sorted([(top_score, 'top'), (bottom_score, 'bottom'), (left_score, 'left'), (right_score, 'right')],
                           key=lambda x: x[0])
    if sorted_scores[0][0] <= -2:
        # print(f'{sorted_scores[0][1]}这条边检测存在问题')
        return sorted_scores[0][1]
    else:
        return ''
