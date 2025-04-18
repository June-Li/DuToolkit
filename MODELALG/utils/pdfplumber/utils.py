from decimal import Decimal, ROUND_HALF_UP
import numbers
from operator import itemgetter
import itertools
from functools import lru_cache as cache
from pdfminer.utils import PDFDocEncoding
from pdfminer.psparser import PSLiteral
from pdfminer.pdftypes import PDFObjRef


DEFAULT_X_TOLERANCE = 3
DEFAULT_Y_TOLERANCE = 3


def cluster_list(xs, tolerance=0):
    tolerance = decimalize(tolerance)
    if tolerance == Decimal(0):
        return [[x] for x in sorted(xs)]
    if len(xs) < 2:
        return [[x] for x in sorted(xs)]
    groups = []
    xs = list(sorted(xs))
    current_group = [xs[0]]
    last = xs[0]
    for x in xs[1:]:
        if x <= (last + tolerance):
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]
        last = x
    groups.append(current_group)
    return groups


def make_cluster_dict(values, tolerance):
    tolerance = decimalize(tolerance)
    clusters = cluster_list(set(values), tolerance)

    nested_tuples = [
        [(val, i) for val in value_cluster] for i, value_cluster in enumerate(clusters)
    ]

    cluster_dict = dict(itertools.chain(*nested_tuples))
    return cluster_dict


def cluster_objects(objs, attr, tolerance):
    if isinstance(attr, (str, int)):
        attr_getter = itemgetter(attr)
    else:
        attr_getter = attr
    objs = to_list(objs)
    values = map(attr_getter, objs)
    cluster_dict = make_cluster_dict(values, tolerance)

    get_0, get_1 = itemgetter(0), itemgetter(1)

    cluster_tuples = sorted(
        ((obj, cluster_dict.get(attr_getter(obj))) for obj in objs), key=get_1
    )

    grouped = itertools.groupby(cluster_tuples, key=get_1)

    clusters = [list(map(get_0, v)) for k, v in grouped]

    return clusters


def decode_text(s):
    """
    Decodes a PDFDocEncoding string to Unicode.
    Adds py3 compatibility to pdfminer's version.
    """
    if type(s) == bytes and s.startswith(b"\xfe\xff"):
        return str(s[2:], "utf-16be", "ignore")
    else:
        ords = (ord(c) if type(c) == str else c for c in s)
        return "".join(PDFDocEncoding[o] for o in ords)


def decode_psl_list(_list):
    return [
        decode_text(value.name) if isinstance(value, PSLiteral) else value
        for value in _list
    ]


def resolve(x):
    if type(x) == PDFObjRef:
        return x.resolve()
    else:
        return x


def get_dict_type(d):
    if type(d) is not dict:
        return None
    t = d.get("Type")
    if type(t) is PSLiteral:
        return decode_text(t.name)
    else:
        return t


def resolve_all(x):
    """
    Recursively resolves the given object and all the internals.
    """
    t = type(x)
    if t == PDFObjRef:
        resolved = x.resolve()

        # Avoid infinite recursion
        if get_dict_type(resolved) == "Page":
            return x

        return resolve_all(resolved)
    elif t in (list, tuple):
        return t(resolve_all(v) for v in x)
    elif t == dict:
        if get_dict_type(x) == "Annot":
            exceptions = ["Parent"]
        else:
            exceptions = []
        return dict((k, v if k in exceptions else resolve_all(v)) for k, v in x.items())
    else:
        return x


@cache(maxsize=int(10e4))
def _decimalize(v, q=None):
    # Convert int-like
    if isinstance(v, numbers.Integral):
        return Decimal(int(v))

    # Convert float-like
    elif isinstance(v, numbers.Real):
        if q is not None:
            return Decimal(repr(v)).quantize(Decimal(repr(q)), rounding=ROUND_HALF_UP)
        else:
            return Decimal(repr(v))
    else:
        raise ValueError(f"Cannot convert {v} to Decimal.")


def decimalize(v, q=None):
    # If already a decimal, just return itself
    if type(v) == Decimal:
        return v

    # If tuple/list passed, bulk-convert
    if isinstance(v, (tuple, list)):
        return type(v)(decimalize(x, q) for x in v)
    else:
        return _decimalize(v, q)


def is_dataframe(collection):
    cls = collection.__class__
    name = ".".join([cls.__module__, cls.__name__])
    return name == "pandas.core.frame.DataFrame"


def to_list(collection):
    if is_dataframe(collection):
        return collection.to_dict("records")  # pragma: nocover
    else:
        return list(collection)


def dedupe_chars(chars, tolerance=1):
    """
    Removes duplicate chars — those sharing the same text, fontname, size,
    and positioning (within `tolerance`) as other characters in the set.
    """
    key = itemgetter("fontname", "size", "upright", "text")
    pos_key = itemgetter("doctop", "x0")
    t = decimalize(tolerance)

    def yield_unique_chars(chars):
        sorted_chars = sorted(chars, key=key)
        for grp, grp_chars in itertools.groupby(sorted_chars, key=key):
            for y_cluster in cluster_objects(grp_chars, "doctop", t):
                for x_cluster in cluster_objects(y_cluster, "x0", t):
                    yield sorted(x_cluster, key=pos_key)[0]

    deduped = yield_unique_chars(chars)
    return sorted(deduped, key=chars.index)


def collate_line(line_chars, tolerance=DEFAULT_X_TOLERANCE):
    tolerance = decimalize(tolerance)
    coll = ""
    last_x1 = None
    for char in sorted(line_chars, key=itemgetter("x0")):
        if (last_x1 is not None) and (char["x0"] > (last_x1 + tolerance)):
            coll += " "
        last_x1 = char["x1"]
        coll += char["text"]
    return coll


def objects_to_rect(objects):
    return {
        "x0": min(map(itemgetter("x0"), objects)),
        "x1": max(map(itemgetter("x1"), objects)),
        "top": min(map(itemgetter("top"), objects)),
        "bottom": max(map(itemgetter("bottom"), objects)),
    }


def objects_to_bbox(objects):
    return (
        min(map(itemgetter("x0"), objects)),
        min(map(itemgetter("top"), objects)),
        max(map(itemgetter("x1"), objects)),
        max(map(itemgetter("bottom"), objects)),
    )


obj_to_bbox = itemgetter("x0", "top", "x1", "bottom")


def bbox_to_rect(bbox):
    return {"x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}


DEFAULT_WORD_EXTRACTION_SETTINGS = dict(
    x_tolerance=DEFAULT_X_TOLERANCE,
    y_tolerance=DEFAULT_Y_TOLERANCE,
    keep_blank_chars=False,
    use_text_flow=False,
    horizontal_ltr=True,  # Should words be read left-to-right?
    vertical_ttb=True,  # Should vertical words be read top-to-bottom?
    extra_attrs=[],
)


class WordExtractor:
    def __init__(self, **settings):
        for s, val in settings.items():
            if s not in DEFAULT_WORD_EXTRACTION_SETTINGS:
                raise ValueError(f"{s} is not a valid WordExtractor parameter")

            if s in {"x_tolerance", "y_tolerance"}:
                val = decimalize(val)

            setattr(self, s, val)

    def merge_chars(self, ordered_chars):
        x0, top, x1, bottom = objects_to_bbox(ordered_chars)
        upright = ordered_chars[0]["upright"]

        direction = 1 if (self.horizontal_ltr if upright else self.vertical_ttb) else -1

        word = {
            "text": "".join(map(itemgetter("text"), ordered_chars)),
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bottom,
            "upright": upright,
            "direction": direction,
        }

        for key in self.extra_attrs:
            word[key] = ordered_chars[0][key]

        return word

    def char_begins_new_word(self, current_chars, next_char):
        upright = current_chars[0]["upright"]
        intraline_tol = self.x_tolerance if upright else self.y_tolerance
        interline_tol = self.y_tolerance if upright else self.x_tolerance

        word_x0, word_top, word_x1, word_bottom = objects_to_bbox(current_chars)

        return (
            (next_char["x0"] > word_x1 + intraline_tol)
            or (next_char["x1"] < word_x0 - intraline_tol)
            or (next_char["top"] > word_bottom + interline_tol)
            or (next_char["bottom"] < word_top - interline_tol)
        )

    def iter_chars_to_words(self, chars):
        current_word = []

        for chr in chars:
            if not self.keep_blank_chars and chr["text"].isspace():
                if current_word:
                    yield current_word
                    current_word = []

            elif current_word and self.char_begins_new_word(current_word, chr):
                yield current_word
                current_word = [chr]

            else:
                current_word.append(chr)

        if current_word:
            yield current_word

    def iter_sort_chars(self, chars):
        def upright_key(x):
            return -int(x["upright"])

        for upright_cluster in cluster_objects(chars, upright_key, 0):
            upright = upright_cluster[0]["upright"]
            cluster_key = "doctop" if upright else "x0"

            # Cluster by line
            subclusters = cluster_objects(
                upright_cluster, cluster_key, self.y_tolerance
            )

            for sc in subclusters:
                # Sort within line
                sort_key = "x0" if upright else "doctop"
                sc = sorted(sc, key=itemgetter(sort_key))

                # Reverse order if necessary
                if not (self.horizontal_ltr if upright else self.vertical_ttb):
                    sc = reversed(sc)

                yield from sc

    def iter_extract(self, chars):
        if not self.use_text_flow:
            chars = self.iter_sort_chars(chars)

        grouping_key = itemgetter("upright", *self.extra_attrs)
        grouped = itertools.groupby(chars, grouping_key)

        for keyvals, char_group in grouped:
            for word_chars in self.iter_chars_to_words(char_group):
                yield self.merge_chars(word_chars)

    def extract(self, chars):
        return list(self.iter_extract(chars))


def extract_words(chars, **kwargs):
    settings = dict(DEFAULT_WORD_EXTRACTION_SETTINGS)
    settings.update(kwargs)
    return WordExtractor(**settings).extract(chars)


def extract_text(
    chars, x_tolerance=DEFAULT_X_TOLERANCE, y_tolerance=DEFAULT_Y_TOLERANCE
):

    if len(chars) == 0:
        return None

    chars = to_list(chars)
    y_tolerance = 3.5## 20221123 放宽对Y的限制，原先的y_tolerance为3.5
    doctop_clusters = cluster_objects(chars, "doctop", y_tolerance)
    lines = (collate_line(line_chars, x_tolerance) for line_chars in doctop_clusters)

    coll = "\n".join(lines)
    return coll


collate_chars = extract_text


def filter_objects(objs, fn):
    if isinstance(objs, dict):
        return dict((k, filter_objects(v, fn)) for k, v in objs.items())

    initial_type = type(objs)
    objs = to_list(objs)
    filtered = filter(fn, objs)

    return initial_type(filtered)


def get_bbox_overlap(a, b):
    a_left, a_top, a_right, a_bottom = decimalize(a)
    b_left, b_top, b_right, b_bottom = decimalize(b)
    o_left = max(a_left, b_left)
    o_right = min(a_right, b_right)
    o_bottom = min(a_bottom, b_bottom)
    o_top = max(a_top, b_top)
    o_width = o_right - o_left
    o_height = o_bottom - o_top
    if o_height >= 0 and o_width >= 0 and o_height + o_width > 0:
        return (o_left, o_top, o_right, o_bottom)
    else:
        return None


def calculate_area(bbox):
    left, top, right, bottom = bbox
    if left > right or top > bottom:
        raise ValueError(f"{bbox} has a negative width or height.")
    return (right - left) * (bottom - top)


def clip_obj(obj, bbox):
    bbox = decimalize(bbox)

    overlap = get_bbox_overlap(obj_to_bbox(obj), bbox)
    if overlap is None:
        return None

    dims = bbox_to_rect(overlap)
    copy = dict(obj)

    for attr in ["x0", "top", "x1", "bottom"]:
        copy[attr] = dims[attr]

    if dims["top"] != obj["bottom"] or dims["top"] != obj["bottom"]:
        diff = dims["top"] - obj["top"]
        copy["doctop"] = obj["doctop"] + diff

    copy["width"] = copy["x1"] - copy["x0"]
    copy["height"] = copy["bottom"] - copy["top"]

    return copy


def intersects_bbox(objs, bbox):
    """
    Filters objs to only those intersecting the bbox
    """
    initial_type = type(objs)
    objs = to_list(objs)
    matching = [
        obj for obj in objs if get_bbox_overlap(obj_to_bbox(obj), bbox) is not None
    ]
    return initial_type(matching)


def within_bbox(objs, bbox):
    """
    Filters objs to only those fully within the bbox
    """
    if isinstance(objs, dict):
        return dict((k, within_bbox(v, bbox)) for k, v in objs.items())

    initial_type = type(objs)
    objs = to_list(objs)
    matching = [
        obj
        for obj in objs
        if get_bbox_overlap(obj_to_bbox(obj), bbox) == obj_to_bbox(obj)
    ]
    return initial_type(matching)


def crop_to_bbox(objs, bbox):
    """
    Filters objs to only those intersecting the bbox,
    and crops the extent of the objects to the bbox.
    """
    if isinstance(objs, dict):
        return dict((k, crop_to_bbox(v, bbox)) for k, v in objs.items())

    initial_type = type(objs)
    objs = to_list(objs)
    cropped = list(filter(None, (clip_obj(obj, bbox) for obj in objs)))
    return initial_type(cropped)


def move_object(obj, axis, value):
    assert axis in ("h", "v")
    if axis == "h":
        new_items = (
            ("x0", obj["x0"] + value),
            ("x1", obj["x1"] + value),
        )
    if axis == "v":
        new_items = [
            ("top", obj["top"] + value),
            ("bottom", obj["bottom"] + value),
        ]
        if "doctop" in obj:
            new_items += [("doctop", obj["doctop"] + value)]
        if "y0" in obj:
            new_items += [
                ("y0", obj["y0"] - value),
                ("y1", obj["y1"] - value),
            ]
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


def snap_objects(objs, attr, tolerance):
    axis = {"x0": "h", "x1": "h", "top": "v", "bottom": "v"}[attr]
    clusters = cluster_objects(objs, attr, tolerance)
    avgs = [sum(map(itemgetter(attr), objs)) / len(objs) for objs in clusters]
    snapped_clusters = [
        [move_object(obj, axis, avg - obj[attr]) for obj in cluster]
        for cluster, avg in zip(clusters, avgs)
    ]
    return list(itertools.chain(*snapped_clusters))


def resize_object(obj, key, value):
    assert key in {"x0", "x1", "top", "bottom"}
    old_value = obj[key]
    diff = value - old_value
    new_items = [
        (key, value),
    ]
    if key == "x0":
        assert value <= obj["x1"]
        new_items.append(("width", obj["x1"] - value))
    elif key == "x1":
        assert value >= obj["x0"]
        new_items.append(("width", value - obj["x0"]))
    elif key == "top":
        assert value <= obj["bottom"]
        new_items.append(("doctop", obj["doctop"] + diff))
        new_items.append(("height", obj["height"] - diff))
        if "y1" in obj:
            new_items.append(("y1", obj["y1"] - diff))
    elif key == "bottom":
        assert value >= obj["top"]
        new_items.append(("height", obj["height"] + diff))
        if "y0" in obj:
            new_items.append(("y0", obj["y0"] - diff))
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


def curve_to_edges(curve):
    point_pairs = zip(curve["points"], curve["points"][1:])
    return [
        {
            "x0": min(p0[0], p1[0]),
            "x1": max(p0[0], p1[0]),
            "top": min(p0[1], p1[1]),
            "doctop": min(p0[1], p1[1]) + (curve["doctop"] - curve["top"]),
            "bottom": max(p0[1], p1[1]),
            "width": abs(p0[0] - p1[0]),
            "height": abs(p0[1] - p1[1]),
            "orientation": "v" if p0[0] == p1[0] else ("h" if p0[1] == p1[1] else None),
        }
        for p0, p1 in point_pairs
    ]


def rect_to_edges(rect):
    # lixm于2020: 过滤掉一些边
    # xuzhiang于2022:过滤掉一些边（<=0.6）
    if rect['width'] >= 4 and rect['height'] >= 4:
        return []
    # elif rect['height'] <= 0.601:
    #     top = dict(rect)
    #     top.update(
    #         {
    #             "object_type": "rect_edge",
    #             "height": decimalize(0),
    #             "y0": rect["y1"],
    #             "bottom": rect["top"],
    #             "orientation": "h",
    #         }
    #     )
    #     return [top]
    # elif rect['width'] <= 0.601:
    #     left = dict(rect)
    #     left.update(
    #         {
    #             "object_type": "rect_edge",
    #             "width": decimalize(0),
    #             "x1": rect["x0"],
    #             "orientation": "v",
    #         }
    #     )
    #     return [left]
    top, bottom, left, right = [dict(rect) for x in range(4)]
    top.update(
        {
            "object_type": "rect_edge",
            "height": decimalize(0),
            "y0": rect["y1"],
            "bottom": rect["top"],
            "orientation": "h",
        }
    )
    bottom.update(
        {
            "object_type": "rect_edge",
            "height": decimalize(0),
            "y1": rect["y0"],
            "top": rect["top"] + rect["height"],
            "doctop": rect["doctop"] + rect["height"],
            "orientation": "h",
        }
    )
    left.update(
        {
            "object_type": "rect_edge",
            "width": decimalize(0),
            "x1": rect["x0"],
            "orientation": "v",
        }
    )
    right.update(
        {
            "object_type": "rect_edge",
            "width": decimalize(0),
            "x0": rect["x1"],
            "orientation": "v",
        }
    )
    return [top, bottom, left, right]


def line_to_edge(line):
    edge = dict(line)
    edge["orientation"] = "h" if (line["top"] == line["bottom"]) else "v"
    return edge


def obj_to_edges(obj):
    return {
        "line": lambda x: [line_to_edge(x)],
        "rect": rect_to_edges,
        "rect_edge": rect_to_edges,
        "curve": curve_to_edges,
    }[obj["object_type"]](obj)


def filter_edges(edges, orientation=None, edge_type=None, min_length=1):

    if orientation not in ("v", "h", None):
        raise ValueError("Orientation must be 'v' or 'h'")

    def test(e):
        dim = "height" if e["orientation"] == "v" else "width"
        et_correct = e["object_type"] == edge_type if edge_type is not None else True
        orient_correct = orientation is None or e["orientation"] == orientation
        return et_correct and orient_correct and (e[dim] >= min_length)

    edges = filter(test, edges)

    return list(edges)
