import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../")
sys.path.append(os.path.abspath(root_dir))

import base64
import copy
import datetime
import gc
import json

# import logging
import math
import re
import subprocess
import time
import traceback
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from math import cos, fabs, radians, sin

import cv2
import fitz
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode

from WORKFLOW.OTHER.llm_api.v0 import llm_processor


class Log:
    def __init__(self, logger_name=None, output_path="/Logs"):
        # filename = os.path.join(
        #     output_path, "log.txt"
        # )  # '- + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '
        # # 创建logger
        # self.logger = logging.getLogger(logger)
        # self.logger.setLevel(logging.DEBUG)

        # # 定义输出格式
        # format = logging.Formatter(
        #     fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        #     datefmt="%m/%d/%Y %H:%M:%S",
        # )

        # # 创建输出到控制台handler sh
        # sh = logging.StreamHandler(stream=sys.stdout)
        # sh.setLevel(logging.INFO)
        # sh.setFormatter(format)

        # # 创建写入文件handler fh
        # # fh = logging.FileHandler(filename=filename, encoding='utf-8', mode='w')
        # fh = RotatingFileHandler(
        #     filename=filename,
        #     maxBytes=100 * 1024 * 1024,
        #     backupCount=1,
        #     encoding="utf-8",
        # )
        # fh.setLevel(logging.INFO)
        # fh.setFormatter(format)

        # # 给logger添加两个handler
        # self.logger.addHandler(sh)
        # self.logger.addHandler(fh)

        # 移除默认的 sink
        logger.remove()

        # 自定义格式化函数
        pattern = re.compile(r"<([^>]*)>")

        def format_record(record):

            def check_and_replace_angle_brackets(string):
                return re.sub(pattern, r"[\1]", string)

            file_name = check_and_replace_angle_brackets(record["file"].name)
            function_name = check_and_replace_angle_brackets(record["function"])
            message = (
                check_and_replace_angle_brackets(record["message"])
                .replace("{", "{{")
                .replace("}", "}}")
            )

            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <level>{level}</level> - "
                + f"<cyan>{file_name}</cyan>:<cyan>{function_name}</cyan>:<cyan>{record['line']}</cyan> - <level>{message}</level>"
            )
            if record["extra"]:
                log_format += f" | {json.dumps(record['extra'])}".replace(
                    "{", "{{"
                ).replace("}", "}}")
            return log_format + "\n"

        # 添加控制台输出
        logger.add(
            sys.stdout,
            format=format_record,
            level="INFO",
            enqueue=True,
        )

        # 添加文件输出
        log_file = os.path.join(output_path, "log.txt")
        logger.add(
            log_file,
            rotation="100 MB",  # 当文件达到100MB时轮转
            retention=1,  # 保留1个备份
            encoding="utf-8",
            format=format_record,
            level="INFO",
            enqueue=True,
        )

        # self.logger = logger.bind(name=logger_name if logger_name else "default")
        self.logger = logger

    def get_logger(self):
        return self.logger


class GPTOP:
    def __int__(self):
        pass

    def make_prompt(self, blocks_box, blocks_text, prompt_file_path):
        prompt = open(os.path.join(root_dir, prompt_file_path), "r").read()

        if not len(blocks_box):
            return [], [], []
        sort_idx = np.argsort(np.array(blocks_box)[:, 1])
        blocks_box = list(np.array(blocks_box)[sort_idx])
        blocks_text = list(np.array(blocks_text)[sort_idx])

        prompts = []
        v_total_box, v_total_text = [], []
        temp_blocks_box = copy.deepcopy(blocks_box)
        temp_blocks_text = copy.deepcopy(blocks_text)
        while len(temp_blocks_box):
            v_box = [[int(i) for i in temp_blocks_box[0]]]
            v_text = [temp_blocks_text[0]]
            remove_idx = [0]
            for idx, box in enumerate(temp_blocks_box[1:]):
                idx += 1
                if abs(int(box[1]) - v_box[-1][5]) < 1 * abs(int(box[1]) - int(box[7])):
                    if (
                        cal_line_iou(
                            [int(box[0]), int(box[2])], [v_box[-1][0], v_box[-1][2]], 0
                        )
                        > 0.3
                        or cal_line_iou(
                            [int(box[0]), int(box[2])], [v_box[-1][0], v_box[-1][2]], 1
                        )
                        > 0.3
                    ):
                        v_box.append([int(i) for i in box])
                        v_text.append(temp_blocks_text[idx])
                        remove_idx.append(idx)
            if len(v_text) > 1:
                prompts.append(prompt + '"' + '"、"'.join(v_text) + '"')
                v_total_box.append(v_box)
                v_total_text.append(v_text)
            retain_idx = list(set(range(len(temp_blocks_box))) - set(remove_idx))
            temp_blocks_box = list(np.array(temp_blocks_box)[retain_idx])
            temp_blocks_text = list(np.array(temp_blocks_text)[retain_idx])
        return prompts, v_total_box, v_total_text

    def use_gpt_result_merge_box_v1(self, out_info):
        response = out_info["response"]
        v_box = out_info["v_box"]
        # v_text = out_info['v_text']
        try:
            if "[" in response and "]" in response:
                gpt_out_list = eval(
                    "[" + response.split("[", 1)[-1][::-1].split("]", 1)[-1][::-1] + "]"
                )
            else:
                return out_info
        except:
            return out_info

        v_box_update = []
        v_text_update = []
        box_idx = 0
        for e in gpt_out_list:
            if isinstance(e, list):
                v_box_update.append(
                    [
                        int(min(np.array(v_box)[box_idx : box_idx + len(e), 0])),
                        int(min(np.array(v_box)[box_idx : box_idx + len(e), 1])),
                        int(max(np.array(v_box)[box_idx : box_idx + len(e), 2])),
                        int(min(np.array(v_box)[box_idx : box_idx + len(e), 1])),
                        int(max(np.array(v_box)[box_idx : box_idx + len(e), 2])),
                        int(max(np.array(v_box)[box_idx : box_idx + len(e), 5])),
                        int(min(np.array(v_box)[box_idx : box_idx + len(e), 0])),
                        int(max(np.array(v_box)[box_idx : box_idx + len(e), 5])),
                    ]
                )
                v_text_update.append("".join(e))
                box_idx += len(e)
            else:
                v_box_update.append(v_box[box_idx])
                v_text_update.append(e)
                box_idx += 1
        out_info["v_box_update"] = v_box_update
        out_info["v_text_update"] = v_text_update
        return out_info

    def use_gpt_result_merge_box_v2(self, out_info):
        response = out_info["response"]
        v_box = out_info["v_box"]
        v_text = out_info["v_text"]
        try:
            if "[" in response and "]" in response:
                gpt_out_list = eval(
                    "[" + response.split("[", 1)[-1][::-1].split("]", 1)[-1][::-1] + "]"
                )
            else:
                return out_info
        except:
            return out_info

        v_box_update = []
        v_text_update = []
        if len("".join(gpt_out_list)) == len("".join(v_text)):
            # merge_list = [0]
            # for elem in gpt_out_list:
            #     merge_list.append(len(elem))
            merge_list = []
            box_idx = 0
            for elem in gpt_out_list:
                if len(v_text[box_idx]) == len(elem):
                    merge_list.append([box_idx])
                    box_idx += 1
                else:
                    temp_str = v_text[box_idx]
                    count_0 = 1
                    for idx_1, t in enumerate(v_text[box_idx + 1 :]):
                        temp_str += t
                        count_0 += 1
                        if len(temp_str) == len(elem):
                            break
                    merge_list.append(list(range(box_idx, box_idx + count_0)))
                    box_idx += count_0

            for idx_0, e in enumerate(merge_list):
                v_box_update.append(
                    [
                        int(min(np.array(v_box)[e[0] : e[-1] + 1, 0])),
                        int(min(np.array(v_box)[e[0] : e[-1] + 1, 1])),
                        int(max(np.array(v_box)[e[0] : e[-1] + 1, 2])),
                        int(min(np.array(v_box)[e[0] : e[-1] + 1, 1])),
                        int(max(np.array(v_box)[e[0] : e[-1] + 1, 2])),
                        int(max(np.array(v_box)[e[0] : e[-1] + 1, 5])),
                        int(min(np.array(v_box)[e[0] : e[-1] + 1, 0])),
                        int(max(np.array(v_box)[e[0] : e[-1] + 1, 5])),
                    ]
                )
                # v_text_update.append(gpt_out_list[idx_0])
        else:
            v_box_update = out_info["v_box"]
            gpt_out_list = out_info["v_text"]
        out_info["v_box_update"] = v_box_update
        out_info["v_text_update"] = gpt_out_list
        return out_info

    def get_merge_box_and_text(self, total_blocks_box, total_blocks_text, out_info):
        merge_total_blocks_box, merge_total_blocks_text = [], []
        for block_out in out_info["gpt_result"]:
            if "v_box_update" in block_out.keys():
                merge_total_blocks_box += block_out["v_box_update"]
                merge_total_blocks_text += block_out["v_text_update"]
        if len(merge_total_blocks_box) == 0:
            return total_blocks_box, total_blocks_text
        remove_idx = []
        for idx, box in enumerate(total_blocks_box):
            for box_m in merge_total_blocks_box:
                if (
                    cal_iou(
                        [box[0], box[1], box[4], box[5]],
                        [box_m[0], box_m[1], box_m[4], box_m[5]],
                        0,
                    )[0]
                    > 0.6
                ):
                    remove_idx.append(idx)
                    break
        retain_idx = list(set(range(len(total_blocks_box))) - set(remove_idx))
        merge_total_blocks_box += list(np.array(total_blocks_box)[retain_idx])
        merge_total_blocks_text += list(np.array(total_blocks_text)[retain_idx])
        for idx, box in enumerate(merge_total_blocks_box):
            merge_total_blocks_box[idx] = [int(i) for i in box]
        return merge_total_blocks_box, merge_total_blocks_text


class UploadFileToOss:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, oss_address):
        if not UploadFileToOss._is_initialized:
            if not os.path.exists(root_dir + "/upload/"):
                os.makedirs(root_dir + "/upload/")
            self.command_prefix = (
                "curl -X 'POST' "
                # + "'https://cubeflow.sensedeal.vip/api/files/upload'"
                + oss_address
                + f" -H 'X-Upload-Token: {os.environ.get('UPLOAD_TOKEN', '')}'"
                + " -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'files=@"
                + os.path.abspath(root_dir)
                + "/upload/"
            )
            self.command_suffix = ";type=image/jpg'"
            UploadFileToOss._is_initialized = True

    def upload_image(self, img):
        img_name = str(time.time()) + ".jpg"
        cv2.imwrite(os.path.abspath(root_dir) + "/upload/" + img_name, img)

        response_code, response_message, response_down_url = None, None, None
        for i in range(3):
            try:
                response = subprocess.run(
                    self.command_prefix + img_name + self.command_suffix,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                response_json = json.loads(response.stdout)
                response_code, response_message, response_down_url = (
                    response_json["code"],
                    response_json["message"],
                    response_json["items"][0]["down_url"],
                )
                os.remove(os.path.abspath(root_dir) + "/upload/" + img_name)
                return response_code, response_message, response_down_url
            except:
                pass
        os.remove(os.path.abspath(root_dir) + "/upload/" + img_name)
        return response_code, response_message, response_down_url


def clean_memory(device="cuda"):
    starttime = time.time()
    if device == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        import torch_npu

        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    gc.collect()
    logger.info(f"清理内存耗时：{time.time() - starttime}s")


def select_device(device="cuda:0"):
    # device = 'cpu' or 'cuda:0' or 'mlu:0' or 'npu:0' 不支持多卡
    if device.lower() == "cpu":
        return torch.device("cpu")
    else:
        assert bool(
            re.match(r"^(cuda|mlu|npu)([:：]\d+)?$", device, re.IGNORECASE)
        ), f"设备参数错误，请仔细检查>>>>>>"
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"  # check availability
        return torch.device(device)


def dict_deep_update(d1, d2):
    """递归更新字典d1，用d2中的键值对。"""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            dict_deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1


def handle_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)  # 或者使用 str(obj) 也可以
    raise TypeError


def convert_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Type not serializable")


def check_gpt(configs):
    if configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["mul_line_merge"]:
        try:
            llm_processor.get_answer(
                question=datetime.now().strftime("%Y%m%d-%H%M%S"),
                question_type="text",
                task_description="你是ChatGPT, 一个由OpenAI训练的大型语言模型, 你旨在回答并解决人们的任何问题，并且可以使用多种语言与人交流。",
                model=configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["model"],
                url=configs["API_BASE"],
                max_retries=configs["PROCESSES_CONTROL"]["TableModule"]["GPT"][
                    "max_retries"
                ],
            )
            print("chatgpt当前可以调用……")
        except:
            print("chatgpt当前不可调用……")
            configs["PROCESSES_CONTROL"]["TableModule"]["GPT"]["mul_line_merge"] = False


def draw_ocr_box_txt(
    image,
    boxes,
    txts,
    scores=None,
    drop_score=0.5,
    font_path="/volume/config/STHeiti_Light.ttc",
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new("RGB", (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        draw_box = [
            box[0][0],
            box[0][1],
            box[1][0],
            box[1][1],
            box[2][0],
            box[2][1],
            box[3][0],
            box[3][1],
        ]
        draw_left.polygon(draw_box, fill=color)
        draw_right.polygon(draw_box, outline=color)
        box_height = math.sqrt(
            (box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2
        )
        box_width = math.sqrt(
            (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2
        )
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.5), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                # char_size = font.getsize(c)
                chr_box = font.getbbox(c)
                char_size = (chr_box[2] - chr_box[0], chr_box[3] - chr_box[1])
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.5), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def draw_boxes(
    img, boxes, color=(0, 0, 255), thickness=2, out_dir="/workspace/JuneLi/debug.jpg"
):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
    if out_dir:
        cv2.imwrite(out_dir, img)
    return img


def draw_poly_boxes(
    img,
    boxes,
    isclosed=True,
    color=(0, 0, 255),
    thickness=2,
    out_dir="/workspace/JuneLi/debug.jpg",
):
    cv2.polylines(img, np.array(boxes), isclosed, color, thickness)
    if out_dir:
        cv2.imwrite(out_dir, img)
    return img


def sort_box(text_recs):
    if len(text_recs) < 1:
        return np.array([])
    text_recs = text_recs[text_recs[:, 1].argsort()]
    text_recs__ = []
    temp_list = [text_recs[0]]
    for ii in range(1, len(text_recs)):
        if (
            abs(text_recs[ii][1] - text_recs[ii - 1][1])
            / abs(text_recs[ii - 1][1] - text_recs[ii - 1][7])
            < 0.4
        ):
            temp_list.append(text_recs[ii])
        else:
            temp_list = np.array(temp_list)
            temp_list = list(temp_list[temp_list[:, 0].argsort()])
            text_recs__.extend(temp_list)
            temp_list = [text_recs[ii]]
    if len(temp_list) > 0:
        temp_list = np.array(temp_list)
        temp_list = list(temp_list[temp_list[:, 0].argsort()])
        text_recs__.extend(temp_list)
    text_recs = np.array(text_recs__)
    return text_recs


def encode_image(img):
    """
    将array类型的图片编码为base64
    Args:
        img: numpy array 类型的图片
    Returns:
        str: base64编码的字符串
    """
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def pdf2img(pdf_path, dpi, max_len):
    try:
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = np.shape(img)
            if max(h, w) > max_len:
                if h > w:
                    img = cv2.resize(img, (max_len * w // h, max_len))
                else:
                    img = cv2.resize(img, (max_len, max_len * h // w))
            yield img
    except:
        logger.error(" ···-> PDF文件读取失败！")
        logger.error(traceback.format_exc())


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    rot_flag = False
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.ascontiguousarray(np.rot90(dst_img), dtype=np.uint8)
        rot_flag = True
    return dst_img, rot_flag


def rotate_point(point, center, angle, height):
    """
    点point(x0, y0)绕点center(x1, y1)旋转angle后的点
    ======================================
    在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
    x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    ======================================
    将图像坐标(x,y)转换到平面坐标(x`,y`)：
    x`=x
    y`=height-y
    :param point:
    :param center: base point (基点)
    :param angle: 旋转角度，正：表示逆时针，负：表示顺时针
    :param height: 用于平面坐标和图像坐标的相互转换
    :return:
    """
    # 将图像坐标转换到平面坐标
    x0, y0 = point
    x1, y1 = center
    y0 = height - y0
    y1 = height - y1
    x = (
        (x0 - x1) * np.cos(np.pi / 180.0 * angle)
        - (y0 - y1) * np.sin(np.pi / 180.0 * angle)
        + x1
    )
    y = (
        (x0 - x1) * np.sin(np.pi / 180.0 * angle)
        + (y0 - y1) * np.cos(np.pi / 180.0 * angle)
        + y1
    )
    # 将平面坐标转换到图像坐标
    y = height - y
    return x, y


def rotate_box(box, center, angle, height):
    p0 = rotate_point((box[0], box[1]), center, angle, height)
    p1 = rotate_point((box[2], box[3]), center, angle, height)
    p2 = rotate_point((box[4], box[5]), center, angle, height)
    p3 = rotate_point((box[6], box[7]), center, angle, height)
    return np.reshape(np.array([p0, p1, p2, p3]), 8).tolist()


def rotate_boxes(boxes, center, angle, height, width):
    height_new = int(
        width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle)))
    )
    width_new = int(
        height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle)))
    )
    offset_w = (width_new - width) / 2
    offset_h = (height_new - height) / 2
    r_boxes = []
    for box in boxes:
        r_box = np.array(rotate_box(box, center, angle, height))
        r_box[[0, 2, 4, 6]] += offset_w
        r_box[[1, 3, 5, 7]] += offset_h
        r_boxes.append(r_box.tolist())
    return np.array(r_boxes, dtype=int).tolist()


def warp_affine_img(img, degree=45):
    """
    Desciption:
            Get img rotated a certain degree,
        and use some color to fill 4 corners of the new img.
    """

    # 获取旋转后4角的填充色
    try:
        filled_color = mode([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]).mode[0]
    except:
        filled_color = mode([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]).mode
    filled_color = (int(filled_color), int(filled_color), int(filled_color))

    height, width = img.shape[:2]

    # 旋转后的尺寸
    height_new = int(
        width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree)))
    )
    width_new = int(
        height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree)))
    )

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2

    # Pay attention to the type of elements of filler_color, which should be
    # the int in pure python, instead of those in numpy.
    img_rotated = cv2.warpAffine(
        img, mat_rotation, (width_new, height_new), borderValue=filled_color
    )
    # 填充四个角
    mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
    mask[:] = 0
    seed_points = [
        (0, 0),
        (0, height_new - 1),
        (width_new - 1, 0),
        (width_new - 1, height_new - 1),
    ]
    for i in seed_points:
        cv2.floodFill(img_rotated, mask, i, filled_color)

    return img_rotated


def cal_iou(box_0, box_1, cal_type=-1):
    iou, overlap_area, flag = 0, 0, False
    min_x = min(box_0[0], box_0[2], box_1[0], box_1[2])
    max_x = max(box_0[0], box_0[2], box_1[0], box_1[2])
    min_y = min(box_0[1], box_0[3], box_1[1], box_1[3])
    max_y = max(box_0[1], box_0[3], box_1[1], box_1[3])
    box_0_w = abs(box_0[0] - box_0[2])
    box_0_h = abs(box_0[1] - box_0[3])
    box_1_w = abs(box_1[0] - box_1[2])
    box_1_h = abs(box_1[1] - box_1[3])
    merge_w = max_x - min_x
    merge_h = max_y - min_y
    overlap_w = box_0_w + box_1_w - merge_w
    overlap_h = box_0_h + box_1_h - merge_h
    if overlap_h > 0 and overlap_w > 0:
        box_0_area = box_0_w * box_0_h
        box_1_area = box_1_w * box_1_h
        overlap_area = overlap_w * overlap_h
        if cal_type == 0:
            iou = overlap_area / box_0_area
        elif cal_type == 1:
            iou = overlap_area / box_1_area
        else:
            iou = overlap_area / (box_0_area + box_1_area - overlap_area)
        if overlap_w > 10 or overlap_h > 10:
            flag = True
    return iou, flag


def cal_iou_parallel(boxes_0, boxes_1, cal_type=-1):
    """
    NxM， boxes_0中每个框和boxes_1中每个框的IoU值；
    :param boxes_0: [[x_0, y_0, x_1, y_1], ……]，左上右右下角，N个
    :param boxes_1: [[x_0, y_0, x_1, y_1], ……]，左上右右下角，M个
    :param cal_type:
    :return:
    """
    if len(boxes_0) == 0 or len(boxes_1) == 0:
        return np.zeros((len(boxes_0), len(boxes_1)), dtype=np.float32)
    boxes_0 = torch.tensor(
        np.array(boxes_0),
        dtype=torch.float32,
        device="cpu",  # "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    boxes_1 = torch.tensor(
        np.array(boxes_1),
        dtype=torch.float32,
        device="cpu",  # "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    area_0 = (boxes_0[:, 2] - boxes_0[:, 0]) * (
        boxes_0[:, 3] - boxes_0[:, 1]
    )  # 每个框的面积 (N,)
    area_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (
        boxes_1[:, 3] - boxes_1[:, 1]
    )  # 每个框的面积 (M,)

    lt = torch.max(boxes_0[:, None, :2], boxes_1[:, :2])  # [N,M,2]
    rb = torch.min(boxes_0[:, None, 2:], boxes_1[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    if cal_type == -1:
        iou = inter / (area_0[:, None] + area_1 - inter)
    elif cal_type == 0:
        iou = inter / (
            area_0[:, None]
            + torch.zeros(area_1.shape[0], dtype=torch.float32, device=area_0.device)
        )
    elif cal_type == 1:
        iou = inter / (
            torch.zeros(area_0.shape[0], dtype=torch.float32, device=area_1.device)[
                :, None
            ]
            + area_1
        )
    else:
        raise ValueError
    if isinstance(iou, torch.Tensor):
        return iou.cpu().numpy()
    else:
        return iou


def cal_line_iou(line_0, line_1, cal_type=-1):
    l = min(line_0 + line_1)
    r = max(line_0 + line_1)
    s_0 = abs(line_0[0] - line_0[1])
    s_1 = abs(line_1[0] - line_1[1])
    if cal_type == -1:
        return ((s_0 + s_1) - (r - l)) / (r - l + 1e-9)
    elif cal_type == 0:
        return ((s_0 + s_1) - (r - l)) / (s_0 + 1e-9)
    elif cal_type == 1:
        return ((s_0 + s_1) - (r - l)) / (s_1 + 1e-9)
    else:
        raise TypeError


def sort_box_text(box_list, text_list):
    box_list_ori, text_list_ori = copy.deepcopy(box_list), copy.deepcopy(text_list)
    box_list, text_list = np.array(box_list), np.array(text_list)
    if len(box_list) == 0:
        return box_list, text_list, []

    deformation_flag = False
    if len(box_list[0]) == 4:
        box_list = np.reshape(box_list, (-1, 8))
        deformation_flag = True
    box_list, text_list = (
        box_list[box_list[:, 1].argsort()],
        text_list[box_list[:, 1].argsort()],
    )
    box_list__ = []
    text_list__ = []
    temp_box_list = [box_list[0]]
    temp_text_list = [text_list[0]]
    for ii in range(1, len(box_list)):
        # if abs(box_list[ii][1] - box_list[ii - 1][1]) / abs(box_list[ii - 1][1] - box_list[ii - 1][7]) < 0.4:
        if (
            (
                np.abs(box_list[ii][1] - np.array(temp_box_list)[:, 1])
                / np.abs(np.array(temp_box_list)[:, 7] - np.array(temp_box_list)[:, 1])
            )
            < 0.4
        ).all():
            temp_box_list.append(box_list[ii])
            temp_text_list.append(text_list[ii])
        else:
            temp_box_list, temp_text_list = np.array(temp_box_list), np.array(
                temp_text_list
            )
            temp_box_list, temp_text_list = list(
                temp_box_list[temp_box_list[:, 0].argsort()]
            ), list(temp_text_list[temp_box_list[:, 0].argsort()])
            box_list__.extend(temp_box_list)
            text_list__.extend(temp_text_list)
            temp_box_list = [box_list[ii]]
            temp_text_list = [text_list[ii]]
    if len(temp_box_list) > 0:
        temp_box_list, temp_text_list = np.array(temp_box_list), np.array(
            temp_text_list
        )
        temp_box_list, temp_text_list = list(
            temp_box_list[temp_box_list[:, 0].argsort()]
        ), list(temp_text_list[temp_box_list[:, 0].argsort()])
        box_list__.extend(temp_box_list)
        text_list__.extend(temp_text_list)
    box_list = np.array(box_list__).tolist()
    text_list = np.array(text_list__).tolist()
    if deformation_flag:
        box_list = np.reshape(box_list, (-1, 4, 2)).tolist()

    # 返回排序后的索引
    sorted_idx = []
    for box in box_list:
        for idx_1, box_ori in enumerate(box_list_ori):
            if list(box) == list(box_ori):
                sorted_idx.append(idx_1)
                break
            if idx_1 == len(box_list_ori) - 1:
                raise ValueError
    if len(sorted_idx) != len(box_list):
        raise ValueError
    return box_list, text_list, sorted_idx


def det_lines(img, cal_type=True):
    """
    检测横线或者竖线(≈90°或≈180°的线)
    :param img:
    :param cal_type: True->横线, False->竖线
    :return:list->
        [
            [[x, y], [x, y]],
            ……
        ]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, 1
    )
    bw = cv2.bitwise_not(bw)

    # bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    line_p = bw.copy()
    if cal_type:
        line_p_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (59, 1))
    else:
        line_p_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 59))
    line_p = cv2.erode(line_p, line_p_structure)
    line_p = cv2.dilate(line_p, line_p_structure)
    line_p = cv2.dilate(
        line_p, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3
    )
    line_p = cv2.erode(
        line_p, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
    )

    contours, line_p = cv2.findContours(line_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for elem in contours:
        elem = np.squeeze(elem)
        elem = np.array(sorted(elem, key=lambda x: x[0]))
        try:
            if cal_type:
                lines.append([elem[0].tolist(), elem[-1].tolist()])
            else:
                lines.append([elem[0].tolist(), elem[2].tolist()])
        except:
            pass
    return lines


def shrink_box(box):
    x_0, x_1 = int(box[0] + 0), int(box[2] - 0)
    if box[3] - box[1] > 35:
        y_0, y_1 = int(box[1] + 8), int(box[3] - 8)
    elif box[3] - box[1] > 30:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 7), int((box[1] + box[3]) // 2 + 7)
    elif box[3] - box[1] > 25:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 6), int((box[1] + box[3]) // 2 + 6)
    elif box[3] - box[1] > 20:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 5), int((box[1] + box[3]) // 2 + 5)
    elif box[3] - box[1] > 15:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 4), int((box[1] + box[3]) // 2 + 4)
    elif box[3] - box[1] > 10:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 3), int((box[1] + box[3]) // 2 + 3)
    else:
        y_0, y_1 = int((box[1] + box[3]) // 2 - 2), int((box[1] + box[3]) // 2 + 2)
    return [x_0, y_0, x_1, y_1]


def horizontal_line_detection(image, mp_param=29):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1
    )
    bw = cv2.bitwise_not(bw)

    horizontal = bw.copy()

    # [horizontal lines]
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (mp_param, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.dilate(
        horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3
    )
    # horizontal = cv2.erode(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    # horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
    # horizontal = cv2.erode(horizontal, (1, 1), iterations=3)
    # cv2.imshow('horizontal', horizontal)
    # cv2.waitKey()

    hor_lines = cv2.HoughLinesP(
        horizontal,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=3,
    )

    if hor_lines is None:
        hor = []
    else:
        temp_line = []
        for line in hor_lines:
            for x1, y1, x2, y2 in line:
                temp_line.append([x1, y1 - 5, x2, y2 - 5])

        hor_lines = sorted(temp_line, key=lambda x: x[1])

        # Selection of best lines from all the horizontal lines detected
        lasty1 = -111111
        lines_x1 = []
        lines_x2 = []
        hor = []
        i = 0
        for x1, y1, x2, y2 in hor_lines:
            if y1 >= lasty1 and y1 <= lasty1 + 10:
                lines_x1.append(x1)
                lines_x2.append(x2)
            else:
                if i != 0 and len(lines_x1) != 0:
                    hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
                lasty1 = y1
                lines_x1 = []
                lines_x2 = []
                lines_x1.append(x1)
                lines_x2.append(x2)
                i += 1
        hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    return hor


def vertical_line_detection(image, mp_param=29):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1
    )
    bw = cv2.bitwise_not(bw)

    vertical = bw.copy()

    # [vertical lines]
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, mp_param))

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(
        vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=3
    )
    # vertical = cv2.erode(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    # vertical = cv2.dilate(vertical, (1, 1), iterations=9)
    # vertical = cv2.erode(vertical, (1, 1), iterations=3)
    # cv2.imwrite('./buffer/vv.jpg', vertical)
    # cv2.imshow('vertical', vertical)
    # cv2.waitKey()

    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 20, np.array([]), 20, 2)

    if ver_lines is None:
        ver = []
    else:
        temp_line = []
        for line in ver_lines:
            for x1, y1, x2, y2 in line:
                temp_line.append([x1, y1, x2, y2])

        # Sorting the list of detected lines by X1
        ver_lines = sorted(temp_line, key=lambda x: x[0])

        ## Selection of best lines from all the vertical lines detected ##
        lastx1 = -111111
        lines_y1 = []
        lines_y2 = []
        ver = []
        count = 0
        lasty1 = -11111
        lasty2 = -11111
        for x1, y1, x2, y2 in ver_lines:
            if (
                x1 >= lastx1
                and x1 <= lastx1 + 15
                and not (
                    (
                        (
                            min(y1, y2) < min(lasty1, lasty2) - 20
                            or min(y1, y2) < min(lasty1, lasty2) + 20
                        )
                    )
                    and (
                        (
                            max(y1, y2) < max(lasty1, lasty2) - 20
                            or max(y1, y2) < max(lasty1, lasty2) + 20
                        )
                    )
                )
            ):
                lines_y1.append(y1)
                lines_y2.append(y2)
                # lasty1 = y1
                # lasty2 = y2
            else:
                if count != 0 and len(lines_y1) != 0:
                    ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
                lastx1 = x1
                lines_y1 = []
                lines_y2 = []
                lines_y1.append(y1)
                lines_y2.append(y2)
                count += 1
                lasty1 = -11111
                lasty2 = -11111
        ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])

    return ver


def find_best_matches(A, B, threshold=0.65):
    """寻找A和B之间最佳匹配"""
    if not A or not B:
        return {}

    cost_matrix = calculate_similarity_matrix(A, B)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    row_idx_filter = []
    col_idx_filter = []
    for row, col in zip(row_ind, col_ind):
        similarity = 1 - cost_matrix[row, col]
        if similarity >= threshold:  # 只考虑相似度大于等于阈值的配对
            row_idx_filter.append(row)
            col_idx_filter.append(col)
    return row_idx_filter, col_idx_filter


def calculate_similarity_matrix(A, B):
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
            cost = 1 - similarity / 100
            matrix[i, j] = cost  # 不再使用np.inf，而是保留高成本
    return matrix


# def chinese_to_arabic(cn_str):
#     """将中文数字转换为阿拉伯数字"""
#     # 中文数字映射
#     CN_NUM = {
#         "一": 1,
#         "二": 2,
#         "三": 3,
#         "四": 4,
#         "五": 5,
#         "六": 6,
#         "七": 7,
#         "八": 8,
#         "九": 9,
#         "十": 10,
#         "百": 100,
#         "零": 0,
#     }
#     if not cn_str:
#         return 0

#     # 只有一个字符的情况
#     if len(cn_str) == 1:
#         return CN_NUM.get(cn_str, 0)

#     # 处理"十"开头的特殊情况
#     if cn_str.startswith("十"):
#         if len(cn_str) == 1:
#             return 10
#         return 10 + CN_NUM.get(cn_str[1], 0)

#     # 处理"十"字在中间的情况
#     if "十" in cn_str:
#         pos = cn_str.index("十")
#         return (
#             CN_NUM.get(cn_str[0], 0) * 10 + CN_NUM.get(cn_str[pos + 1 :], 0)
#             if pos + 1 < len(cn_str)
#             else CN_NUM.get(cn_str[0], 0) * 10
#         )


def chinese_to_arabic(cn_str):
    """将中文数字转换为阿拉伯数字"""
    CN_NUM = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
        "零": 0,
    }

    if not cn_str:
        return 0

    total = 0
    current = 0  # 当前累积的数字

    for char in cn_str:
        num = CN_NUM.get(char)
        if num is None:
            raise ValueError(f"无法识别的字符: {char}")
        if num >= 10:  # 处理单位（十、百）
            # 如果前面没有数字（如"十"），视为1*单位
            if current == 0:
                current = 1
            total += current * num
            current = 0  # 重置当前累积的数字
        else:  # 处理数字
            current += num

    total += current  # 添加最后的数字部分
    return total


def extract_numbers(text):
    """
    提取并转换字符串中的数字
    :param text: str, 输入字符串
    :return: list[int], 提取的数字列表
    """
    # 匹配连续的中文数字或阿拉伯数字
    pattern = r"([一二三四五六七八九十百千零]+|[0-9]+)"
    numbers = []

    for match in re.finditer(pattern, text):
        num_str = match.group(1)
        # 判断是否为阿拉伯数字
        if num_str.isdigit():
            numbers.append(int(num_str))
        else:
            numbers.append(chinese_to_arabic(num_str))

    return numbers


def du_table_convert_to_html_table(data):
    # 写入html开始
    max_row = max([elem[0][2] for elem in data])
    max_col = max([elem[0][3] for elem in data])
    html_table_list = []
    for i in range(max_row):
        html_h_list = []
        for j in range(max_col):
            html_h_list.append("")
        html_table_list.append(html_h_list)

    index = 0
    for elem in data:
        y_0, x_0, y_1, x_1 = elem[0]
        cell_str = "<td "
        cell_str = cell_str + "class=" + '"' + "tg-0lax" + '" '
        cell_str = (
            cell_str + "rowspan=" + '"' + str(y_1 - y_0) + '" '
        )  # 向下融合cell的数量
        cell_str = (
            cell_str + "colspan=" + '"' + str(x_1 - x_0) + '" '
        )  # 向右融合cell的数量
        # cell_str = cell_str + "height=" + '"' + str(box[3] - box[1]) + '" '  # 设置cell的宽
        # cell_str = cell_str + "width=" + '"' + str(box[2] - box[0]) + '" '  # 设置cell的高
        cell_str = cell_str + ">"
        cell_str = cell_str + elem[1].replace("<***۞Enter۞***>", "<br>")  # 文本内容
        cell_str = cell_str + "</td>"  # 结束符
        html_table_list[y_0][x_0] = cell_str
        index += 1
    html_str = """
        <html>
            <head> <meta charset="UTF-8">
                <style>
                table, th, td {
                    border: 1px solid black;
                    font-size: 20px;
                    border-collapse: collapse;
                }
                </style> 
            </head>
            <body>
                <table>

        """
    for i in html_table_list:
        if i == [""] * len(i):
            continue
        html_str += "<tr>\n"
        for j in i:
            if j != "":
                html_str += j + "\n"
        html_str += "</tr>\n"
    html_str += "</table>\n</body>\n</html>\n"
    return html_str
