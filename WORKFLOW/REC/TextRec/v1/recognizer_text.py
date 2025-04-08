# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 10:49
# @Author  : lijun
import os
import sys
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import copy
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# from WORKFLOW.OTHER.torchocr.torch_utils import select_device
import yaml

from MODELALG.utils.common import Log, select_device
from WORKFLOW.OTHER.torchocr.data import create_operators, transform
from WORKFLOW.OTHER.torchocr.modeling.architectures import build_model
from WORKFLOW.OTHER.torchocr.postprocess import build_post_process
from WORKFLOW.OTHER.torchocr.utility import (
    build_rec_process,
    get_others,
    update_rec_head_out_channels,
    width_pad_img,
)
from WORKFLOW.OTHER.torchocr.utils.ckpt import load_ckpt

# import threading


logger = Log(__name__).get_logger()
# lock = threading.Lock()


class CTCLabelDecode:
    def __init__(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ):
        self.character = self.get_character(character, character_path)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def __call__(
        self, preds: np.ndarray, return_word_box: bool = False, **kwargs
    ) -> List[Tuple[str, float]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx, preds_prob, return_word_box, is_remove_duplicate=True
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        return text

    def get_character(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ) -> List[str]:
        if character is None and character_path is None:
            raise ValueError("character must not be None")

        character_list = None
        if character:
            character_list = character

        if character_path:
            character_list = self.read_character_file(character_path)

        if character_list is None:
            raise ValueError("character must not be None")

        character_list = self.insert_special_char(
            character_list, " ", len(character_list)
        )
        character_list = self.insert_special_char(character_list, "blank", 0)
        return character_list

    @staticmethod
    def read_character_file(character_path: Union[str, Path]) -> List[str]:
        character_list = []
        with open(character_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_list.append(line)
        return character_list

    @staticmethod
    def insert_special_char(
        character_list: List[str], special_char: str, loc: int = -1
    ) -> List[str]:
        character_list.insert(loc, special_char)
        return character_list

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: Optional[np.ndarray] = None,
        return_word_box: bool = False,
        is_remove_duplicate: bool = False,
    ) -> List[Tuple[str, float]]:
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            if text_prob is not None:
                conf_list = np.array(text_prob[batch_idx][selection]).tolist()
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            text = "".join(char_list)
            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                            conf_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    @staticmethod
    def get_word_info(
        text: str, selection: np.ndarray
    ) -> Tuple[List[List[str]], List[List[int]], List[str]]:
        """
        Group the decoded characters and record the corresponding decoded positions.
        from https://github.com/PaddlePaddle/PaddleOCR/blob/fbba2178d7093f1dffca65a5b963ec277f1a6125/ppocr/postprocess/rec_postprocess.py#L70

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continous chinese characters (e.g., 你好啊)
                        - 'en&num': continous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection)[0]
        col_width = np.zeros(valid_col.shape)
        if len(valid_col) > 0:
            col_width[1:] = valid_col[1:] - valid_col[:-1]
            col_width[0] = min(
                3 if "\u4e00" <= text[0] <= "\u9fff" else 2, int(valid_col[0])
            )

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            else:
                c_state = "en&num"

            if state is None:
                state = c_state

            if state != c_state or col_width[c_i] > 4:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    @staticmethod
    def get_ignored_tokens() -> List[int]:
        return [0]  # for ctc blank


class CalRecBoxes:
    """计算识别文字的汉字单字和英文单词的坐标框。代码借鉴自PaddlePaddle/PaddleOCR和fanqie03/char-detection"""

    def __init__(self):
        pass

    def __call__(
        self,
        imgs: Optional[List[np.ndarray]],
        dt_boxes: Optional[List[np.ndarray]],
        rec_res: Optional[List[Any]],
    ):
        res = []
        for img, box, rec_res in zip(imgs, dt_boxes, rec_res):
            direction = self.get_box_direction(box)

            rec_txt, rec_conf, rec_word_info = rec_res[0], rec_res[1], rec_res[2]
            h, w = img.shape[:2]
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                rec_txt, img_box, rec_word_info
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            word_box_list = self.reverse_rotate_crop_image(
                copy.deepcopy(box), word_box_list, direction
            )
            res.append(
                [rec_txt, rec_conf, word_box_list, word_box_content_list, conf_list]
            )
        return res

    @staticmethod
    def get_box_direction(box: np.ndarray) -> str:
        direction = "w"
        img_crop_width = int(
            max(
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[2] - box[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(box[0] - box[3]),
                np.linalg.norm(box[1] - box[2]),
            )
        )
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            direction = "h"
        return direction

    @staticmethod
    def cal_ocr_word_box(
        rec_txt: str, box: np.ndarray, rec_word_info: List[Tuple[str, List[int]]]
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        """

        col_num, word_list, word_col_list, state_list, conf_list = rec_word_info
        box = box.tolist()
        bbox_x_start = box[0][0]
        bbox_x_end = box[1][0]
        bbox_y_start = box[0][1]
        bbox_y_end = box[2][1]

        cell_width = (bbox_x_end - bbox_x_start) / col_num
        word_box_list = []
        word_box_content_list = []
        cn_width_list = []
        en_width_list = []
        cn_col_list = []
        en_col_list = []

        def cal_char_width(width_list, word_col_):
            if len(word_col_) == 1:
                return
            char_total_length = (word_col_[-1] - word_col_[0]) * cell_width
            char_width = char_total_length / (len(word_col_) - 1)
            width_list.append(char_width)

        def cal_box(col_list, width_list, word_box_list_):
            if len(col_list) == 0:
                return
            if len(width_list) != 0:
                avg_char_width = np.mean(width_list)
            else:
                avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_txt)

            for center_idx in col_list:
                center_x = (center_idx + 0.5) * cell_width
                cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
                cell_x_end = (
                    min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                    + bbox_x_start
                )
                cell = [
                    [cell_x_start, bbox_y_start],
                    [cell_x_end, bbox_y_start],
                    [cell_x_end, bbox_y_end],
                    [cell_x_start, bbox_y_end],
                ]
                word_box_list_.append(cell)

        for word, word_col, state in zip(word_list, word_col_list, state_list):
            if state == "cn":
                cal_char_width(cn_width_list, word_col)
                cn_col_list += word_col
                word_box_content_list += word
            else:
                cal_char_width(en_width_list, word_col)
                en_col_list += word_col
                word_box_content_list += word

        cal_box(cn_col_list, cn_width_list, word_box_list)
        cal_box(en_col_list, en_width_list, word_box_list)
        sorted_word_box_list = sorted(word_box_list, key=lambda box: box[0][0])
        return word_box_content_list, sorted_word_box_list, conf_list

    @staticmethod
    def adjust_box_overlap(
        word_box_list: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        # 调整bbox有重叠的地方
        for i in range(len(word_box_list) - 1):
            cur, nxt = word_box_list[i], word_box_list[i + 1]
            if cur[1][0] > nxt[0][0]:  # 有交集
                distance = abs(cur[1][0] - nxt[0][0])
                cur[1][0] -= distance / 2
                cur[2][0] -= distance / 2
                nxt[0][0] += distance - distance / 2
                nxt[3][0] += distance - distance / 2
        return word_box_list

    def reverse_rotate_crop_image(
        self,
        bbox_points: np.ndarray,
        word_points_list: List[List[List[int]]],
        direction: str = "w",
    ) -> List[List[List[int]]]:
        """
        get_rotate_crop_image的逆操作
        img为原图
        part_img为crop后的图
        bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
        part_points为在part_img中的点[(x, y), (x, y)]
        """
        bbox_points = np.float32(bbox_points)

        left = int(np.min(bbox_points[:, 0]))
        top = int(np.min(bbox_points[:, 1]))
        bbox_points[:, 0] = bbox_points[:, 0] - left
        bbox_points[:, 1] = bbox_points[:, 1] - top

        img_crop_width = int(np.linalg.norm(bbox_points[0] - bbox_points[1]))
        img_crop_height = int(np.linalg.norm(bbox_points[0] - bbox_points[3]))

        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(bbox_points, pts_std)
        _, IM = cv2.invert(M)

        new_word_points_list = []
        for word_points in word_points_list:
            new_word_points = []
            for point in word_points:
                new_point = point
                if direction == "h":
                    new_point = self.s_rotate(
                        math.radians(-90), new_point[0], new_point[1], 0, 0
                    )
                    new_point[0] = new_point[0] + img_crop_width

                p = np.float32(new_point + [1])
                x, y, z = np.dot(IM, p)
                new_point = [x / z, y / z]

                new_point = [int(new_point[0] + left), int(new_point[1] + top)]
                new_word_points.append(new_point)
            new_word_points = self.order_points(new_word_points)
            new_word_points_list.append(new_word_points)
        return new_word_points_list

    @staticmethod
    def s_rotate(angle, valuex, valuey, pointx, pointy):
        """绕pointx,pointy顺时针旋转
        https://blog.csdn.net/qq_38826019/article/details/84233397
        """
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (
            (valuex - pointx) * math.cos(angle)
            + (valuey - pointy) * math.sin(angle)
            + pointx
        )
        sRotatey = (
            (valuey - pointy) * math.cos(angle)
            - (valuex - pointx) * math.sin(angle)
            + pointy
        )
        return [sRotatex, sRotatey]

    @staticmethod
    def order_points(box: List[List[int]]) -> List[List[int]]:
        """矩形框顺序排列"""

        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))
            elif p.shape == (1, 2):
                return p
            else:
                return p[:1, :]

        box = np.array(box).reshape((-1, 2))
        center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
        if np.any(box[:, 0] == center_x) and np.any(
            box[:, 1] == center_y
        ):  # 有两点横坐标相等，有两点纵坐标相等，菱形
            p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
            p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
            p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
            p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
        elif np.all(box[:, 0] == center_x):  # 四个点的横坐标都相同
            y_sort = np.argsort(box[:, 1])
            p1 = box[y_sort[0]]
            p2 = box[y_sort[1]]
            p3 = box[y_sort[2]]
            p4 = box[y_sort[3]]
        elif np.any(box[:, 0] == center_x) and np.all(
            box[:, 1] != center_y
        ):  # 只有两点横坐标相等，先上下再左右
            p12, p34 = (
                box[np.where(box[:, 1] < center_y)],
                box[np.where(box[:, 1] > center_y)],
            )
            p1, p2 = (
                p12[np.where(p12[:, 0] == np.min(p12[:, 0]))],
                p12[np.where(p12[:, 0] == np.max(p12[:, 0]))],
            )
            p3, p4 = (
                p34[np.where(p34[:, 0] == np.max(p34[:, 0]))],
                p34[np.where(p34[:, 0] == np.min(p34[:, 0]))],
            )
        else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
            p14, p23 = (
                box[np.where(box[:, 0] < center_x)],
                box[np.where(box[:, 0] > center_x)],
            )
            p1, p4 = (
                p14[np.where(p14[:, 1] == np.min(p14[:, 1]))],
                p14[np.where(p14[:, 1] == np.max(p14[:, 1]))],
            )
            p2, p3 = (
                p23[np.where(p23[:, 1] == np.min(p23[:, 1]))],
                p23[np.where(p23[:, 1] == np.max(p23[:, 1]))],
            )

        # 解决单字切割后横坐标完全相同的shape错误
        p1 = convert_to_1x2(p1)
        p2 = convert_to_1x2(p2)
        p3 = convert_to_1x2(p3)
        p4 = convert_to_1x2(p4)
        return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()


class Recognizer:
    def __init__(
        self,
        model_path,
        batch_size=4,
        device="cuda:0",
        alphabets_path=os.path.abspath(cur_dir + "/config/ppocr_keys_v1.txt"),
        half_flag=False,
    ):
        self.batch_size = batch_size
        self.device = select_device(device)
        self.half_flag = half_flag
        self.configs = yaml.load(
            open(cur_dir + "/config/ch_PP-OCRv4_rec_hgnet.yml", "r"),
            Loader=yaml.FullLoader,
        )
        self.configs["Global"]["character_dict_path"] = alphabets_path
        self.configs["PostProcess"]["character_dict_path"] = alphabets_path
        self.configs["Global"]["pretrained_model"] = model_path

        # build post process
        self.post_process_class = build_post_process(self.configs["PostProcess"])

        # build model
        update_rec_head_out_channels(self.configs, self.post_process_class)
        self.model = build_model(self.configs["Architecture"])
        load_ckpt(self.model, self.configs)
        self.model.eval()
        self.model.to(self.device)
        if self.half_flag and self.device.type != "cpu":
            self.model.half()

        # create data ops
        transforms = build_rec_process(self.configs)
        self.configs["Global"]["infer_mode"] = True
        self.ops = create_operators(transforms, self.configs["Global"])
        self.cal_rec_boxes = CalRecBoxes()
        self.character = []
        with open(cur_dir + "/config/ppocr_keys_v1.txt", "r") as f:
            for line in f:
                self.character.append(line.strip())
        self.postprocess_op = CTCLabelDecode(
            character=self.character, character_path=None
        )
        logger.info(" ···-> load model succeeded!")

    def __call__(self, imgs, char_box_enable: bool = False):
        """
        该接口用来识别文本行；
        :param imgs: opencv读取格式图片列表
        :return:识别出来的text，例如：
                [[('text', [conf, conf, conf, conf])],
                 [('……', [……])]]
        """
        if len(imgs) == 0:
            return []
        try:
            # with lock:
            # 预处理根据训练来
            if not isinstance(imgs, list):
                imgs = [imgs]
            batchs = [
                transform(
                    {"image": cv2.imencode(".png", img)[1].tobytes()},
                    self.ops,
                )
                for img in imgs
            ]
            others = get_others(self.configs, batchs[0])
            tfm_imgs = [batch[0] for batch in batchs]

            widths = np.array([img.shape[-1] for img in tfm_imgs])
            idxs = np.argsort(widths)
            txts = []
            for idx in range(0, len(tfm_imgs), self.batch_size):
                batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
                batch_imgs = np.array(
                    [
                        width_pad_img(
                            tfm_imgs[idx_1], tfm_imgs[batch_idxs[-1]].shape[-1]
                        )
                        for idx_1 in batch_idxs
                    ]
                )

                tensor = torch.from_numpy(batch_imgs).float().to(self.device)
                if self.half_flag and self.device.type != "cpu":
                    tensor = tensor.half()
                with torch.no_grad():
                    preds = self.model(tensor, others)
                preds["res"] = preds["res"].cpu().numpy()
                if self.half_flag and self.device.type != "cpu":
                    preds["res"] = preds["res"].astype(np.float32)

                if char_box_enable:
                    # char box post process -> start
                    max_wh_ratio = 6.66
                    wh_ratio_list = []
                    for ino in batch_idxs:
                        h, w = imgs[ino].shape[0:2]
                        wh_ratio = w * 1.0 / h
                        max_wh_ratio = max(max_wh_ratio, wh_ratio)
                        wh_ratio_list.append(wh_ratio)
                    rec_result = self.postprocess_op(
                        preds["res"],
                        True,
                        wh_ratio_list=wh_ratio_list,
                        max_wh_ratio=max_wh_ratio,
                    )
                    dt_boxes = [
                        np.array(
                            [
                                [0, 0],
                                [imgs[idx_1].shape[1], 0],
                                [imgs[idx_1].shape[1], imgs[idx_1].shape[0]],
                                [0, imgs[idx_1].shape[0]],
                            ],
                            dtype=np.float32,
                        )
                        for idx_1 in batch_idxs
                    ]
                    post_result = self.cal_rec_boxes(
                        [imgs[idx_2] for idx_2 in batch_idxs], dt_boxes, rec_result
                    )
                    txts.extend([[(elem[0], elem[4], elem[2])] for elem in post_result])
                    # char box post process -> end
                else:
                    post_result = self.post_process_class(preds)
                    txts.extend(
                        [
                            [(elem[0], [round(elem[1], 4)] * len(elem[0]), [])]
                            for elem in post_result
                        ]
                    )

            # 按输入图像的顺序排序
            idxs = np.argsort(idxs)
            out_txts = [txts[idx] for idx in idxs]
            return out_txts
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    path = "/volume/test_data/rec/"
    model = Recognizer(
        model_path=os.path.abspath("/volume/weights/Recognizer_text_model_v1.pt"),
        device="cuda:0",
        half_flag=False,
        batch_size=4,
    )

    image_name_list = os.listdir(path)
    imgs = []
    for image_name in image_name_list:
        # if image_name != "00001.jpg":
        #     continue
        print("processed img name: ", image_name)
        img = cv2.imread(path + image_name)
        imgs.append(img)
    for i in range(10):
        starttime = time.time()
        out = model(imgs, char_box_enable=True)
        # for idx in range(len(out)):
        #     show_img = copy.deepcopy(imgs[idx])
        #     cv2.polylines(
        #         show_img,
        #         np.array(out[idx][0][2]),
        #         isClosed=True,
        #         color=(0, 255, 0),
        #         thickness=1,
        #     )
        #     print()
        print("rec use time: ", time.time() - starttime)
        # print(out)
