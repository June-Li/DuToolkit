# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 17:27
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")

import os
import threading
import time
import traceback

import cv2
import torch
from rapid_table import RapidTable
from rapid_table.main import RapidTableInput

from MODELALG.utils.common import Log

logger = Log(__name__).get_logger()
lock = threading.Lock()


class Recognizer:
    def __init__(
        self,
        encoder_path,
        decoder_path,
        vocab_path,
        device="cuda:0",
    ):
        with torch.no_grad():
            if "cuda" in device or "npu" in device or "mlu" in device:
                use_cuda = True
            else:
                use_cuda = False

            input_args = RapidTableInput(
                model_type="unitable",
                model_path={
                    "encoder": encoder_path,
                    "decoder": decoder_path,
                    "vocab": vocab_path,
                },
            )
            self.table_engine = RapidTable(input_args)
            logger.info(" ···-> load model succeeded!")

    def __call__(self, imgs, ocr_results=None):
        """
        input:
            img_ori: opencv读取的图片格式;
        Returns:
            outs: 识别结果
        """
        try:
            with lock:
                outs = []
                for idx, img in enumerate(imgs):
                    table_result = self.table_engine(img, ocr_results[idx])
                    outs.append(table_result)
                return outs
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    recognizer = Recognizer()
    path = "/volume/test_data/无框表格/"
    image_name_list = os.listdir(path)
    for image_name in image_name_list:
        img_ori = cv2.imread(path + image_name)
        start = time.time()
        outs = recognizer([img_ori])
        print("per img use time: ", time.time() - start)
