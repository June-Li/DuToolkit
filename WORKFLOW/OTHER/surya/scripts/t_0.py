import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../")
sys.path.append(os.path.abspath(root_dir))

import time

from PIL import Image
from WORKFLOW.OTHER.surya.surya.detection import batch_text_detection
from WORKFLOW.OTHER.surya.surya.layout import batch_layout_detection
from WORKFLOW.OTHER.surya.surya.model.detection.segformer import (
    load_model,
    load_processor,
)
from WORKFLOW.OTHER.surya.surya.settings import settings
import cv2
import fitz
import numpy as np
import torch
from MODELALG.utils.common import pdf2img


# img_path = '/volume/test_data/my_imgs_0/4.jpg'
img_path = "/volume/test_data/多场景数据测试/ESG/20210601-中信证券-全球产业投资2021年下半年投资策略：ESG和AI引领下的全球产业投资策略.pdf"
outpath = "/volume/test_out/temp/11/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
if img_path.split(".")[-1] in ["jpg", "jpeg", "png"]:
    imgs = [cv2.imread(img_path)]
elif img_path.split(".")[-1] in ["pdf", "PDF"]:
    imgs = pdf2img(img_path, dpi=150, max_len=3000)
else:
    raise ValueError
# image = Image.open(img_path)
# img = cv2.imread(img_path)
# count = 0
# while True:
#     try:
#         print('正在尝试第{}次下载>>>>>>'.format(count), end='')
#         model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
#         print('下载成功🚀！！！')
#         break
#     except:
#         print('下载失败！！！')
#         count += 1
#         continue
# model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
# processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
# det_model = load_model()
# det_processor = load_processor()
model = load_model(checkpoint="/volume/weights/surya_layout2")
processor = load_processor(checkpoint="/volume/weights/surya_layout2")
det_model = load_model(checkpoint="/volume/weights/surya_det2")
det_processor = load_processor(checkpoint="/volume/weights/surya_det2")

for idx, img in enumerate(imgs):
    totaltime = time.time()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # starttime = time.time()
    line_predictions = batch_text_detection([pil_img], det_model, det_processor)
    # print("text det use time: ", time.time() - starttime)
    # starttime = time.time()
    layout_predictions = batch_layout_detection(
        [pil_img], model, processor, line_predictions
    )
    # print("layout det use time: ", time.time() - starttime)
    # print("total det use time: ", time.time() - totaltime)
    for elem in layout_predictions[0].bboxes:
        box, box_type = elem.bbox, elem.label
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(
            img,
            box_type,
            (box[0], box[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
        )
    for elem in line_predictions[0].bboxes:
        box = elem.bbox
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imwrite(outpath + str(idx) + ".jpg", img)
    print()
