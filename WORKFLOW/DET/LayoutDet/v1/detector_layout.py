import copy

import math
from PIL import Image
import threading
import torch

from WORKFLOW.OTHER.surya.surya.detection import batch_text_detection
from WORKFLOW.OTHER.surya.surya.layout import batch_layout_detection
from WORKFLOW.OTHER.surya.surya.model.detection.segformer import (
    load_model,
    load_processor,
)
from WORKFLOW.OTHER.surya.surya.schema import PolygonBox, ColumnLine
import cv2
import numpy as np
import time

# from MODELALG.DET.YOLO.YOLOv5.utils.torch_utils import select_device
from MODELALG.utils.common import Log, select_device, vertical_line_detection


logger = Log(__name__).get_logger()
lock = threading.Lock()


class TextDetectionResult:
    def __init__(self):
        self.bboxes = []
        self.vertical_lines = []


class Detector(object):
    def __init__(
        self,
        det_model_path="/volume/weights/surya_det2",
        model_path="/volume/weights/surya_layout2",
        thr=0.65,
        device="cuda:0",
    ):
        self.thr = thr
        self.device = select_device(device)
        self.det_model = (
            load_model(det_model_path, device="cpu", dtype=torch.float32)
            .to(self.device)
            .float()
        )
        self.det_processor = load_processor(det_model_path)
        self.model = (
            load_model(checkpoint=model_path, device="cpu", dtype=torch.float32)
            .to(self.device)
            .float()
        )
        self.processor = load_processor(checkpoint=model_path)
        logger.info(" ···-> load model succeeded!")

    def __call__(
        self, image_list, chrs_box_list, batch_size=16, det_type="ocr"
    ):  # surya 或 ocr
        # with lock:
        total_line_predictions = []
        res = []
        imgs = []
        for img in image_list:
            imgs.append(Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)))
        batch_loop_cnt = math.ceil(float(len(imgs)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(imgs))
            batch_image_list = imgs[start_index:end_index]

            if det_type == "surya":
                line_predictions = batch_text_detection(
                    batch_image_list, self.det_model, self.det_processor
                )
                total_line_predictions.extend(line_predictions)
            else:
                # 使用ocr的box结果 -> start
                line_predictions = []
                for idx_1, chrs_box in enumerate(chrs_box_list):
                    tdr = TextDetectionResult()
                    bboxes = []
                    for box in chrs_box:
                        bboxes.append(
                            PolygonBox(
                                polygon=np.reshape(box, (4, 2)).tolist(), confidence=1.0
                            )
                        )
                    tdr.bboxes = bboxes

                    vertical_lines = []
                    line_det_img = copy.deepcopy(image_list[start_index + idx_1])
                    if line_det_img.shape[0] < 1000:
                        radio = 1
                    else:
                        radio = line_det_img.shape[0] / 1000
                        line_det_img = cv2.resize(
                            line_det_img.copy(),
                            (int(line_det_img.shape[1] / radio), 1000),
                        )
                    v = np.array(
                        np.array(vertical_line_detection(line_det_img, mp_param=100))
                        * radio,
                        dtype=int,
                    ).tolist()
                    for box in v:
                        vertical_lines.append(
                            ColumnLine(bbox=box, vertical=True, horizontal=False)
                        )
                    tdr.vertical_lines = vertical_lines

                    line_predictions.append(tdr)
                total_line_predictions.extend(line_predictions)
                # 使用ocr的box结果 -> end

            # starttime = time.time()
            layout_predictions = batch_layout_detection(
                batch_image_list,
                self.model,
                self.processor,
                copy.deepcopy(line_predictions),
            )
            # print("layout_predictions use time: ", time.time() - starttime)
            for out in layout_predictions:
                r = []
                for box in out.bboxes:
                    if box.confidence > self.thr:
                        r.append(
                            {
                                "bbox": box.bbox,
                                "score": box.confidence,
                                "type": box.label,
                            }
                        )
                res.append(r)

        # show
        # for idx, r in enumerate(res):
        #     img = image_list[idx]
        #     for elem in r:
        #         bbox = elem["bbox"]
        #         cv2.rectangle(
        #             img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2
        #         )
        #     for bbox in chrs_box_list[idx]:
        #         cv2.rectangle(
        #             img, (bbox[0], bbox[1]), (bbox[4], bbox[5]), (255, 0, 0), 2
        #         )
        #     for bbox in total_line_predictions[idx].vertical_lines:
        #         bbox = np.array(bbox.bbox, dtype=int)
        #         cv2.rectangle(
        #             img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2
        #         )
        #     print()
        return res


if __name__ == "__main__":
    from MODELALG.utils.common import pdf2img

    op = Detector()
    # imgs = pdf2img("/volume/test_data/fake/pdf_fake_text_table.pdf")
    # imgs = pdf2img(
    #     "/volume/test_data/多场景数据测试/ESG/20210601-中信证券-全球产业投资2021年下半年投资策略：ESG和AI引领下的全球产业投资策略.pdf",
    #     dpi=150,
    #     max_len=3500,
    # )
    imgs = [cv2.imread("/volume/test_data/my_imgs_0/0.jpg")]
    # res = op(list(imgs))
    for idx, img in enumerate(imgs):
        res = op([img])
        for elem in res[0]:
            bbox = elem["bbox"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        print()
