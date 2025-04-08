import os

import math
from PIL import Image
from WORKFLOW.OTHER.surya.surya.detection import batch_text_detection
from WORKFLOW.OTHER.surya.surya.layout import batch_layout_detection
from WORKFLOW.OTHER.surya.surya.model.detection.segformer import (
    load_model,
    load_processor,
)
from WORKFLOW.OTHER.surya.surya.schema import PolygonBox, ColumnLine
import cv2
import numpy as np

# from MODELALG.DET.YOLO.YOLOv5.utils.torch_utils import select_device
from MODELALG.utils.common import Log, select_device, vertical_line_detection


logger = Log(__name__).get_logger()


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
            load_model(det_model_path, device="cpu").to(self.device).float()
        )
        self.det_processor = load_processor(det_model_path)
        self.model = (
            load_model(checkpoint=model_path, device="cpu").to(self.device).float()
        )
        self.processor = load_processor(checkpoint=model_path)

    def __call__(self, image_list, batch_size=16):
        res = []
        imgs = []
        for img in image_list:
            imgs.append(Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)))
        batch_loop_cnt = math.ceil(float(len(imgs)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(imgs))
            batch_image_list = imgs[start_index:end_index]
            line_predictions = batch_text_detection(
                batch_image_list, self.det_model, self.det_processor
            )
            # line_predictions[0].vertical_lines = [line_predictions[0].vertical_lines[0]]
            # bbox = PolygonBox(
            #     polygon=[[0, 0], [100, 0], [100, 100], [0, 100]], confidence=1.0
            # )
            # bboxes = [bbox]
            bboxes = []
            for polygon in line_predictions[0].bboxes:
                bboxes.append(PolygonBox(polygon=polygon.polygon, confidence=1.0))
            a = TextDetectionResult()
            a.bboxes = bboxes

            vertical_lines = []
            v = vertical_line_detection(image_list[0], mp_param=100)
            for box in v:
                vertical_lines.append(
                    ColumnLine(bbox=box, vertical=True, horizontal=False)
                )
            a.vertical_lines = vertical_lines

            # a.vertical_lines = line_predictions[0].vertical_lines
            line_predictions[0] = a

            #
            # img = image_list[0]
            # for bbox in line_predictions[0].bboxes:
            #     bbox = bbox.bbox
            #     cv2.rectangle(
            #         img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
            #     )
            # for bbox in line_predictions[0].vertical_lines:
            #     bbox = bbox.bbox
            #     cv2.rectangle(
            #         img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2
            #     )
            #

            layout_predictions = batch_layout_detection(
                batch_image_list, self.model, self.processor, line_predictions
            )
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
        return res


if __name__ == "__main__":
    from MODELALG.utils.common import pdf2img
    from tqdm import tqdm

    op = Detector()
    # imgs = pdf2img("/volume/test_data/fake/pdf_fake_text_table.pdf")
    # imgs = pdf2img(
    #     "/volume/test_data/多场景数据测试/ESG/20210601-中信证券-全球产业投资2021年下半年投资策略：ESG和AI引领下的全球产业投资策略.pdf",
    #     dpi=150,
    #     max_len=3500,
    # )
    base_path = "/volume/test_data/my_imgs_0/"
    filename_list = os.listdir(base_path)
    white_list = ["49.jpg"]
    for idx, filename in tqdm(enumerate(filename_list)):
        if not filename in white_list:
            continue
        img = cv2.imread(base_path + filename)
        res = op([img])
        for elem in res[0]:
            bbox = elem["bbox"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.imwrite("/volume/test_out/my_imgs_0/" + filename, img)
        print()
