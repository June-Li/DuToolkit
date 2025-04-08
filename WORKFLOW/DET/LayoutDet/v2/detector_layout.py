import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import copy
import math
import threading
import time
import traceback

import cv2
import numpy as np
from doclayout_yolo import YOLOv10

# from MODELALG.DET.YOLO.YOLOv5.utils.torch_utils import select_device
from MODELALG.utils.common import Log, cal_iou_parallel, select_device

logger = Log(__name__).get_logger()
lock = threading.Lock()


class TextDetectionResult:
    def __init__(self):
        self.bboxes = []
        self.vertical_lines = []


class Detector(object):
    def __init__(
        self,
        model_path="/volume/weights/Detector_layout_model.pt",
        thr=0.2,
        device="cuda:0",
    ):
        self.thr = thr
        self.device = select_device(device)
        self.model = YOLOv10(model_path).to(self.device).float()
        logger.info(" ···-> load model succeeded!")

    def __call__(self, image_list, batch_size=16):
        # with lock:
        res = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]

            # starttime = time.time()
            layout_predictions = self.model.predict(
                batch_image_list,  # Image to predict
                imgsz=1024,  # Prediction image size
                conf=self.thr,  # Confidence threshold
                device=self.device,  # Device to use (e.g., 'cuda:0' or 'cpu')
            )
            # print("layout_predictions use time: ", time.time() - starttime)

            # 去掉重复的框
            pages_predictions = {}
            for idx_0, out in enumerate(layout_predictions):
                ori_boxes_list = out.boxes.xyxy.cpu().numpy().astype(int).tolist()
                ori_cls_list = out.boxes.cls.cpu().numpy().tolist()
                ori_conf_list = out.boxes.conf.cpu().numpy().tolist()
                box_sort_idx_by_area = np.argsort(
                    (np.array(ori_boxes_list)[:, 2] - np.array(ori_boxes_list)[:, 0])
                    * (np.array(ori_boxes_list)[:, 3] - np.array(ori_boxes_list)[:, 1])
                )[::-1]
                ori_boxes_list = [ori_boxes_list[i] for i in box_sort_idx_by_area]
                ori_cls_list = [ori_cls_list[i] for i in box_sort_idx_by_area]
                ori_conf_list = [ori_conf_list[i] for i in box_sort_idx_by_area]
                boxes_list = [ori_boxes_list[0]]
                cls_list = [ori_cls_list[0]]
                conf_list = [ori_conf_list[0]]
                for box, cls, conf in zip(
                    ori_boxes_list[1:], ori_cls_list[1:], ori_conf_list[1:]
                ):
                    iou_matrix = cal_iou_parallel(boxes_list, [box], cal_type=1)
                    if np.max(iou_matrix) < 0.5:
                        boxes_list.append(box)
                        cls_list.append(cls)
                        conf_list.append(conf)
                pages_predictions[idx_0] = {
                    "boxes_list": np.array(boxes_list).astype(int).tolist(),
                    "cls_list": np.array(cls_list).astype(int).tolist(),
                    "conf_list": np.array(conf_list).astype(float).tolist(),
                }

            cls_idx_type_dict = {
                0: "text",  # doclayout: "title",
                1: "text",  # doclayout: "plain text",
                2: "text",  # doclayout: "abandon",
                3: "figure",  # doclayout: "figure",
                4: "text",  # doclayout: "figure_caption",
                5: "table",  # doclayout: "table",
                6: "text",  # doclayout: "table_caption",
                7: "text",  # doclayout: "table_footnote",
                8: "formula",  # doclayout: "isolate_formula",
                9: "text",  # doclayout: "formula_caption",
            }
            for page_idx in pages_predictions:
                r = []
                for box, cls, conf in zip(
                    pages_predictions[page_idx]["boxes_list"],
                    pages_predictions[page_idx]["cls_list"],
                    pages_predictions[page_idx]["conf_list"],
                ):
                    if cls == 8:
                        continue
                    if conf > self.thr:
                        r.append(
                            {
                                "bbox": box,
                                "score": conf,
                                "type": cls_idx_type_dict[cls],
                                "subtype": "title" if cls == 0 else None,
                            }
                        )
                res.append(r)

        # show
        img = copy.deepcopy(image_list[0])
        for elem in res[0]:
            bbox = elem["bbox"]
            type = elem["type"]
            score = elem["score"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{type}: {score:.2f}",
                (bbox[0], bbox[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
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
    imgs = [cv2.imread("/volume/test_data/fake/pdf_fake_text_table_0.png")]
    # res = op(list(imgs))
    for idx, img in enumerate(imgs):
        res = op([img])
        for elem in res[0]:
            bbox = elem["bbox"]
            type = elem["type"]
            score = elem["score"]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{type}: {score:.2f}",
                (bbox[0], bbox[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        print()
