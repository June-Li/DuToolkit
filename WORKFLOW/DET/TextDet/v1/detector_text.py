import os
import sys
import time
import traceback
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))

import cv2
import json
import copy
import torch
import yaml

from WORKFLOW.OTHER.torchocr.data import create_operators, transform
from WORKFLOW.OTHER.torchocr.modeling.architectures import build_model
from WORKFLOW.OTHER.torchocr.postprocess import build_post_process
from WORKFLOW.OTHER.torchocr.utils.ckpt import load_ckpt
from WORKFLOW.OTHER.torchocr.utility import build_det_process
from MODELALG.utils.common import Log, select_device, draw_poly_boxes, pdf2img


logger = Log(__name__).get_logger()


class Detector:
    def __init__(
        self,
        model_path,
        device="cuda:0",
        half_flag=False,
        unclip_ratio=2.0,
        expand_left_radio=0.0,
        expand_right_radio=0.0,
    ):
        self.cfg = yaml.load(
            open(cur_dir + "/config/ch_PP-OCRv4_det_teacher.yml", "r"),
            Loader=yaml.FullLoader,
        )
        self.cfg["Global"]["pretrained_model"] = model_path
        self.cfg["PostProcess"]["unclip_ratio"] = unclip_ratio
        self.cfg["PostProcess"]["expand_left_radio"] = expand_left_radio
        self.cfg["PostProcess"]["expand_right_radio"] = expand_right_radio
        self.global_config = self.cfg["Global"]

        # build model
        self.half_flag = half_flag
        self.device = select_device(device)
        self.model = build_model(self.cfg["Architecture"])
        load_ckpt(self.model, self.cfg)
        self.model.to(self.device)
        self.model.eval()
        if self.half_flag and self.device.type != "cpu":
            self.model.half()

        # build post process
        self.post_process_class = build_post_process(self.cfg["PostProcess"])

        # create data ops
        transforms = build_det_process(self.cfg)
        self.ops = create_operators(transforms, self.global_config)
        logger.info(" ···-> load model succeeded!")

    def __call__(self, img):
        """
        该接口用来检测文本行；
        :param img: opencv读取格式图片
        :return box_array: 文本行的box
                eg：
                    [[[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
                     ……]
        :return score_array: 文本行置信度得分
                eg：
                    [0.951253, ……]
        """
        try:
            # with lock:
            # 预处理根据训练来
            # total_time = time.time()
            # data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
            # data = self.resize(data)
            # tensor = self.transform(data["img"])
            # tensor = tensor.unsqueeze(dim=0)
            # tensor = tensor.to(self.device)
            # if self.half_flag and self.device.type != "cpu":
            #     tensor = tensor.half()
            # # starttime = time.time()
            # with torch.no_grad():
            #     out = self.model(tensor)
            #     # traced_script_module = torch.jit.trace(self.model, tensor)
            #     # traced_script_module.save("/workspace/JuneLi/a.pt")
            # # print("模型推理耗时: ", time.time() - starttime)
            # # starttime = time.time()
            # out = out.cpu().numpy()
            # # print("结果放到cpu耗时：", time.time() - starttime)
            # # starttime = time.time()
            # if self.half_flag and self.device.type != "cpu":
            #     out = out.astype(np.float32)
            # # print("结果类型转换耗时: ", time.time() - starttime)
            # # starttime = time.time()
            # radio = [
            #     tensor.shape[2] / img.shape[0],
            #     tensor.shape[3] / img.shape[1],
            # ]
            # post_process_out = self.post_process(
            #     {"res": out},
            #     [
            #         -1,
            #         [list(tensor.shape[2:]) + radio],
            #     ],
            # )
            # # print("后处理耗时: ", time.time() - starttime)
            # box_array, score_array = (
            #     post_process_out[0]["points"],
            #     post_process_out[0]["scores"],
            # )
            # if len(box_array) > 0:
            #     idx = [x.sum() > 0 for x in box_array]
            #     box_array = [box_array[i] for i, v in enumerate(idx) if v]
            #     score_array = [score_array[i] for i, v in enumerate(idx) if v]
            #
            #     box_array = np.array(box_array, dtype=float)
            #     box_array[:, :, 0] /= radio[1]
            #     box_array[:, :, 1] /= radio[0]
            #     box_array = np.array(box_array, dtype=int).tolist()
            # else:
            #     box_array, score_array = [], []
            #
            # # print(
            # #     "DB总耗时：",
            # #     time.time() - total_time,
            # # )
            #
            # for idx, box in enumerate(box_array):
            #     h = box[3][1] - box[0][1]
            #     box_array[idx][0][0] = max(
            #         box_array[idx][0][0]
            #         - int(h * self.cfg["post_process"]["expand_left_radio"]),
            #         0,
            #     )
            #     box_array[idx][1][0] = min(
            #         box_array[idx][1][0]
            #         + int(h * self.cfg["post_process"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][2][0] = min(
            #         box_array[idx][2][0]
            #         + int(h * self.cfg["post_process"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][3][0] = max(
            #         box_array[idx][3][0]
            #         - int(h * self.cfg["post_process"]["expand_left_radio"]),
            #         0,
            #     )
            data = {"image": cv2.imencode(".png", img)[1].tobytes()}
            batch = transform(data, self.ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images).to(self.device)
            if self.half_flag and self.device.type != "cpu":
                images = images.half()
            with torch.no_grad():
                starttime = time.time()
                preds = self.model(images)
                print("model infer use time: ", time.time() - starttime)
            post_result = self.post_process_class(preds, [-1, shape_list])

            box_array, score_array = (
                post_result[0]["points"].tolist(),
                post_result[0]["scores"],
            )
            # for idx, box in enumerate(box_array):
            #     h = box[3][1] - box[0][1]
            #     box_array[idx][0][0] = max(
            #         box_array[idx][0][0]
            #         - int(h * self.cfg["PostProcess"]["expand_left_radio"]),
            #         0,
            #     )
            #     box_array[idx][1][0] = min(
            #         box_array[idx][1][0]
            #         + int(h * self.cfg["PostProcess"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][2][0] = min(
            #         box_array[idx][2][0]
            #         + int(h * self.cfg["PostProcess"]["expand_right_radio"]),
            #         img.shape[1],
            #     )
            #     box_array[idx][3][0] = max(
            #         box_array[idx][3][0]
            #         - int(h * self.cfg["PostProcess"]["expand_left_radio"]),
            #         0,
            #     )
            retain = [
                idx
                for idx, box in enumerate(box_array)
                if np.var(img[box[0][1] : box[-1][1], box[0][0] : box[1][0]]) > 100
            ]
            box_array = np.array(box_array)[retain].tolist()
            score_array = np.array(score_array)[retain].tolist()
            return box_array, score_array
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python detector_text.py
    file_path = "/volume/test_data/my_imgs_0/6.jpg"
    # file_path = "/volume/test_data/多场景数据测试/other/19196dd61279e44359856d5437b8a74d.pdf"
    # path = "/volume/test_data/fake/0.jpg"
    imgs = [cv2.imread(file_path)]
    # imgs = list(pdf2img(file_path, dpi=150, max_len=3500))
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # model = Detector(os.path.abspath(root_dir + '/MODEL/DET/DB/DBv0/TextDet/20210601/best.pt'), gpu='0')
    model = Detector(
        os.path.abspath("/volume/weights/Detector_text_pp_ocrv4_server_model.pt"),
        device="cuda:0",
        unclip_ratio=2.0,
    )
    starttime = time.time()
    box_array, score_array = model(imgs[0])
    print("80-HiJuneLi文本检测总耗时: ", time.time() - starttime)
    # while True:
    #     starttime = time.time()
    #     box_array, score_array = model(img)
    #     print("80-HiJuneLi文本检测总耗时: ", time.time() - starttime)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_poly_boxes(imgs[0], box_array, thickness=1)
    print()

    # out_path = "/".join(path.replace("/test_data/", "/test_out/").split("/")[:-1])
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # cv2.imwrite(path.replace("/test_data/", "/test_out/"), img)
