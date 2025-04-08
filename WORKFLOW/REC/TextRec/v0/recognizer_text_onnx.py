# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 10:49
# @Author  : lijun
import os
import sys
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/REC/CRNN/CRNNv0/")))

import cv2
import traceback
import torch
import numpy as np
import onnxruntime as ort
from MODELALG.REC.CRNN.CRNNv0.networks import build_model
from MODELALG.REC.CRNN.CRNNv0.datasets.RecDataSet import RecDataProcess
from MODELALG.REC.CRNN.CRNNv0.utils import CTCLabelConverter
from MODELALG.REC.CRNN.CRNNv0.utils.torch_utils import select_device
from MODELALG.utils.common import Log


logger = Log(__name__).get_logger()


class Recognizer:
    def __init__(
        self,
        model_path,
        batch_size=16,
        device="0",
        alphabets_path=os.path.abspath(
            root_dir + "/MODELALG/REC/CRNN/CRNNv0/datasets/alphabets/ppocr_keys_v1.txt"
        ),
        half_flag=False,
    ):
        ckpt = torch.load("/volume/weights/Recognizer_text_model.pt", map_location="cpu")
        cfg = ckpt["cfg"]
        self.ort_session = ort.InferenceSession(
            "/volume/weights/Recognizer_text_model.onnx",
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        # "user_compute_stream": 0,
                    },
                ),
            ],
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        self.process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
        self.converter = CTCLabelConverter(alphabets_path)
        self.batch_size = 32
        logger.info(" ···-> load model succeeded!")

    def __call__(self, imgs):
        """
        该接口用来识别文本行；
        :param imgs: opencv读取格式图片列表
        :return:识别出来的text，例如：
                [[('text', [conf, conf, conf, conf])],
                 [('……', [……])]]
        """
        try:
            # 预处理根据训练来
            # totaltime = time.time()
            if not isinstance(imgs, list):
                imgs = [imgs]
            imgs = [
                self.process.normalize_img(
                    self.process.resize_with_specific_height(img)
                )
                for img in imgs
            ]
            widths = np.array([img.shape[1] for img in imgs])
            idxs = np.argsort(widths)
            txts = []
            for idx in range(0, len(imgs), self.batch_size):
                starttime = time.time()
                batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
                batch_imgs = [
                    self.process.width_pad_img(imgs[idx], 1200)
                    for idx in batch_idxs
                ]
                batch_imgs = np.stack(batch_imgs).astype(np.float32)
                batch_imgs = batch_imgs.transpose([0, 3, 1, 2])
                batch_imgs = torch.tensor(batch_imgs, dtype=torch.float32)

                # # img = cv2.imread("/volume/test_data//my_imgs_0/0.jpg")
                # # img = self.process.normalize_img(self.process.resize_with_specific_height(img))
                # # batch_imgs = [self.process.width_pad_img(img, img.shape[1])] * 32
                # batch_imgs = []
                # filename_list = os.listdir("/volume/test_data/my_imgs_0/")
                # for filename in filename_list[:32]:
                #     img = cv2.imread("/volume/test_data/my_imgs_0/" + filename)
                #     img = self.process.normalize_img(self.process.resize_with_specific_height(img))
                #     batch_imgs.append(self.process.width_pad_img(img, 1200))
                # batch_imgs = np.stack(batch_imgs).astype(np.float32)
                # batch_imgs = batch_imgs.transpose([0, 3, 1, 2])
                # batch_imgs = torch.tensor(batch_imgs, dtype=torch.float32)

                # batch_imgs = torch.tensor(
                #     torch.rand(8, 3, 32, 1000) * 2 - 1, dtype=torch.float32
                # )
                ort_input = {self.input_name: batch_imgs.numpy()}
                logger.info("数据准备耗时: {}".format(str(round(time.time() - starttime, 5))))
                starttime = time.time()
                ort_output = self.ort_session.run([self.output_name], ort_input)[0]
                logger.info(
                    "模型推理耗时: {} ({})".format(
                        str(round(time.time() - starttime, 5)), str(batch_imgs.shape)
                    )
                )
                starttime = time.time()
                out = torch.from_numpy(ort_output).softmax(dim=2).numpy()
                logger.info(
                    "softmax耗时: {}".format(str(round(time.time() - starttime, 5)))
                )
                starttime = time.time()
                txts.extend(self.converter.decode(out))
                logger.info("结果处理耗时: {}".format(str(round(time.time() - starttime, 5))))
            # 按输入图像的顺序排序
            idxs = np.argsort(idxs)
            out_txts = [[txts[idx]] for idx in idxs]
            # logger.info("文本识别总耗时: {}".format(str(round(time.time() - totaltime, 5))))
            return out_txts
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    path = "/volume/test_data/my_imgs_0/"
    model = Recognizer("/volume/weights/Recognizer_text_model.onnx", device="0")
    image_name_list = os.listdir(path)
    imgs = []
    for image_name in image_name_list:
        print("processed img name: ", image_name)
        img = cv2.imread(path + image_name)
        imgs.append(img)
    out = model(imgs)
    print(out)

    # model_path = "/volume/weights/Recognizer_text_model.pt"
    # ckpt = torch.load(model_path, map_location="cpu")
    # cfg = ckpt["cfg"]
    #
    #
    # def make_img():
    #     process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
    #     img = cv2.imread("/volume/test_data//my_imgs_0/0.jpg")
    #     img = process.normalize_img(process.resize_with_specific_height(img))
    #     batch_imgs = [process.width_pad_img(img, img.shape[1])] * 32
    #     batch_imgs = np.stack(batch_imgs).astype(np.float32)
    #     batch_imgs = batch_imgs.transpose([0, 3, 1, 2])
    #     dummy_input = torch.tensor(batch_imgs, dtype=torch.float32)
    #     return dummy_input
    # ort_session = ort.InferenceSession(
    #     "/volume/weights/Recognizer_text_model.onnx",
    #     providers=[
    #         (
    #             "CUDAExecutionProvider",
    #             {
    #                 "device_id": 0,
    #                 # "user_compute_stream": 0,
    #             },
    #         ),
    #     ],
    # )
    # input_name = ort_session.get_inputs()[0].name
    # output_name = ort_session.get_outputs()[0].name
    # dummy_input = make_img()
    # # dummy_input = torch.tensor(torch.randn(8, 3, 32, 1000))
    # ort_input = {input_name: dummy_input.numpy()}
    #
    # for i in range(10):
    #     start_time = time.time()
    #     ort_outputs = ort_session.run([output_name], ort_input)
    #     print("use time: ", time.time() - start_time)
    #     time.sleep(1)
