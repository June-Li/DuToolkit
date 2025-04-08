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
        self.device = device
        self.half_flag = half_flag
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.model = build_model(cfg["model"])
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        self.model.load_state_dict(state_dict)

        self.device = select_device(self.device)
        self.model.to(self.device)
        self.model.eval()
        if self.half_flag and self.device.type != "cpu":
            self.model.half()

        self.process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
        self.converter = CTCLabelConverter(alphabets_path)
        self.batch_size = batch_size
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
                # starttime = time.time()
                batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
                batch_imgs = [
                    self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1])
                    for idx in batch_idxs
                ]
                batch_imgs = np.stack(batch_imgs)
                tensor = torch.from_numpy(batch_imgs.transpose([0, 3, 1, 2])).float()
                tensor = tensor.to(self.device)
                if self.half_flag and self.device.type != "cpu":
                    tensor = tensor.half()
                # logger.info("数据准备耗时: {}".format(str(round(time.time() - starttime, 5))))
                # starttime = time.time()
                with torch.no_grad():
                    out = self.model(tensor)
                    out = out.softmax(dim=2)
                # logger.info(
                #     "模型推理耗时: {} ({})".format(
                #         str(round(time.time() - starttime, 5)), str(tensor.shape)
                #     )
                # )
                # starttime = time.time()
                txts.extend(self.converter.decode_by_tensor(out))
                # logger.info("结果处理耗时: {}".format(str(round(time.time() - starttime, 5))))
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
    model = Recognizer("/volume/weights/Recognizer_text_model.pt", device="0")
    image_name_list = os.listdir(path)
    imgs = []
    for image_name in image_name_list:
        print("processed img name: ", image_name)
        img = cv2.imread(path + image_name)
        imgs.append(img)
    out = model(imgs)
    print(out)
