# -*- coding: utf-8 -*-
# @Time    : 2021/6/4 17:27
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(
    os.path.abspath(os.path.join(root_dir, "MODELALG/CLS/ClsCollect/ClsCollectv0/"))
)

import numpy as np
import torch
import torchvision.transforms as transforms
# import threading

from MODELALG.CLS.ClsCollect.ClsCollectv0.utils import get_network
# from MODELALG.CLS.ClsCollect.ClsCollectv0.torch_utils import select_device
import cv2
import time
import traceback
from MODELALG.utils.common import Log, select_device


logger = Log(__name__).get_logger()
# lock = threading.Lock()


class cfg_store:
    def __init__(self):
        self.net = "mobilenetv2"
        self.device = "0"


class Classifier:
    def __init__(
        self, net, model_path, batch_size=16, device="cuda:0", half_flag=False
    ):
        self.args = cfg_store()
        self.args.net = net
        self.args.device = device
        self.half_flag = half_flag

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.device = select_device(self.device)
        self.net = get_network(self.args).to(self.device)
        self.net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.net.eval()
        if self.half_flag and self.device.type != "cpu":
            self.net.half()
        self.normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        )
        self.preprocess = transforms.Compose([transforms.ToTensor(), self.normalize])
        logger.info(" ···-> load model succeeded!")

    def default_loader(self, image, preprocess):
        h, w = np.shape(image)[0], np.shape(image)[1]
        image = cv2.resize(image, (32 * w // h, 32))
        h, w = np.shape(image)[0], np.shape(image)[1]
        if w >= 1280:
            image = cv2.resize(image, (1280, 32))
        else:
            mask_img = np.ones((32, 1280, 3), dtype=np.uint8) * int(
                np.argmax(np.bincount(image.flatten(order="C")))
            )
            mask_img[:, (1280 - w) // 2 : (1280 - w) // 2 + w] = image
            image = mask_img
        img_tensor = preprocess(image)
        return img_tensor

    def __call__(self, imgs_ori):
        """
        该接口用来分类文本行的方向，0°和180°
        :param imgs_ori: opencv读取格式图片列表
        :return: 对文本条分类出来的列表，例如：
                [180, 180, 0, 180, ……]
        """
        try:
            # with lock:
            # totaltime = time.time()
            if self.half_flag and self.device.type != "cpu":
                imgs = [
                    self.default_loader(img, self.preprocess).half().to(self.device)
                    for img in imgs_ori
                ]
            else:
                imgs = [
                    self.default_loader(img, self.preprocess).to(self.device)
                    for img in imgs_ori
                ]
            idxs = list(range(len(imgs)))
            cls_out = []
            score = []
            for idx in range(0, len(imgs), self.batch_size):
                # starttime = time.time()
                batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
                batch_imgs = [imgs[idx_] for idx_ in batch_idxs]
                batch_imgs = torch.stack(batch_imgs)
                # batch_imgs = torch.from_numpy(batch_imgs)
                # batch_imgs = batch_imgs.to(self.device)
                # logger.info(
                #     "*" * 20
                #     + "数据准备耗时: {}".format(str(round(time.time() - starttime, 5)))
                # )
                with torch.no_grad():
                    # starttime = time.time()
                    output = self.net(batch_imgs)
                    # logger.info(
                    #     "*" * 20
                    #     + "模型推理耗时: {}".format(str(round(time.time() - starttime, 5)))
                    # )
                    # starttime = time.time()
                    output = output.detach().cpu()
                    # logger.info(
                    #     "*" * 20
                    #     + "to cpu耗时: {}".format(str(round(time.time() - starttime, 5)))
                    # )
                    # starttime = time.time()
                    _, pred = output.topk(1, 1, largest=True, sorted=True)
                    cls_out.extend(np.squeeze(pred.cpu().numpy(), axis=-1))
                    output_softmax = (
                        torch.nn.functional.softmax(output, dim=-1).cpu().numpy()
                    )
                    index = np.argmax(output_softmax, axis=-1)
                    score.extend(
                        [output_softmax[index_][i] for index_, i in enumerate(index)]
                    )
                    # logger.info(
                    #     "*" * 20
                    #     + "后处理耗时: {}".format(str(round(time.time() - starttime, 5)))
                    # )
            # logger.info(
            #     "*" * 20 + "文本分类总耗时: {}".format(str(round(time.time() - totaltime, 5)))
            # )
            return list(np.multiply(np.array(cls_out), 180)), score
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    classifier_ = Classifier(
        "shufflenetv2",
        os.path.abspath(
            root_dir + "/MODEL/CLS/ClsCollect/ClsCollectv0/TextCls/20210607/best.pt"
        ),
        device="0",
    )
    path = cur_dir + "/test_data/my_imgs_0/1/"
    image_name_list = os.listdir(path)
    images = []
    for image_name in image_name_list:
        image = cv2.imread(path + image_name)
        images.append(image)
    pred_, score_ = classifier_(images)
    print("predict output: ", pred_, "\n", score_)
