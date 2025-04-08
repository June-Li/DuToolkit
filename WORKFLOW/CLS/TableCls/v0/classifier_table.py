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

import time
import traceback
import numpy as np
import torch
import torchvision.transforms as transforms
# import threading

from MODELALG.CLS.ClsCollect.ClsCollectv0.utils import get_network
# from MODELALG.CLS.ClsCollect.ClsCollectv0.torch_utils import select_device
from WORKFLOW.CLS.TableCls.v0.utils.make_dataloader import default_loader
import cv2
from MODELALG.utils.common import Log, select_device


logger = Log(__name__).get_logger()
# lock = threading.Lock()


class cfg_store:
    def __init__(self):
        self.net = "mobilenetv2"
        self.device = "0"


class Classifier:
    def __init__(
        self,
        net,
        model_path,
        batch_size=4,
        device="cuda:0",
        img_size=512,
        half_flag=False,
    ):
        self.args = cfg_store()
        self.args.net = net
        self.args.device = device
        self.half_flag = half_flag

        self.img_size = img_size
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.device = select_device(self.device)
        self.net = get_network(self.args).to(self.device)
        # self.net = torch.nn.DataParallel(self.net)
        # self.net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        state_dict = torch.load(self.model_path, map_location="cpu")
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.net.load_state_dict(new_state_dict)
        self.net.to(self.device).float()
        self.net.eval()
        if self.half_flag and self.device.type != "cpu":
            self.net.half()
        self.normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        )
        self.preprocess = transforms.Compose([transforms.ToTensor(), self.normalize])
        logger.info(" ···-> load model succeeded!")

    def __call__(self, imgs_ori):
        """
        该接口用来分类表格是有线还是无线，0-有线和1-无线
        :param imgs_ori: opencv读取格式图片列表
        :return: 对文本条分类出来的列表，例如：
                [1, 1, 0, 1, ……]
                [0.999, 1.0, 0.999, 0.841, ……]
        """
        try:
            # with lock:
            if self.half_flag and self.device.type != "cpu":
                imgs = [
                    default_loader(img, self.preprocess, self.img_size).half()
                    for img in imgs_ori
                ]
            else:
                imgs = [
                    default_loader(img, self.preprocess, self.img_size)
                    for img in imgs_ori
                ]
            idxs = list(range(len(imgs)))
            cls_out = []
            score = []
            for idx in range(0, len(imgs), self.batch_size):
                batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
                batch_imgs = [imgs[idx_] for idx_ in batch_idxs]
                batch_imgs = np.stack(batch_imgs)
                batch_imgs = torch.from_numpy(batch_imgs)
                batch_imgs = batch_imgs.to(self.device)
                with torch.no_grad():
                    # start_time = time.time()
                    output = self.net(batch_imgs)
                    # print('pre img use time: ', time.time()-start_time)
                    _, pred = output.topk(1, 1, largest=True, sorted=True)
                    cls_out.extend(np.squeeze(pred.cpu().numpy(), axis=-1))

                    output_softmax = (
                        torch.nn.functional.softmax(output, dim=-1).cpu().numpy()
                    )
                    index = np.argmax(output_softmax, axis=-1)
                    score.extend(
                        [output_softmax[index_][i] for index_, i in enumerate(index)]
                    )

            return list(np.multiply(np.array(cls_out), 1)), score
        except Exception as e:
            logger.error(" ···-> inference faild!!!")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # classifier_ = Classifier('shufflenetv2',
    #                          os.path.abspath(root_dir + '/MODEL/CLS/ClsCollect/ClsCollectv0/TextCls/20210607/best.pt'),
    #                          gpu='0')
    classifier_ = Classifier(
        "shufflenetv2",
        # cur_dir + '/checkpoint/shufflenetv2/Wednesday_30_June_2021_18h_22m_07s/shufflenetv2-150-regular.pth',
        cur_dir
        + "/checkpoint/shufflenetv2/Wednesday_30_June_2021_18h_22m_07s/shufflenetv2-26-best.pth",
        device="0",
        batch_size=64,
        img_size=512,
    )
    path = cur_dir + "/test_data/my_imgs_3/"
    out_path = path.replace("/test_data/", "/test_out/")
    if not os.path.exists(out_path + "/line/"):
        os.makedirs(out_path + "/line/")
    if not os.path.exists(out_path + "/noline/"):
        os.makedirs(out_path + "/noline/")
    image_name_list = os.listdir(path)
    images = []
    for image_name in image_name_list:
        image = cv2.imread(path + image_name)
        images.append(image)
    pred_, score_ = classifier_(images)
    for index, i in enumerate(pred_):
        out_img = images[index].copy()
        if i == 0 and score_[index] >= 0.9:
            cv2.putText(
                out_img,
                "line: " + str(score_[index]),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
            )
            cv2.imwrite(out_path + "/line/" + image_name_list[index], out_img)
        else:
            cv2.putText(
                out_img,
                "noline: " + str(score_[index]),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
            )
            cv2.imwrite(out_path + "/noline/" + image_name_list[index], out_img)
    print("predict output: ", pred_, "\n", score_)
