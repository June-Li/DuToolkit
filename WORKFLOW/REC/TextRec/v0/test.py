# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 10:49
# @Author  : lijun
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../../../../")
sys.path.append(os.path.abspath(root_dir))
sys.path.append(os.path.abspath(os.path.join(root_dir, "MODELALG/REC/CRNN/CRNNv0/")))

import cv2
import torch
import numpy as np
from tqdm import tqdm
from MODELALG.REC.CRNN.CRNNv0.networks import build_model
from MODELALG.REC.CRNN.CRNNv0.datasets.RecDataSet import RecDataProcess
from MODELALG.REC.CRNN.CRNNv0.utils import CTCLabelConverter
from MODELALG.REC.CRNN.CRNNv0.utils.torch_utils import select_device
from WORKFLOW.CLS.TextCls.v0.classifier_text import Classifier as Classifier_text


class Recognizer:
    def __init__(
        self,
        model_path,
        batch_size=16,
        gpu="0",
        alphabets_path=os.path.abspath(
            root_dir + "/MODELALG/REC/CRNN/CRNNv0/datasets/alphabets/ppocr_keys_v1.txt"
        ),
    ):
        self.gpu = gpu
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.model = build_model(cfg["model"])
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        self.model.load_state_dict(state_dict)

        self.device = select_device(self.gpu)
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
        self.converter = CTCLabelConverter(alphabets_path)
        self.batch_size = batch_size

    def inference(self, imgs):
        """
        该接口用来识别文本行；
        :param imgs: opencv读取格式图片列表
        :return:识别出来的text，例如：
                [[('text', [conf, conf, conf, conf])],
                 [('……', [……])]]
        """
        # 预处理根据训练来
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [
            self.process.normalize_img(self.process.resize_with_specific_height(img))
            for img in imgs
        ]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
            batch_imgs = [
                self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1])
                for idx in batch_idxs
            ]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0, 3, 1, 2])).float()
            tensor = tensor.to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        # 按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts


if __name__ == "__main__":
    # def longestCommonSubsequence(text1, text2):
    #     m, n = len(text1), len(text2)
    #     dp = [[0] * (n + 1) for _ in range(m + 1)]
    #
    #     for i in range(1, m + 1):
    #         for j in range(1, n + 1):
    #             if text1[i - 1] == text2[j - 1]:
    #                 dp[i][j] = dp[i - 1][j - 1] + 1
    #             else:
    #                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    #
    #     return dp[m][n]

    # path = '/workspace/JuneLi/bbtv/Datasets/public/ICDAR2015/images/'
    # model = Recognizer(os.path.abspath(root_dir + '/MODEL/REC/CRNN/CRNNv0/TextRec/20210601/best.pt'),
    #                    gpu='0')
    #
    # labels = open(path.replace('images/', '') + 'labels.txt', 'r').readlines()
    # label_dict = {}
    # for label in labels:
    #     label = label.strip('\n')
    #     key = label.split(', ')[0]
    #     word = label.split(', ')[-1][1:-1]
    #     label_dict[key] = word
    #
    # image_name_list = os.listdir(path)
    # acc_list = []
    # for index, image_name in enumerate(image_name_list):
    #     if image_name == 'coords.txt':
    #         continue
    #     img = cv2.imread(path + image_name)
    #     out = model.inference(img)
    #     if len(out[0][0][0]) > 0:
    #         # method 1:
    #         single_acc = longestCommonSubsequence(label_dict[image_name], out[0][0][0])/max(len(out[0][0][0]), len(label_dict[image_name]))
    #         acc_list.append(single_acc)
    #         print(single_acc)
    #
    #         # method 2：
    #         # if label_dict[image_name] == out[0][0][0]:
    #         #     acc_list.append(1)
    #         #     print('current acc: ', np.average(np.array(acc_list)))
    #         # else:
    #         #     acc_list.append(0)
    #         #     print('current acc: ', np.average(np.array(acc_list)))
    #     else:
    #         acc_list.append(0)
    #     print('processed img name: ', image_name, ' : ', out, '  label: ', label_dict[image_name])
    #     if index % 200 == 0:
    #         print('current acc: ', np.average(np.array(acc_list)))
    # print(np.average(np.array(acc_list)))

    def minDistance(word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        if n * m == 0:
            return n + m
        D = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)

        return D[n][m]

    device = "0"
    use_orientation_cls = True
    logs = open("./logs/PDFFake-logs.txt", "w")
    data_path = "/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/REC/TextRec/data/PDFFake/test.txt"
    model = Recognizer(
        os.path.abspath("/volume/weights/Recognizer_text_model.pt"), gpu=device
    )
    text_classifier = Classifier_text(
        "shufflenetv2",
        os.path.abspath("/volume/weights/Classifier_text_model.pt"),
        gpu=device,
    )
    labels = open(data_path, "r").readlines()
    min_edit_distance_list = []
    for idx, label in enumerate(tqdm(labels)):
        img_path = label.split("\t")[0]
        target_text = label.split("\t")[1]
        img = cv2.imread(img_path)
        if use_orientation_cls:
            h, w = np.shape(img)[:2]
            if h > 1.5 * w:
                img = np.rot90(img, 1)
            angle_list, score = text_classifier.inference([img])
            if angle_list[0] == 180 and score[0] > 0.7:
                img = np.rot90(img, 2)
        out = model.inference(img)
        min_edit_distance = minDistance(target_text, out[0][0][0])
        min_edit_distance_list.append(min_edit_distance)
        logs.write(
            img_path.split("/")[-1]
            + "\t"
            + target_text
            + "\t"
            + out[0][0][0]
            + "\t"
            + str(min_edit_distance)
            + "\n"
        )
    logs.write(
        "\n\n\naverage edit distance: "
        + str(np.average(np.array(min_edit_distance_list)))
    )
    print("average edit distance: ", np.average(np.array(min_edit_distance_list)))

    # img_name_list = os.listdir('/workspace/JuneLi/bbtv/Datasets/public/WenmuZhou_pytorch_ocr_use_dataset/dataset/CH/LSVT/recognition/train/')
    # for img_name in img_name_list:
    #     img = cv2.imread('/workspace/JuneLi/bbtv/Datasets/public/WenmuZhou_pytorch_ocr_use_dataset/dataset/CH/LSVT/recognition/train/' + img_name)
    #     h, w = np.shape(img)[:2]
    #     if h > 1.5*w:
    #         print(img_name)
