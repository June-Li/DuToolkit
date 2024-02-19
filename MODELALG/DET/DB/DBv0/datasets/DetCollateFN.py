# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 15:45
# @Author  : lijun
import PIL
import copy
import numpy as np
import torch
from torchvision import transforms

__all__ = ["DetCollectFN"]


class DetCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                    if isinstance(v, np.ndarray):
                        v = self.unified_points(v)
                        v = torch.tensor(v)
                    if isinstance(v, PIL.Image.Image):
                        v = transforms.ToTensor()(v)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict

    def unified_points(self, points_array):
        out = []
        max_len = max([len(i) for i in points_array])
        for idx, points in enumerate(points_array):
            if isinstance(points, np.ndarray):
                points = points.tolist()
            points_len = len(points)
            if points_len < max_len:
                for _ in range(max_len - points_len):
                    points.append(copy.deepcopy(points[-1]))
            out.append(points)
        out = np.array(out, dtype=int)
        return out
