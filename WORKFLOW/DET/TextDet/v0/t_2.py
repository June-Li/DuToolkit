import os
import torch
import sys
import numpy as np


# baidu_ckpt = torch.load('/volume/weights/Detector_text_model.pt', map_location='cpu')
# baidu_state_dict = {}
# for k, v in baidu_ckpt['state_dict'].items():
#     baidu_state_dict[k.replace('module.', '')] = v
#
# my_ckpt = torch.load('output/DBNet/checkpoint/save/v3/best.pth', map_location='cpu')
# my_state_dict = {}
# for k, v in my_ckpt['state_dict'].items():
#     my_state_dict[k.replace('module.', '')] = v
#
# u = []
# b = []
# m = []
# for key in baidu_state_dict.keys():
#     if key in my_state_dict.keys():
#         u.append({key: baidu_state_dict[key]})
#     else:
#         b.append({key: baidu_state_dict[key]})
#
# for key in my_state_dict.keys():
#     if key in baidu_state_dict.keys():
#         pass
#     else:
#         m.append({key: my_state_dict[key]})
#
# # torch.save({'state_dict': baidu_state_dict}, 'output/DBNet/checkpoint/baidu_ckpt')
# # torch.save({'state_dict': my_state_dict}, 'output/DBNet/checkpoint/my_ckpt')
# # torch.save(my_ckpt['optimizer'], 'output/DBNet/checkpoint/my_ckpt')
#
# print()

"""***************"""
# my_ckpt = torch.load('output/DBNet/checkpoint/save/v3/best.pth', map_location='cpu')
# cfg = my_ckpt['cfg']
#
# my_state_dict = {}
# for k, v in my_ckpt['state_dict'].items():
#     my_state_dict[k.replace('module.', '')] = v
# torch.save({'state_dict': my_state_dict, 'cfg': cfg}, 'output/DBNet/checkpoint/my.pth')
#
# my_state_dict = {}
# for k, v in my_ckpt['state_dict'].items():
#     if 'num_batches_tracked' not in k:
#         my_state_dict[k.replace('module.', '')] = v
# torch.save({'state_dict': my_state_dict, 'cfg': cfg}, 'output/DBNet/checkpoint/my_slim.pth')
"""***************"""
# my_ckpt = torch.load('output/DBNet/checkpoint/save/v3/best.pth', map_location='cpu')
# cfg = my_ckpt['cfg']
#
# baidu_ckpt = torch.load('/volume/weights/Detector_text_model.pt', map_location='cpu')
# cfg = baidu_ckpt['cfg']
#
# baidu_state_dict = {}
# for k, v in baidu_ckpt['state_dict'].items():
#     v = torch.tensor(v, dtype=torch.float64)
#     baidu_state_dict[k.replace('module.', '')] = v
# torch.save({'state_dict': baidu_state_dict, 'cfg': cfg}, 'output/DBNet/checkpoint/baidu.pth')
"""***************"""
# my_ckpt = torch.load('output/DBNet/checkpoint/save/v3/best.pth', map_location='cpu')
# cfg = my_ckpt['cfg']

# baidu_ckpt = torch.load('output/DBNet/checkpoint/save/v3/best.pth', map_location='cpu')
# baidu_ckpt = torch.load('/volume/weights/Detector_text_model.pt', map_location='cpu')
# baidu_ckpt = torch.load('output/DBNet/checkpoint/baidu.pth', map_location='cpu')
# cfg = baidu_ckpt['cfg']
#
# avg_list = []
# baidu_state_dict = {}
# for k, v in baidu_ckpt['state_dict'].items():
#     avg = np.average(v.numpy())
#     max_ = np.max(v.numpy())
#     avg_list.append(avg)
#     # if True:  # avg > 10:
#     # if avg > 10:
#     #     print(k, ': ', avg)
#     if max_ > 100:
#         print(k, ': ', max_)
#     v = torch.tensor(v, dtype=torch.float64)
#     baidu_state_dict[k.replace('module.', '')] = v
# print(np.average(avg_list))
# """***************"""
# baidu_ckpt = torch.load('/volume/weights/Detector_text_model.pt', map_location='cpu')
# cfg = baidu_ckpt['cfg']
#
# baidu_state_dict = {}
# for k, v in baidu_ckpt['state_dict'].items():  # k == 'head.binarize.conv1.weight' or
#     if k == 'head.binarize.conv_bn1.running_mean':
#         v = torch.div(v, 1e2)
#     elif k == 'head.binarize.conv_bn1.running_var':
#         v = torch.div(v, 1e4)
#     baidu_state_dict[k.replace('module.', '')] = v
# torch.save({'state_dict': baidu_state_dict, 'cfg': cfg}, 'output/DBNet/checkpoint/baidu.pth')
"""***************"""
baidu_ckpt = torch.load("/volume/weights/Detector_text_model.pt", map_location="cpu")
cfg = baidu_ckpt["cfg"]

baidu_state_dict = {}
for k, v in baidu_ckpt["state_dict"].items():  # k == 'head.binarize.conv1.weight' or
    if k == "head.binarize.conv1.weight":
        v = torch.div(v, 1e2)
    elif k == "head.binarize.conv_bn1.running_mean":
        v = torch.div(v, 1e2)
    elif k == "head.binarize.conv_bn1.running_var":
        v = torch.div(v, 1e4)
    baidu_state_dict[k.replace("module.", "")] = v
torch.save(
    {"state_dict": baidu_state_dict, "cfg": cfg}, "output/DBNet/checkpoint/baidu.pth"
)
