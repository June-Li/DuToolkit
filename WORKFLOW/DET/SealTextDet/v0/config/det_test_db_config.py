# -*- coding: utf-8 -*-
# @Time    : 2020/06/15 17:22
# @Author  : lijun

from addict import Dict

config = Dict()
config.exp_name = "DBNet"
config.test_options = {
    # 'resume_from': '/workspace/JuneLi/bbtv/SensedealImgAlg/MODEL/DET/DB/DBv0/TextDet/20210601/best.pt',  # 继续训练地址
    "resume_from": "/workspace/JuneLi/bbtv/SensedealImgAlg/WORKFLOW/DET/TextDet/v0/output/DBNet/checkpoint/best.pth",  # 继续训练地址
    # 'resume_from': '/workspace/JuneLi/bbtv/SensedealImgAlg/WORKFLOW/DET/TextDet/v0/weights/det_r50_vd_db/best_accuracy',  # 继续训练地址
    # 'resume_from': '/workspace/JuneLi/bbtv/ExpCode/PaddleOCR-1.0-2021/inference/ch_ppocr_server_v1.1_det_infer/',
    # 'third_party_name': 'paddle',
    "device": "0,1",  # gpu的索引号，或者cpu
}

config.model = {
    # backbone 可以设置'pretrained': False/True
    "type": "DetModel",
    "backbone": {
        "type": "ResNet",
        "layers": 18,
        "pretrained": False,
    },  # ResNet or MobileNetV3
    "neck": {"type": "DB_fpn", "out_channels": 256},
    "head": {"type": "DBHead"},
    "in_channels": 3,
}

config.post_process = {
    "type": "DBPostProcessV1",
    "thresh": 0.3,  # 二值化输出map的阈值
    "box_thresh": 0.7,  # 低于此阈值的box丢弃
    "unclip_ratio": 1.5,  # 扩大框的比例
}

# for dataset
# lable文件
config.dataset = {
    "eval": {
        "dataset": {
            "type": "JsonDataset",
            "file": r"/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/DET/TextDet/icdar2015/test.json",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "pre_processes": [
                {
                    "type": "ResizeShortSize",
                    "args": {"short_size": 736, "resize_text_polys": False},
                }
            ],
            "filter_keys": [],  # 需要从data_dict里过滤掉的key
            "ignore_tags": ["*", "###"],
            "img_mode": "RGB",
        },
        "loader": {
            "type": "DataLoader",
            "batch_size": 1,  # 必须为1
            "shuffle": False,
            "num_workers": 1,
            "collate_fn": {"type": "DetCollectFN"},
        },
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)
