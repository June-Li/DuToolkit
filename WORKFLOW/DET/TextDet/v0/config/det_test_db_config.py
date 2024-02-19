# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 11:11
# @Author  : lijun
from addict import Dict

config = Dict()
config.exp_name = "DBNet"
config.val_options = {
    "resume_from": "/volume/weights/Detector_text_model.pt",  # 继续训练地址
    "device": "cuda:0",  # 不建议修改，如果想可见某些GPU可以用：CUDA_VISIBLE_DEVICES=0,1,2,3，或者直接写在代码中：os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"，或者DataParallel(model, device_ids=[0, 1, 2])
}

config.model = {
    # backbone 可以设置'pretrained': False/True
    "type": "DetModel",
    "backbone": {
        "type": "ResNet",
        "layers": 18,
        "pretrained": True,
    },  # ResNet or MobileNetV3
    "neck": {"type": "DB_fpn", "out_channels": 256},
    "head": {"type": "DBHead"},
    "in_channels": 3,
}

config.post_process = {
    "type": "DBPostProcess",
    "thresh": 0.3,  # 二值化输出map的阈值
    "box_thresh": 0.7,  # 低于此阈值的box丢弃
    "unclip_ratio": 1.5,  # 扩大框的比例
}

# for dataset
# ##lable文件
### 存在问题，gt中str-->label 是放在loss中还是放在dataloader中
config.dataset = {
    "eval": {
        "dataset": {
            "type": "JsonDataset",
            "file": r"/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/DET/TextDet/data/PDFFake/test.json",
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
            "num_workers": 16,
            "collate_fn": {"type": "DetCollectFN"},
        },
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)
