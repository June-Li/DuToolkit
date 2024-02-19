# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 14:52
# @Author  : lijun

from addict import Dict

config = Dict()
config.exp_name = "DBNet"
config.train_options = {
    # for train
    # 'resume_from': '/volume/weights/Detector_text_model.pt'
    "resume_from": "/workspace/JuneLi/bbtv/SensedealImgAlg/WORKFLOW/DET/TextDet/v0/output/DBNet/checkpoint/save/v2/119.pth",  # 继续训练地址
    # 'resume_from': False,
    "third_party_name": "",  # 加载paddle模型可选
    "checkpoint_save_dir": f"./output/{config.exp_name}/checkpoint",  # 模型保存地址，log文件也保存在这里
    "device": "cuda:0",  # 不建议修改，如果想可见某些GPU可以用：CUDA_VISIBLE_DEVICES=0,1,2,3，或者直接写在代码中：os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"，或者DataParallel(model, device_ids=[0, 1, 2])
    "epochs": 120,
    "fine_tune_stage": ["backbone", "neck", "head"],  # ['backbone', 'neck', 'head'],
    "print_interval": 1,  # step为单位
    "val_interval": 5,  # epoch为单位
    "ckpt_save_type": "HighestAcc",  # HighestAcc：只保存最高准确率模型 ；FixedEpochStep：每隔ckpt_save_epoch个epoch保存一个
    "ckpt_save_epoch": 1,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

config.SEED = 927
config.optimizer = {
    "type": "Adam",
    "lr": 0.001,
    "weight_decay": 1e-4,
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

config.loss = {"type": "DBLoss", "alpha": 1, "beta": 10}

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
    "train": {
        "dataset": {
            "type": "JsonDataset",
            "file": r"/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/DET/TextDet/data/total/train.json",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            # db 预处理，不需要修改
            "pre_processes": [
                {
                    "type": "IaaAugment",
                    "args": [
                        {"type": "Fliplr", "args": {"p": 0.5}},
                        {"type": "Affine", "args": {"rotate": [-10, 10]}},
                        {"type": "Resize", "args": {"size": [0.5, 3]}},
                    ],
                },
                {
                    "type": "EastRandomCropData",
                    "args": {"size": [640, 640], "max_tries": 50, "keep_ratio": True},
                },
                {
                    "type": "MakeBorderMap",
                    "args": {"shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7},
                },
                {
                    "type": "MakeShrinkMap",
                    "args": {"shrink_ratio": 0.4, "min_text_size": 8},
                },
            ],
            "filter_keys": [
                "img_path",
                "img_name",
                "text_polys",
                "texts",
                "ignore_tags",
                "shape",
            ],  # 需要从data_dict里过滤掉的key
            "ignore_tags": ["*", "###"],
            "img_mode": "RGB",
        },
        "loader": {
            "type": "DataLoader",  # 使用torch dataloader只需要改为 DataLoader
            "batch_size": 36,
            "shuffle": True,
            "num_workers": 16,
            "collate_fn": {"type": ""},
        },
    },
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
            "batch_size": 1,  # 必须为1，因为测试的时候不像训练时候可以裁剪640*640，所以每张照片大小不一是不能放在同一batch，如果resize又会导致结果不准
            "shuffle": False,
            "num_workers": 8,
            "collate_fn": {"type": "DetCollectFN"},
        },
    },
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)
