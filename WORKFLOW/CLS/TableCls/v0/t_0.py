import os
import shutil
import cv2
import numpy as np
import random

base_path = "/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/CLS/TableCls/v0/train/1/"
image_name_list = os.listdir(base_path)
random.shuffle(image_name_list)
test_use_name = image_name_list[:500]
for name in test_use_name:
    shutil.move(base_path + name, base_path.replace("/train/", "/test/") + name)
