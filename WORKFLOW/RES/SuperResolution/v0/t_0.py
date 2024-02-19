import os
import cv2
import numpy as np
from PIL import Image
import PIL


base_path = "/workspace/JuneLi/bbtv/SensedealImgAlg/DATASETS/RES/SuperResolution/v0/train/target/"
img_name_list = os.listdir(base_path)
for img_name in img_name_list:
    tar_img = Image.open(base_path + img_name)
    if tar_img.mode != "RGB":
        tar_img = tar_img.convert("RGB")
        print(tar_img.mode)
