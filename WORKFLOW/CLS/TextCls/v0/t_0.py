import os
import shutil
import cv2
import numpy as np


image_name_list = os.listdir("/home/bbtv/datasets/Chinese_dataset")
count = 0
for image_name in image_name_list:
    if not image_name.endswith(".jpg"):
        continue
    image = cv2.imread("/home/bbtv/datasets/Chinese_dataset/" + image_name)
    image = np.ascontiguousarray(np.rot90(image), dtype=np.uint8)
    image = np.ascontiguousarray(np.rot90(image), dtype=np.uint8)
    if count % 100 == 0:
        shutil.copy(
            "/home/bbtv/datasets/Chinese_dataset/" + image_name,
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/test/0/" + image_name,
        )
        cv2.imwrite(
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/test/1/" + image_name, image
        )
    elif count % 100 == 50:
        shutil.copy(
            "/home/bbtv/datasets/Chinese_dataset/" + image_name,
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/val/0/" + image_name,
        )
        cv2.imwrite(
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/val/1/" + image_name, image
        )
    else:
        shutil.copy(
            "/home/bbtv/datasets/Chinese_dataset/" + image_name,
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/train/0/" + image_name,
        )
        cv2.imwrite(
            "/home/bbtv/pytorch-cifar100/data/ocr_angle/train/1/" + image_name, image
        )
    count += 1
    if count % 100 == 0:
        print("processed num: ", count)
