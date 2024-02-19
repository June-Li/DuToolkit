from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random
import torch
import cv2


def img_label_path(root_dir=os.path.abspath("./data/ocr_angle/train/")):
    cls_name_list = os.listdir(root_dir)
    total_img_path_list = []
    for cls_name in cls_name_list:
        image_name_path = os.path.join(root_dir, cls_name)
        image_name_list = os.listdir(image_name_path)
        for image_name in image_name_list:
            total_img_path_list.append(os.path.join(image_name_path, image_name))
    random.shuffle(total_img_path_list)
    file_list = []
    number_list = []
    for total_img_path in total_img_path_list:
        file_list.append(total_img_path)
        number_list.append(total_img_path.split("/")[-2])
    return file_list, number_list


def default_loader(image, preprocess, img_size):
    image = cv2.resize(image, (img_size, img_size))
    img_tensor = preprocess(image)
    return img_tensor


class dt_set(Dataset):
    def __init__(self, loader, preprocess, file_list, number_list, img_size):
        # 定义好 image 的路径
        self.preprocess = preprocess
        self.images = file_list
        self.target = number_list
        self.loader = loader
        self.img_size = img_size

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(cv2.imread(fn), self.preprocess, self.img_size)
        target = torch.tensor(int(self.target[index]), dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.images)


def get_dataloader(root_dir, shuffle, num_workers, batch_size, img_size):
    file_list, number_list = img_label_path(root_dir)
    normalize = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
    )

    preprocess = transforms.Compose(
        [
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data = dt_set(
        loader=default_loader,
        preprocess=preprocess,
        file_list=file_list,
        number_list=number_list,
        img_size=img_size,
    )
    return DataLoader(
        data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )


if __name__ == "__main__":
    get_dataloader("./data/ocr_angle/train/", True, 4, 16)
    print("load sucesses")
