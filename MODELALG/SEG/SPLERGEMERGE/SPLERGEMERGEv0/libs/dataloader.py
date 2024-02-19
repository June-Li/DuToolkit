import os
import pickle
import torch
import numpy as np
import cv2
from termcolor import cprint
from torchvision import transforms

from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.utils import resize_image
from MODELALG.SEG.SPLERGESPLIT.SPLERGESPLITv0.libs.utils import normalize_numpy_image


class SplitTableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train_images_path,
        train_labels_path,
        transforms=None,
        fix_resize=False,
    ):
        self.fix_resize = fix_resize
        self.root = root
        self.transforms = transforms
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path

        cprint(self.root, "yellow")
        cprint(self.train_images_path, "yellow")
        cprint(self.train_labels_path, "yellow")

        self.img_paths = list(
            sorted(os.listdir(os.path.join(self.root, self.train_images_path)))
        )

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.train_images_path, self.img_paths[idx])
        img_name = img_path.split("/")[-1][:-4]

        row_label_path = os.path.join(
            self.root, self.train_labels_path, img_name + "_row.txt"
        )
        col_label_path = os.path.join(
            self.root, self.train_labels_path, img_name + "_col.txt"
        )

        image = cv2.imread(img_path)
        image = image.astype("float32")

        if image.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            image = image[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))

        C, H, W = image.shape
        image = resize_image(image, fix_resize=self.fix_resize)
        image = normalize_numpy_image(image)

        image = image.numpy()

        if not (os.path.isfile(row_label_path) and os.path.isfile(col_label_path)):
            cprint("[error] label file not found", "red", attrs=["bold"])
            exit(0)

        _, o_H, o_W = image.shape
        scale = o_H / H

        with open(row_label_path, "r") as f:
            row_label = f.read().split("\n")

        with open(col_label_path, "r") as f:
            col_label = f.read().split("\n")

        row_label = np.array([[int(x) for x in row_label if x != ""]])
        col_label = np.array([[int(x) for x in col_label if x != ""]])

        row_target = cv2.resize(row_label, (o_H, 1), interpolation=cv2.INTER_NEAREST)
        col_target = cv2.resize(col_label, (o_W, 1), interpolation=cv2.INTER_NEAREST)

        row_target[row_target == 255] = 1
        col_target[col_target == 255] = 1

        row_target = torch.tensor(row_target[0])
        col_target = torch.tensor(col_target[0])

        target = [row_target, col_target]

        image = image.transpose((1, 2, 0))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, img_path, W, H

    def __len__(self):
        return len(self.img_paths)


class MergeTableDataset(torch.utils.data.Dataset):
    def __init__(self, root, train_img_path, train_seg_path, train_labels_path):
        self.root = root
        self.train_img_path = train_img_path
        self.train_seg_path = train_seg_path
        self.train_labels_path = train_labels_path
        self.img_paths_list = list(
            sorted(os.listdir(os.path.join(self.root, self.train_img_path)))
        )

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root, self.train_img_path, self.img_paths_list[idx]
        )
        seg_path = os.path.join(
            self.root,
            self.train_seg_path,
            self.img_paths_list[idx].replace(".png", ".pkl").replace(".jpg", ".pkl"),
        )
        target_path = os.path.join(
            self.root,
            self.train_labels_path,
            self.img_paths_list[idx].replace(".png", ".pkl").replace(".jpg", ".pkl"),
        )

        img, seg = self.resize(
            cv2.imread(img_path, 0), pickle.load(open(seg_path, "rb"))
        )
        mask_row, mask_col, mask_gird = self.make_binary_img(img, seg)

        img = self.transform(img)
        input_feature = np.concatenate(
            [img, np.stack([mask_row, mask_col, mask_gird], axis=0)], axis=0
        )

        target = pickle.load(open(target_path, "rb"))
        return input_feature, target, self.img_paths_list[idx]

    def __len__(self):
        return len(self.img_paths_list)

    @staticmethod
    def transform(image):
        transformer = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485], std=[0.229]
                ),  # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        image = transformer(image)
        return image

    @staticmethod
    def resize(image, seg, min_size=600, max_size=1024):
        H, W = image.shape[:2]
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        image = cv2.resize(image, (int(W * scale), int(H * scale)))
        seg[0] = np.array(np.array(seg[0]) * scale, dtype=int)
        seg[1] = np.array(np.array(seg[1]) * scale, dtype=int)
        return image, seg

    @staticmethod
    def make_binary_img(img, seg):
        mask_row = np.zeros(img.shape[:2], dtype=int)
        mask_col = np.zeros(img.shape[:2], dtype=int)
        mask_grid = np.zeros(img.shape[:2], dtype=int)
        for r in seg[0]:
            mask_row[r - 3 : r + 4, :] = 1
            mask_grid[r - 3 : r + 4, :] = 1
        for c in seg[1]:
            mask_col[:, c - 3 : c + 4] = 1
            mask_grid[:, c - 3 : c + 4] = 1
        return mask_row, mask_col, mask_grid

    @staticmethod
    def make_mask_img(ori_img, boxes):
        h, w = np.shape(ori_img)[:2]
        img = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            if box[2] - box[0] > 25:
                x_0, x_1 = int(box[0] + 8), int(box[2] - 8)
            else:
                continue
            if box[3] - box[1] > 35:
                y_0, y_1 = int(box[1] + 8), int(box[3] - 8)
            elif box[3] - box[1] > 30:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 7), int(
                    (box[1] + box[3]) // 2 + 7
                )
            elif box[3] - box[1] > 25:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 6), int(
                    (box[1] + box[3]) // 2 + 6
                )
            elif box[3] - box[1] > 20:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 5), int(
                    (box[1] + box[3]) // 2 + 5
                )
            elif box[3] - box[1] > 15:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 4), int(
                    (box[1] + box[3]) // 2 + 4
                )
            elif box[3] - box[1] > 10:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 3), int(
                    (box[1] + box[3]) // 2 + 3
                )
            else:
                y_0, y_1 = int((box[1] + box[3]) // 2 - 2), int(
                    (box[1] + box[3]) // 2 + 2
                )
            box = [x_0, y_0, x_1, y_1]
            img[box[1] : box[3], box[0] : box[2]] = 255
        return img
