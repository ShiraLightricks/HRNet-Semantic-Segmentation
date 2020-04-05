# ------------------------------------------------------------------------------
# Copyright (c) Lightricks
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from .base_dataset import BaseDataset


class MHP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=10,
                 multi_scale=True,
                 flip=True,
                 ignore_label=0,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super().__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std, )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {}
        # self.label_mapping = {0: ignore_label,
        #                       1: ignore_label, 2: ignore_label,
        #                       3: ignore_label, 4: ignore_label,
        #                       5: ignore_label, 6: ignore_label,
        #                       7: 0, 8: 1, 9: ignore_label,
        #                       10: ignore_label, 11: 2, 12: 3,
        #                       13: 4, 14: ignore_label, 15: ignore_label,
        #                       16: ignore_label, 17: 5, 18: ignore_label,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        #                       25: 12, 26: 13, 27: 14, 28: 15,
        #                       29: ignore_label, 30: ignore_label,
        #                       31: 16, 32: 17, 33: 18}

    def read_files(self):
        """
        read list and creates files list with dict objects that contain
        path for images
        path for annotation  (if this is not test)
        name of image
        wight of image = 1
        :return: files list
        """
        files = []
        # if this is test folder then there are no labels
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        """
        convert label according to label_mapping
        :param label: label image
        :param inverse: use inverse label_mapping
        :return:
        """
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        size = image.shape

        # if there is no annotation (test):
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name