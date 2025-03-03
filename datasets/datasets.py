import os
from typing import Union, Tuple
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from PIL import Image
import numpy as np
import megfile as mf
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

prepare_train_data = A.Compose([
    # 1. 空间级增强
    A.RandomCrop(width=self.patch_size_train, height=self.patch_size_train, p=1.0),
    A.D4(p=1.0),
    A.Perspective(scale=[0.05, 0.1], p=0.4),
    # 2. 亮度和对比度增强
    A.OneOf([
        # 光照效果
        A.Illumination(mode="linear", intensity_range=[0.05, 0.2], effect_type="both", angle_range=[0, 360], p=1.0),
        A.Illumination(mode="corner", intensity_range=[0.05, 0.2], effect_type="both", p=1.0),
        A.Illumination(mode="gaussian", intensity_range=[0.05, 0.2], effect_type="both", center_range=[0.2, 0.8], sigma_range=[0.2, 0.8], p=1.0)


    ], p=0.6),
    # 3. 颜色和纹理增强
    A.OneOf([

    ]),
    # 4. 归一化，转tensor
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2(),
])

"""
数据增强：
空间级增强
    1. 随机裁剪 https://explore.albumentations.ai/transform/RandomCrop
    1. 随机八种变换 https://explore.albumentations.ai/transform/D4
    2. 透视变换 https://explore.albumentations.ai/transform/Perspective
    x. 形态学操作 https://explore.albumentations.ai/transform/Morphological

亮度和对比度增强：CLAHE，RandomBrightnessContrast，Illumination，RandomShadow  
    1. 照明效果扰动 https://explore.albumentations.ai/transform/Illumination
    2. 亮度和对比度 https://explore.albumentations.ai/transform/RandomBrightnessContrast
    3. 自适应直方图均衡 https://explore.albumentations.ai/transform/CLAHE
    4. 随机阴影 https://explore.albumentations.ai/transform/RandomShadow

颜色和纹理增强；FancyPCA，Emboss，ChromaticAberration
    1. 随机色彩变化 https://explore.albumentations.ai/transform/FancyPCA
    2. 浮雕压花 https://explore.albumentations.ai/transform/Emboss
    3. 红蓝色差 https://explore.albumentations.ai/transform/ChromaticAberration
"""

class NTIRE_LIE_Dataset(Dataset):
    """
    文件组织格式：
    Train & Val:
        - Input
            - image1
            - image2
            - ...
        - GT
            - image1
            - image2
            - ...
    Test:
        - image1
        - image2
        - ...
    """
    def __init__(
            self,
            data_root: str,
            subfolder: str,
            patch_size_tv: Union[int, Tuple[int, int]] = 256,
    ):
        """
        Args:
            data_root: 数据集根目录 /path/to/NTIRE202x
            patch_size_tv: 训练train和验证val时的裁剪尺寸，为int时训练验证使用的尺寸一致，为tuple时分别使用不同尺寸，测试时不使用该参数
            subfolder: 子集，[Train, Val, Test]
                - Train: 进行数据增强
                - Val:   进行中心裁剪
                - Test:  不进行操作
        """
        super().__init__()
        self.data_root = data_root
        assert subfolder in {"Train", "Val", "Test"}, \
            f"输入值必须是 'Train', 'Val', 'Test' 中的一个，但收到了: {subfolder}"
        self.subfolder = subfolder
        self.image_name_list = self.set_name_list()

        self.patch_size_train, self.patch_size_val = patch_size_tv if isinstance(patch_size_tv, tuple) else (patch_size_tv, patch_size_tv)


        self.prepare_train_data = A.Compose([

        ])
        self.prepare_val_data = A.Compose([
            A.CenterCrop(height=self.prepare_val_data, width=self.patch_size_val),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
        self.prepare_test_data = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def set_name_list(self):
        if self.subfolder == "Test":
            return mf.smart_listdir(mf.smart_path_join(self.data_root, self.subfolder))
        elif self.subfolder in ["Train", "Val"]:
            gt_image_name_list = mf.smart_listdir(mf.smart_path_join(self.data_root, self.subfolder, "GT"))
            in_image_name_list = mf.smart_listdir(mf.smart_path_join(self.data_root, self.subfolder, "Input"))
            return list(set(gt_image_name_list) & set(in_image_name_list))

        self.data_augmentation = A.Compose([
            A.RandomCrop(width=self.patch_size_train, height=self.patch_size_train, p=1.0),
            A.RandomRotate90(p=0.6),
            A.RandomResizedCrop
        ])

        # transform = A.Compose([
        #     A.RandomCrop(width=crop_size, height=crop_size, always_apply=True),
        #     A.RandomRotate90(always_apply=True),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        # ], additional_targets={'image2': 'image'})  # Bind the second image

    @staticmethod
    def load_image(image_path):
        image_np = np.array(Image.open(mf.smart_open(image_path, 'rb')))


    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.subfolder == "Test":
            pass


def image_info(image):
    print("=-"*20)
    print("type:", type(image))
    print("shape:", image.shape)
    print("dtype:", image.dtype)



if __name__ == "__main__":
    # data_root = r"D:\dataset\NTIRE_2025"
    # gt_root = mf.smart_path_join(data_root, "Train", "GT")
    # input_root = mf.smart_path_join(data_root, "Train", "Input")
    # print("gt_root: ", gt_root)
    #
    # print("是否存在：", mf.smart_exists(data_root))
    # print(gt_root)
    #
    # list_gt_root = mf.smart_listdir(gt_root)
    # list_input_root = mf.smart_listdir(input_root)
    # print("list gt root: ", list_gt_root)
    # print("len of list: ", len(list_gt_root))
    # print(list_input_root == list_gt_root)
    #
    # image_name_list = list_gt_root and list_input_root
    # print(image_name_list)
    # print("len of image_name_list: ", len(image_name_list))

    image_path = r"D:\dataset\NTIRE_2025\Test\141.png"
    image_np = np.array(Image.open(mf.smart_open(image_path, 'rb')))
    image_info(image_np)






























