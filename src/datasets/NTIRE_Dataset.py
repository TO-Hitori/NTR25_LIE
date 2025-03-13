import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from PIL import Image
import numpy as np
import megfile as mf
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


"""
数据增强：
空间级增强
    1. 随机裁剪 https://explore.albumentations.ai/transform/RandomCrop
    2. 随机八种变换 https://explore.albumentations.ai/transform/D4
    3. 透视变换 https://explore.albumentations.ai/transform/Perspective
    x. 形态学操作 https://explore.albumentations.ai/transform/Morphological

亮度和对比度增强：CLAHE，RandomBrightnessContrast，Illumination，RandomShadow  
    1. 照明效果扰动 https://explore.albumentations.ai/transform/Illumination
    2. 亮度和对比度 https://explore.albumentations.ai/transform/RandomBrightnessContrast
    3. 随机阴影 https://explore.albumentations.ai/transform/RandomShadow

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
            patch_size: int = 512,
    ):
        """
        Args:
            data_root: 数据集根目录 /path/to/NTIRE202x
            patch_size_tv: 训练train和验证val时的裁剪尺寸，为int时训练验证使用的尺寸一致，为tuple时分别使用不同尺寸，测试时不使用该参数
            subfolder: 子集，[Train, Val, Test]
                - Train: 进行数据增强: input和gt进行相同的空间级操作，对input进行像素级扰动
                - Val:   进行中心裁剪
                - Test:  不进行操作
        """
        super().__init__()
        self.data_root = data_root
        assert subfolder in {"Train", "Val", "Test"}, \
            f"输入值必须是 'Train', 'Val', 'Test' 中的一个，但收到了: {subfolder}"
        self.subfolder = subfolder
        self.image_name_list = self.set_name_list()

        self.patch_size = patch_size

        self.prepare_val_data = A.Compose([
            A.CenterCrop(height=self.patch_size, width=self.patch_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'image1': 'image'})

        self.numpy2tensor = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ])

        self.set_preprocessing_config(patch_size=patch_size)

    def set_preprocessing_config(
            self,
            patch_size,
            p_Perspective=0.1,
            p_l1=0.2,
            p_l2=0.2,
            p_l3=0.2,
            p_rbc=0.6,
            p_of_l=0.6,
            p_shadow=0.8,
            p_fec=0.2

    ):
        """
        patch_size
        p_Perspective
        p_l1
        p_l2
        p_l3
        p_rbc
        p_of_l
        p_shadow
        p_fec
        Returns:

        """
        self.patch_size = patch_size
        self.prepare_train_both = A.Compose([
            # 1. 空间级增强
            A.RandomCrop(width=patch_size, height=patch_size, p=1.0),
            A.D4(p=1.0),
            A.Perspective(scale=[0.05, 0.1], p=p_Perspective),
        ], additional_targets={'image1': 'image'})

        self.prepare_train_input_only = A.Compose([
            A.OneOf([
                # 特定方向光照效果
                A.Illumination(mode="linear", intensity_range=[0.05, 0.2], effect_type="both", angle_range=[0, 360],
                               p=p_l1),
                A.Illumination(mode="corner", intensity_range=[0.05, 0.2], effect_type="both", p=p_l2),
                A.Illumination(mode="gaussian", intensity_range=[0.05, 0.2], effect_type="both",
                               center_range=[0.2, 0.8], sigma_range=[0.2, 0.8], p=p_l3),
                # 随机调整图像亮度和对比度
                A.RandomBrightnessContrast(brightness_limit=[-0.3, 0.1], contrast_limit=[-0.3, 0.1],
                                           brightness_by_max=False, ensure_safe_range=True, p=p_rbc)
            ], p=p_of_l),
            # 随机阴影
            A.RandomShadow(shadow_roi=[0, 0, 1, 1], num_shadows_limit=[1, 3], shadow_dimension=3,
                           shadow_intensity_range=[0.4, 0.6], p=p_shadow),

            # 3. 颜色和纹理增强
            A.OneOf([
                A.OneOf([
                    A.FancyPCA(alpha=0.5, p=0.5),
                    A.Emboss(alpha=[0.2, 0.5], strength=[0.2, 0.5], p=0.5),
                    A.ChromaticAberration(p=0.5)
                ], p=p_fec)
            ]),
            # 4. 归一化，转tensor
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def set_name_list(self):
        if self.subfolder == "Test":
            return mf.smart_listdir(mf.smart_path_join(self.data_root, "Test"))
        elif self.subfolder in ["Train", "Val"]:
            gt_image_name_list = mf.smart_listdir(mf.smart_path_join(self.data_root, self.subfolder, "GT"))
            in_image_name_list = mf.smart_listdir(mf.smart_path_join(self.data_root, self.subfolder, "Input"))
            return list(set(gt_image_name_list) & set(in_image_name_list))

    @staticmethod
    def load_image(image_path):
        image_np = np.array(Image.open(mf.smart_open(image_path, 'rb')))
        return image_np

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.subfolder == "Test":
            image_in_np = self.load_image(mf.smart_path_join(self.data_root, "Test", self.image_name_list[idx]))
            image_in_ts = self.numpy2tensor(image=image_in_np)["image"]
            return {
                'image_in': image_in_ts,
            }
        elif self.subfolder == "Val":
            image_in_np = self.load_image(mf.smart_path_join(self.data_root, "Val", "Input", self.image_name_list[idx]))
            image_gt_np = self.load_image(mf.smart_path_join(self.data_root, "Val", "GT", self.image_name_list[idx]))
            image_sync_crop = self.prepare_val_data(image=image_in_np, image1=image_gt_np)
            image_in_ts = image_sync_crop['image']
            image_gt_ts = image_sync_crop['image1']
            return {
                'image_in': image_in_ts,
                'image_gt': image_gt_ts
            }

        elif self.subfolder == "Train":
            image_in_np = self.load_image(mf.smart_path_join(self.data_root, "Train", "Input", self.image_name_list[idx]))
            image_gt_np = self.load_image(mf.smart_path_join(self.data_root, "Train", "GT", self.image_name_list[idx]))
            image_sync_crop = self.prepare_train_both(image=image_in_np, image1=image_gt_np)
            image_in_crop = image_sync_crop['image']
            image_gt_crop = image_sync_crop['image1']

            image_gt_ts = self.numpy2tensor(image=image_gt_crop)['image']
            image_gt_ts_2 = F.interpolate(image_gt_ts.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
            image_gt_ts_4 = F.interpolate(image_gt_ts.unsqueeze(0), scale_factor=0.25, mode='bilinear').squeeze(0)
            image_in_ts = self.prepare_train_input_only(image=image_in_crop)['image']

            return {
                'image_in': image_in_ts,
                'image_gt': image_gt_ts,
                'image_gt_2': image_gt_ts_2,
                'image_gt_4': image_gt_ts_4,
                'image_in_ori': self.numpy2tensor(image=image_in_crop)['image'] # 未进行数据增强的输入图像
            }

def image_info(image): # for debug
    print("=-"*20)
    print("--------------------type:", type(image))
    print("--------------------shape:", image.shape)
    print("--------------------dtype:", image.dtype)
    print("--------------------range: ", image.min(), image.max())

def save_from_single_tansor(x, path): # for debug
    x = (x * 0.5 + 0.5) * 255.0
    x_np = x.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(x_np)
    image.save(path)


def cumulative_sum(lst):
    return [sum(lst[:i+1]) for i in range(len(lst))]


def get_state_idx(global_step, steps): # for train
    steps = np.array(steps)
    comp = (global_step < steps).nonzero()[0]
    state_idx = len(steps) - 1 if len(comp) == 0 else comp[0]
    return state_idx

from torch.utils.data import DataLoader
from torchvision.utils import save_image



from PIL import Image


if __name__ == "__main__":
    """
    # 加载数据集功能测试
    # 渐进式训练策略
    """
    # 初始化参数
    # progressive training
    steps_per_stage = [400, 4, 4, 4]
    steps = cumulative_sum(steps_per_stage)
    batch_sizes = [16, 1, 1, 1]
    patch_sizes = [1024, 128, 512, 1024]
    device = torch.device('cuda')
    epochs = 1000
    data_root = r"D:\dataset\NTIRE_2025"
    print(mf.smart_exists(data_root))

   # 构建数据集列表
    ntr_datasets = [
        NTIRE_LIE_Dataset(
            data_root=data_root,
            subfolder="Train",
            patch_size=ps
        ) for ps in patch_sizes
    ]
    print(ntr_datasets)
    for ds in ntr_datasets:
        print(ds.patch_size)

    # 构建数据加载器
    data_loaders = [
        DataLoader(
            dataset=ntr_datasets[i],
            batch_size=batch_sizes[i],
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        ) for i in range(len(batch_sizes))
    ]
    print(data_loaders)
    for dl in data_loaders:
        print(f"batch_size: {dl.batch_size}")
        print(f"patch_size: {dl.dataset.patch_size}")

    # 开始训练循环
    global_step = 0
    print("--start train!!!")
    for epoch in range(epochs):
        state_idx = get_state_idx(global_step, steps)
        loader = data_loaders[state_idx]
        print("---epoch: ", epoch)
        print(f"------state_idx: {state_idx}, batch_size: {loader.batch_size}, patch_size: {loader.dataset.patch_size}")

        for i, sample in enumerate(loader):
            global_step += 1
            print(f"                    step: {global_step}, ")
            image_in = sample['image_in'].to(device)
            image_gt = sample['image_gt'].to(device)
            image_gt_2 = sample['image_gt_2'].to(device)
            image_gt_4 = sample['image_gt_4'].to(device)



            image_gt, image_gt_2, image_gt_4 = map(lambda x: x*0.5+0.5, (image_gt, image_gt_2, image_gt_4))
            image_info(image_gt)
            image_info(image_gt_2)
            image_info(image_gt_4)

            save_image(image_gt, f'./image_gt_{i}.png')
            save_image(image_gt_2, f'./image_gt_2_{i}.png')
            save_image(image_gt_4, f'./image_gt_4_{i}.png')


            if state_idx != len(batch_sizes) - 1 and global_step > steps[state_idx]:
                break




























