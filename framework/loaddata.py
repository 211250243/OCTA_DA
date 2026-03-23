import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class BMPDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_size=None):
        """
        初始化BMP数据集加载器
        
        参数:
            image_dir (str): 包含输入图像的目录路径
            label_dir (str): 包含标签图像的目录路径
            transform (callable, optional): 可选的数据增强/转换
            target_size (tuple, optional): 目标尺寸 (H, W)，如果提供则进行零填充
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        
        # 获取所有图像文件名(确保image和label目录中的文件一一对应)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.bmp', '.tif', '.png'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(('.bmp', '.tif', '.png'))])
        
        # 验证文件对应关系
        assert len(self.image_files) == len(self.label_files), "图像和标签数量不匹配"
        for img, lbl in zip(self.image_files, self.label_files):
            assert img == lbl, f"文件名不匹配: {img} != {lbl}"

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取单个样本
        返回:
            image (torch.Tensor): 图像张量，形状为 [C, H, W]
            label (torch.Tensor): 标签张量，形状为 [C, H, W]
            filename (str): 当前样本的文件名(不带扩展名)
        """
        # 获取文件名
        filename = os.path.splitext(self.image_files[idx])[0]
        
        # 读取图像和标签
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        lbl_path = os.path.join(self.label_dir, self.label_files[idx])
        
        # 使用OpenCV读取灰度图像(单通道)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保读取成功
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        if label is None:
            raise ValueError(f"无法读取标签: {lbl_path}")
            
        # 添加通道维度(从H,W变为1,H,W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        
        # 转换为float32并归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0
        
        # 转换为torch张量
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        # 如果指定了target_size，进行填充
        if self.target_size is not None:
            current_height, current_width = image.shape[1], image.shape[2]
            target_height, target_width = self.target_size
            
            # 计算需要填充的像素数
            pad_height = max(target_height - current_height, 0)
            pad_width = max(target_width - current_width, 0)
            
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # 对图像和标签进行填充
            image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            label = torch.nn.functional.pad(label, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # 应用转换(如果有)
        if self.transform:
            # 合并图像和标签一起变换
            stacked = torch.cat([image, label], dim=0)  # shape: [2, H, W]
            stacked = self.transform(stacked)
            image, label = stacked[0:1], stacked[1:2]  # 重新拆分
        return image, label, filename


# # 示例使用
# if __name__ == "__main__":
#     # 定义数据转换
#     transform = transforms.Compose([
#         # transforms.ToTensor()  # 不再需要，因为我们已经手动转换为张量
#         # 可以添加其他转换，如标准化、数据增强等
#     ])
    
#     # 创建数据集实例
#     image_dir = "./images"  # 替换为你的图像目录
#     label_dir = "./labels"  # 替换为你的标签目录
#     dataset = BMPDataset(image_dir, label_dir, transform=transform)
    
#     # 创建数据加载器
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
#     net = Directention(in_channels=1)
#     # 测试数据加载器
#     for batch_idx, (images, labels, filenames) in enumerate(dataloader):
#         print(f"Batch {batch_idx}:")
#         print(f" - 图像形状: {images.shape} (min={images.min()}, max={images.max()})")
#         print(f" - 标签形状: {labels.shape} (min={labels.min()}, max={labels.max()})")
#         print(f" - 文件名: {filenames}")
#         print(net(images).shape)
#         # 这里可以添加你的训练/验证代码
#         # ...
        
#         # 只测试第一个batch
#         if batch_idx == 0:
#             break