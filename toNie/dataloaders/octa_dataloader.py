import os
from glob import glob
from typing import Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset

# 支持 3mm / 6mm 当作两个域，兼容 Fundus dataloader 的接口与返回格式
# 默认目录结构（相对 base_dir）：
#   <domain_subdir>/<split>/images/*.png
#   <domain_subdir>/<split>/labels/*.png
# 其中 split: train / value(验证) / test

_DOMAIN_MAP: Dict[str, str] = {
    "Domain1": "3mm",
    "Domain2": "6mm",
    "3mm": "3mm",
    "6mm": "6mm",
}


def _resolve_domain(domain: str) -> str:
    if domain not in _DOMAIN_MAP:
        raise ValueError(f"未知的域名/数据子目录: {domain}. 可用: {list(_DOMAIN_MAP.keys())}")
    return _DOMAIN_MAP[domain]


def _resolve_split(split: str) -> str:
    s = split.lower()
    if s in {"val", "value", "valid", "validation"}:
        return "value"
    if s in {"train", "test"}:
        return s
    raise ValueError(f"未知的 split: {split}. 仅支持 train / val(value) / test")


class OCTASegmentation(Dataset):
    """
    OCTA-500 单通道血管分割数据集。
    接口与 FundusSegmentation 对齐：返回 {'image', 'label', 'img_name'}
    """

    def __init__(
        self,
        base_dir: str,
        dataset: str = "Domain1",
        split: str = "train",
        transform=None,
    ):
        self._base_dir = base_dir
        self.dataset = _resolve_domain(dataset)
        self.split = _resolve_split(split)
        self.transform = transform

        self._image_dir = os.path.join(self._base_dir, self.dataset, self.split, "images")
        self._label_dir = os.path.join(self._base_dir, self.dataset, self.split, "labels")

        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(f"找不到图像目录: {self._image_dir}")
        if not os.path.isdir(self._label_dir):
            raise FileNotFoundError(f"找不到标签目录: {self._label_dir}")

        exts = ("*.png", "*.bmp", "*.tif", "*.tiff")
        imagelist = []
        for ext in exts:
            imagelist.extend(glob(os.path.join(self._image_dir, ext)))
        imagelist = sorted(imagelist)

        if not imagelist:
            raise RuntimeError(f"在 {self._image_dir} 未找到图像文件 (支持扩展名: {exts})")

        self.image_list = []
        for image_path in imagelist:
            fname = os.path.basename(image_path)
            label_path = os.path.join(self._label_dir, fname)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"找不到与图像同名的标签: {label_path}")
            self.image_list.append({"image": image_path, "label": label_path})

        print(f"{self._image_dir}")
        print(f"Number of images in {self.split}: {len(self.image_list)}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        sample_info = self.image_list[index]
        img_path = sample_info["image"]
        lbl_path = sample_info["label"]

        # OCTA 为单通道灰度
        # 灰度 → 三通道，保持与现有网络输入一致
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        img_name = os.path.basename(img_path)

        sample = {"image": img, "label": lbl, "img_name": img_name}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class OCTASegmentation_2transform(Dataset):
    """
    OCTA-500，返回弱/强增广的成对样本，接口对齐 FundusSegmentation_2transform。
    """

    def __init__(
        self,
        base_dir: str,
        dataset: str = "Domain1",
        split: str = "train",
        transform_weak=None,
        transform_strong=None,
    ):
        self._base_dir = base_dir
        self.dataset = _resolve_domain(dataset)
        self.split = _resolve_split(split)
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

        self._image_dir = os.path.join(self._base_dir, self.dataset, self.split, "images")
        self._label_dir = os.path.join(self._base_dir, self.dataset, self.split, "labels")

        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(f"找不到图像目录: {self._image_dir}")
        if not os.path.isdir(self._label_dir):
            raise FileNotFoundError(f"找不到标签目录: {self._label_dir}")

        exts = ("*.png", "*.bmp", "*.tif", "*.tiff")
        imagelist = []
        for ext in exts:
            imagelist.extend(glob(os.path.join(self._image_dir, ext)))
        imagelist = sorted(imagelist)

        if not imagelist:
            raise RuntimeError(f"在 {self._image_dir} 未找到图像文件 (支持扩展名: {exts})")

        self.image_list = []
        for image_path in imagelist:
            fname = os.path.basename(image_path)
            label_path = os.path.join(self._label_dir, fname)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"找不到与图像同名的标签: {label_path}")
            self.image_list.append({"image": image_path, "label": label_path})

        print(f"{self._image_dir}")
        print(f"Number of images in {self.split}: {len(self.image_list)}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        sample_info = self.image_list[index]
        img_path = sample_info["image"]
        lbl_path = sample_info["label"]

        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        img_name = os.path.basename(img_path)

        base_sample = {"image": img, "label": lbl, "img_name": img_name}

        sample_weak = self.transform_weak(base_sample) if self.transform_weak else base_sample
        sample_strong = self.transform_strong(base_sample) if self.transform_strong else base_sample

        return sample_weak, sample_strong

