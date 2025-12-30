import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, voc_root, year="2007", split="train", img_size=416, augment=True):
        self.voc_root = voc_root
        self.img_size = img_size
        self.augment = augment
        self.classes = self._get_classes("../config/voc.names")
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 加载图像路径
        if split == "train":
            self.img_paths = self._load_img_paths(year, ["train", "val"])
        else:
            self.img_paths = self._load_img_paths(year, [split])

    def _get_classes(self, class_path):
        with open(class_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def _load_img_paths(self, year, splits):
        img_paths = []
        for split in splits:
            txt_path = os.path.join(self.voc_root, f"VOC{year}/ImageSets/Main/{split}.txt")
            with open(txt_path, "r") as f:
                img_ids = [line.strip() for line in f.readlines()]
            for img_id in img_ids:
                img_path = os.path.join(self.voc_root, f"VOC{year}/JPEGImages/{img_id}.jpg")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
        return img_paths

    def _parse_annotation(self, img_path):
        # 解析XML标注文件
        img_dir = os.path.dirname(img_path)
        anno_dir = img_dir.replace("JPEGImages", "Annotations")
        img_name = os.path.basename(img_path).replace(".jpg", ".xml")
        anno_path = os.path.join(anno_dir, img_name)

        tree = ET.parse(anno_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        bboxes = []
        for obj in root.iter("object"):
            # 获取类别
            cls_name = obj.find("name").text
            if cls_name not in self.class2idx:
                continue
            cls_idx = self.class2idx[cls_name]

            # 获取边界框
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # 边界框归一化
            xmin = max(0, min(xmin, img_w))
            ymin = max(0, min(ymin, img_h))
            xmax = max(0, min(xmax, img_w))
            ymax = max(0, min(ymax, img_h))
            if xmax <= xmin or ymax <= ymin:
                continue

            # 转换为中心坐标+宽高
            cx = (xmin + xmax) / (2 * img_w)
            cy = (ymin + ymax) / (2 * img_h)
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            bboxes.append([cx, cy, w, h, cls_idx])

        return np.array(bboxes, dtype=np.float32)

    def _augment(self, img, bboxes):
        # 水平翻转（简单数据增强）
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            bboxes[:, 0] = 1 - bboxes[:, 0]  # 翻转x坐标
        return img, bboxes

    def _preprocess(self, img):
        # 图像缩放与归一化
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        return img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 解析标注
        bboxes = self._parse_annotation(img_path)
        # 数据增强
        if self.augment:
            img, bboxes = self._augment(img, bboxes)
        # 图像预处理
        img = self._preprocess(img)
        # 转换为Tensor
        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros((0, 5)).float()
        return img, bboxes