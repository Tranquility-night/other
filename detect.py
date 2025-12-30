import os
import torch
import cv2
import numpy as np
import argparse

from models.yolov3 import YOLOV3

# 配置参数
CLASS_PATH = "./config/voc.names"
IMG_SIZE = 416
CONF_THRESH = 0.5
NMS_THRESH = 0.45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classes(class_path):
    with open(class_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def preprocess_img(img_path, img_size):
    # 读取并预处理图像
    img = cv2.imread(img_path)
    img_ori = img.copy()
    img_h, img_w = img.shape[:2]
    # 缩放
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)
    return img, img_ori, img_w, img_h

def postprocess(preds, img_w, img_h, img_size, conf_thresh, nms_thresh, num_classes):
    strides = [img_size // 52, img_size // 26, img_size // 13]
    all_boxes = []
    all_confs = []
    all_cls = []

    for i, (pred, stride) in enumerate(zip(preds, strides)):
        feat_size = img_size // stride
        num_anchors = 3
        # 解析预测
        pred = pred.reshape(1, feat_size, feat_size, num_anchors, 5 + num_classes)
        pred_cx = (torch.sigmoid(pred[..., 0]) + torch.arange(feat_size, device=DEVICE).repeat(feat_size, 1).unsqueeze(-1)) * stride
        pred_cy = (torch.sigmoid(pred[..., 1]) + torch.arange(feat_size, device=DEVICE).repeat(feat_size, 1).t().unsqueeze(-1)) * stride
        pred_w = torch.exp(pred[..., 2]) * (torch.tensor([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], device=DEVICE)[[6,7,8],[3,4,5],[0,1,2]][i][:, 0].unsqueeze(0).unsqueeze(0))
        pred_h = torch.exp(pred[..., 3]) * (torch.tensor([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], device=DEVICE)[[6,7,8],[3,4,5],[0,1,2]][i][:, 1].unsqueeze(0).unsqueeze(0))
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        pred_cls_idx = torch.argmax(pred_cls, dim=-1)
        pred_cls_conf = torch.max(pred_cls, dim=-1)[0]

        # 筛选置信度
        mask = pred_conf * pred_cls_conf > conf_thresh
        if mask.sum() == 0:
            continue

        # 转换为像素坐标
        cx = pred_cx[mask].cpu().numpy()
        cy = pred_cy[mask].cpu().numpy()
        w = pred_w[mask].cpu().numpy()
        h = pred_h[mask].cpu().numpy()
        conf = (pred_conf * pred_cls_conf)[mask].cpu().numpy()
        cls_idx = pred_cls_idx[mask].cpu().numpy()

        # 转换为xmin, ymin, xmax, ymax
        xmin = (cx - w / 2) * (img_w / img_size)
        ymin = (cy - h / 2) * (img_h / img_size)
        xmax = (cx + w / 2) * (img_w / img_size)
        ymax = (cy + h / 2) * (img_h / img_size)

        # 裁剪边界框
        xmin = np.clip(xmin, 0, img_w)
        ymin = np.clip(ymin, 0, img_h)
        xmax = np.clip(xmax, 0, img_w)
        ymax = np.clip(ymax, 0, img_h)

        all_boxes.append(np.stack([xmin, ymin, xmax, ymax], axis=-1))
        all_confs.append(conf)
        all_cls.append(cls_idx)

    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # 合并所有尺度结果
    boxes = np.concatenate(all_boxes, axis=0)
    confs = np.concatenate(all_confs, axis=0)
    cls_idx = np.concatenate(all_cls, axis=0)

    # 非极大值抑制（NMS）
    indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), confs.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    indices = indices.flatten() if len(indices.shape) > 1 else [indices]
    boxes = boxes[indices]
    confs = confs[indices]
    cls_idx = cls_idx[indices]

    return boxes, confs, cls_idx

def draw_detections(img_ori, boxes, confs, cls_idx, classes):
    for box, conf, cls in zip(boxes, confs, cls_idx):
        xmin, ymin, xmax, ymax = box.astype(int)
        # 绘制边界框
        cv2.rectangle(img_ori, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 绘制类别与置信度
        label = f"{classes[cls]}: {conf:.2f}"
        cv2.putText(img_ori, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_ori

def main(args):
    # 加载类别
    classes = load_classes(CLASS_PATH)
    num_classes = len(classes)

    # 初始化模型
    model = YOLOV3(num_classes=num_classes).to(DEVICE)
    # 加载权重
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()

    # 图像预处理
    img, img_ori, img_w, img_h = preprocess_img(args.img_path, IMG_SIZE)

    # 前向传播（无梯度）
    with torch.no_grad():
        preds = model(img)

    # 后处理
    boxes, confs, cls_idx = postprocess(preds, img_w, img_h, IMG_SIZE, CONF_THRESH, NMS_THRESH, num_classes)

    # 绘制检测结果
    if len(boxes) > 0:
        img_ori = draw_detections(img_ori, boxes, confs, cls_idx, classes)

    # 保存与显示结果
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, os.path.basename(args.img_path))
    cv2.imwrite(save_path, img_ori)
    print(f"Detection result saved to {save_path}")

    cv2.imshow("YOLO-V3 Detection", img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-V3 Object Detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_dir", type=str, default="./detect_results", help="Directory to save detection result")
    args = parser.parse_args()
    main(args)