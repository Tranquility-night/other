import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes=20, anchor_masks=[[6,7,8], [3,4,5], [0,1,2]],
                 anchors=np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]),
                 img_size=416):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.img_size = img_size
        self.strides = [img_size // 52, img_size // 26, img_size // 13]  # 8, 16, 32
        self.lambda_coord = 5.0  # 坐标损失权重
        self.lambda_obj = 1.0    # 前景损失权重
        self.lambda_noobj = 0.5  # 背景损失权重
        self.lambda_cls = 1.0    # 类别损失权重

    def _get_anchor_boxes(self, feat_size, stride, device):
        # 生成锚框
        anchors = self.anchors[self.anchor_masks[feat_size.index(stride)]] / stride
        h, w = feat_size, feat_size
        # 生成网格坐标
        grid_x = torch.arange(w, device=device).repeat(h, 1).unsqueeze(-1)
        grid_y = torch.arange(h, device=device).repeat(w, 1).t().unsqueeze(-1)
        grid = torch.cat([grid_x, grid_y], dim=-1).float()
        # 锚框宽高
        anchor_w = anchors[:, 0].unsqueeze(0).unsqueeze(0)
        anchor_h = anchors[:, 1].unsqueeze(0).unsqueeze(0)
        anchor_boxes = torch.cat([torch.zeros_like(anchor_w), torch.zeros_like(anchor_h), anchor_w, anchor_h], dim=-1)
        return grid, anchor_boxes

    def _iou(self, box1, box2):
        # 计算IOU（box格式：cx, cy, w, h）
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

        # 交集坐标
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 并集面积
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def forward(self, preds, targets):
        # preds: (pred52, pred26, pred13)
        # targets: [B, N, 5] (cx, cy, w, h, cls_idx)
        device = preds[0].device
        batch_size = preds[0].size(0)
        total_loss = 0.0

        # 遍历三个尺度
        for i, (pred, stride) in enumerate(zip(preds, self.strides)):
            feat_size = self.img_size // stride
            grid, anchor_boxes = self._get_anchor_boxes(feat_size, stride, device)
            num_anchors = anchor_boxes.size(-2)

            # 解析预测值
            pred = pred.reshape(batch_size, feat_size, feat_size, num_anchors, 5 + self.num_classes)
            pred_cx = torch.sigmoid(pred[..., 0]) + grid[..., 0:1]  # 中心x
            pred_cy = torch.sigmoid(pred[..., 1]) + grid[..., 1:2]  # 中心y
            pred_w = torch.exp(pred[..., 2:3]) * anchor_boxes[..., 2:3]  # 宽
            pred_h = torch.exp(pred[..., 3:4]) * anchor_boxes[..., 3:4]  # 高
            pred_conf = torch.sigmoid(pred[..., 4:5])  # 置信度
            pred_cls = torch.sigmoid(pred[..., 5:])    # 类别概率

            # 构建目标张量
            obj_mask = torch.zeros((batch_size, feat_size, feat_size, num_anchors, 1), device=device)
            noobj_mask = torch.ones((batch_size, feat_size, feat_size, num_anchors, 1), device=device)
            target_cx = torch.zeros_like(pred_cx)
            target_cy = torch.zeros_like(pred_cy)
            target_w = torch.zeros_like(pred_w)
            target_h = torch.zeros_like(pred_h)
            target_conf = torch.zeros_like(pred_conf)
            target_cls = torch.zeros_like(pred_cls)

            # 遍历每个批次
            for b in range(batch_size):
                if targets[b].size(0) == 0:
                    continue
                # 目标框缩放至当前特征尺度
                target_boxes = targets[b][:, :4] * feat_size
                target_cls_idx = targets[b][:, 4].long()

                # 计算目标框与锚框的IOU
                anchor_wh = anchor_boxes[..., 2:4].repeat(target_boxes.size(0), 1, 1)
                target_wh = target_boxes[:, 2:4].unsqueeze(1).repeat(1, num_anchors, 1)
                iou_anchors = self._iou(
                    torch.cat([torch.zeros_like(target_wh), target_wh], dim=-1),
                    torch.cat([torch.zeros_like(anchor_wh), anchor_wh], dim=-1)
                )
                # 匹配最佳锚框
                best_anchor_idx = torch.argmax(iou_anchors, dim=1)

                # 网格坐标
                grid_x_idx = (target_boxes[:, 0]).long()
                grid_y_idx = (target_boxes[:, 1]).long()

                # 更新掩码与目标值
                for t in range(target_boxes.size(0)):
                    gx = grid_x_idx[t]
                    gy = grid_y_idx[t]
                    a = best_anchor_idx[t]
                    # 前景掩码
                    obj_mask[b, gy, gx, a, :] = 1.0
                    noobj_mask[b, gy, gx, a, :] = 0.0
                    # 坐标目标
                    target_cx[b, gy, gx, a, :] = target_boxes[t, 0] - gx
                    target_cy[b, gy, gx, a, :] = target_boxes[t, 1] - gy
                    target_w[b, gy, gx, a, :] = torch.log(target_boxes[t, 2] / anchor_boxes[a, 2] + 1e-6)
                    target_h[b, gy, gx, a, :] = torch.log(target_boxes[t, 3] / anchor_boxes[a, 3] + 1e-6)
                    # 置信度目标
                    target_conf[b, gy, gx, a, :] = 1.0
                    # 类别目标
                    target_cls[b, gy, gx, a, target_cls_idx[t]] = 1.0

            # 计算各部分损失
            # 坐标损失（MSE）
            loss_cx = F.mse_loss(pred_cx * obj_mask, target_cx * obj_mask)
            loss_cy = F.mse_loss(pred_cy * obj_mask, target_cy * obj_mask)
            loss_w = F.mse_loss(pred_w * obj_mask, target_w * obj_mask)
            loss_h = F.mse_loss(pred_h * obj_mask, target_h * obj_mask)
            loss_coord = (loss_cx + loss_cy + loss_w + loss_h) * self.lambda_coord

            # 置信度损失（BCE）
            loss_obj = F.binary_cross_entropy(pred_conf * obj_mask, target_conf * obj_mask) * self.lambda_obj
            loss_noobj = F.binary_cross_entropy(pred_conf * noobj_mask, target_conf * noobj_mask) * self.lambda_noobj
            loss_conf = loss_obj + loss_noobj

            # 类别损失（BCE）
            loss_cls = F.binary_cross_entropy(pred_cls * obj_mask.repeat(1,1,1,1,self.num_classes),
                                              target_cls * obj_mask.repeat(1,1,1,1,self.num_classes)) * self.lambda_cls

            # 尺度总损失
            scale_loss = loss_coord + loss_conf + loss_cls
            total_loss += scale_loss

        return total_loss / len(preds)