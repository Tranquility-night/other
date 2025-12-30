import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.yolov3 import YOLOV3
from utils.dataset import VOCDataset
from utils.loss import YOLOv3Loss

# 配置参数
VOC_ROOT = "./data/VOCdevkit"
IMG_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
STEP_SIZE = 10
GAMMA = 0.5
SAVE_DIR = "./weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # 1. 加载数据集
    train_dataset = VOCDataset(VOC_ROOT, year="2007", split="train", img_size=IMG_SIZE, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 2. 初始化模型、损失函数、优化器
    model = YOLOV3(num_classes=20).to(DEVICE)
    criterion = YOLOv3Loss(num_classes=20, img_size=IMG_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # 3. 训练过程
    train_losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, bboxes in pbar:
            # 数据转移到设备
            imgs = torch.stack(imgs).to(DEVICE)
            bboxes_list = [bbox.to(DEVICE) for bbox in bboxes]

            # 前向传播
            preds = model(imgs)
            # 计算损失
            loss = criterion(preds, bboxes_list)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            total_loss += loss.item()
            pbar.set_postfix({"Batch Loss": loss.item()})

        # 学习率衰减
        scheduler.step()

        # 记录平均损失
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(SAVE_DIR, f"yolov3_voc_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # 4. 保存最终模型
    final_model_path = os.path.join(SAVE_DIR, "yolov3_voc_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # 5. 绘制损失曲线
    plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLO-V3 Training Loss Curve")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()