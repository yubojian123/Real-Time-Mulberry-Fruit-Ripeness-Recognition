from ultralytics import YOLO
import os
import torch
import multiprocessing

# 清理缓存
torch.cuda.empty_cache()

# 设置 W&B 环境变量
os.environ["WANDB_API_KEY"] = "dcfab3ddaa0e68af2d7b6a2ae451b57da060876d"
os.environ["WANDB_MODE"] = "offline"

def train_model():
    model = YOLO(r"")
    model.train(data=r'C:\Users\1\Downloads\yolo\ultralytics-main\ultralytics\cfg\datasets\coco.yaml',
                epochs=200,
                imgsz=640,
                batch=16)

if __name__ == '__main__':
    # 创建训练进程
    train_process = multiprocessing.Process(target=train_model)
    train_process.start()
    train_process.join()
