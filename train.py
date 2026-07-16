from ultralytics import YOLO
import os
import multiprocessing


# Set environment variables for W&B
#os.environ["WANDB_API_KEY"] = "dcfab3ddaa0e68af2d7b6a2ae451b57da060876d"
#os.environ["WANDB_MODE"] = "offline"

def train_model():

    model = YOLO(r"D:\2024研三上\yolo\weld.yaml")

    model.train(data=r'C:\Users\1\Downloads\yolo\blue.yaml', epochs=200, imgsz=640, batch=8,optimizer='SGD')

if __name__ == '__main__':
    train_model()

#YOLOv8n summary: 225 layers, 3011433 parameters, 3011417 gradients, 8.2 GFLOPs
#C2f_FasterBlock summary: 246 layers, 2306233 parameters, 2306217 gradients, 6.4 GFLOPs
#C2f_FasterBlock+MLCA(Mixed local channel attention)  summary: 259 layers, 2650261 parameters, 2650245 gradients, 7.2 GFLOPs
#C2f_FasterBlock+MLCA(Mixed local channel attention) + VFLoss损失函数 summary+SGD :259 layers, 2650261 parameters, 2650245 gradients, 7.2 GFLOPs
#