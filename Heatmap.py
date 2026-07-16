import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os


def basic_heatmap():
    """最基本的热力图版本"""
    try:
        # 设置路径
        model_path = r'D:\2024研三上\yolo\yolo_training\ndt_detection\weights\best.pt'
        image_path = r'D:\2024研三上\yolo\WELD_1\1 (1).png'

        print("1. 检查文件是否存在...")
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return
        print("文件检查通过!")

        print("2. 加载YOLO模型...")
        model = YOLO(model_path)
        print(f"模型加载成功! 类别: {model.names}")

        print("3. 进行推理...")
        results = model(image_path)

        print("4. 保存结果...")
        # 保存检测结果
        for i, r in enumerate(results):
            r.save(f'result_{i}.jpg')

        print("5. 生成热力图风格的图像...")
        # 使用matplotlib生成热力图风格的可视化
        import matplotlib.pyplot as plt

        # 读取原图
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 创建简单的热力图效果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 显示原图
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # 显示检测结果
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        ax2.imshow(result_img_rgb)
        ax2.set_title('Detection Result')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig('heatmap_visualization.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        print("任务完成!")
        print("生成的文件:")
        print("- result_0.jpg: 检测结果图像")
        print("- heatmap_visualization.jpg: 对比可视化图像")

        # 打印检测信息
        print("\n检测到的目标:")
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  {i + 1}. {model.names[cls]} (置信度: {conf:.3f})")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    basic_heatmap()