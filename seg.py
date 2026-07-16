from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO(r'D:\2024研三上\yolo\last.pt')

# 预测
results = model(r"D:\2024研三上\yolo\WELD_1\1 (1).png")

# 获取第一个结果
result = results[0]

# 保存带分割结果的图像
result.save("segmentation_result.jpg")

# 获取分割掩码
if result.masks is not None:
    # 获取所有掩码
    masks = result.masks.data
    print(f"检测到 {len(masks)} 个分割掩码")

    # 保存每个掩码
    for i, mask in enumerate(masks):
        # 将掩码转换为numpy数组并调整尺寸
        mask_np = mask.cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)

        # 保存单个掩码
        cv2.imwrite(f"mask_{i}.png", mask_np)