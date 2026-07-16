import gc
import torch
from ultralytics import YOLO
from pathlib import Path


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU内存清理完成，当前占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print("系统内存清理完成")


def check_gpu_status():
    """检查GPU状态"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"🎯 检测到GPU: {gpu_name}")
        print(f"🎯 GPU内存: {gpu_memory:.1f} GB")
        return True, gpu_memory
    else:
        print("🔶 未检测到GPU，将使用CPU训练")
        return False, 0


def get_training_config(gpu_available, gpu_memory):
    """根据硬件配置返回训练参数"""
    if gpu_available and gpu_memory >= 8:
        # 8GB以上GPU
        config = {
            'device': 0,
            'batch': 16,
            'imgsz': 640,
            'workers': 2
        }
    elif gpu_available and gpu_memory >= 4:
        # 4-8GB GPU
        config = {
            'device': 0,
            'batch': 8,
            'imgsz': 640,
            'workers': 1
        }
    elif gpu_available and gpu_memory >= 2:
        # 2-4GB GPU
        config = {
            'device': 0,
            'batch': 4,
            'imgsz': 320,
            'workers': 0
        }
    else:
        # CPU训练
        config = {
            'device': 'cpu',
            'batch': 4,
            'imgsz': 320,
            'workers': 0
        }

    return config


def train_yolov8():
    """主训练函数"""
    print("🚀 开始YOLOv8训练...")

    # 数据集配置文件路径
    yaml_path = "D:/2024/yolo/ndt_dataset/dataset.yaml"

    # 验证YAML文件是否存在
    if not Path(yaml_path).exists():
        print(f"❌ 错误: 找不到YAML配置文件 {yaml_path}")
        print("请先运行数据集分割器生成数据集")
        return None

    print(f"✅ 找到配置文件: {yaml_path}")

    # 检查GPU状态
    gpu_available, gpu_memory = check_gpu_status()

    # 获取训练配置
    config = get_training_config(gpu_available, gpu_memory)

    # 训练前清理内存
    cleanup_memory()

    # 加载模型
    print("📦 加载YOLOv8模型...")
    model = YOLO('yolov8n.pt')  # 使用预训练模型

    # 训练参数
    train_args = {
        'data': yaml_path,
        'epochs': 100,
        'imgsz': config['imgsz'],
        'batch': config['batch'],
        'workers': config['workers'],
        'device': config['device'],
        'patience': 15,  # 早停
        'cache': False,  # 禁用缓存以减少内存使用
        'amp': True,  # 自动混合精度（节省显存）
        'verbose': True,  # 显示详细日志
        'save': True,
        'exist_ok': True,  # 允许覆盖现有文件
        'project': 'yolo_training',
        'name': 'ndt_detection'
    }

    print(f"\n🎯 训练配置:")
    print(f"  设备: {'GPU' if config['device'] != 'cpu' else 'CPU'}")
    print(f"  批次大小: {config['batch']}")
    print(f"  图像尺寸: {config['imgsz']}")
    print(f"  数据加载进程: {config['workers']}")
    print(f"  训练轮次: 100")
    print(f"  早停耐心值: 15")

    try:
        # 开始训练
        print("\n⏳ 开始训练...")
        results = model.train(**train_args)

        print("\n🎉 训练完成!")
        return results

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg or "页面文件" in error_msg:
            print(f"\n⚠️ 内存不足错误: {e}")
            print("尝试使用更保守的设置...")

            cleanup_memory()

            # 使用更保守的设置
            conservative_args = {
                'data': yaml_path,
                'epochs': 100,
                'imgsz': 320,  # 减小图像尺寸
                'batch': 2,  # 更小的批次
                'workers': 0,  # 不使用多进程
                'device': 'cpu',  # 强制使用CPU
                'patience': 20,
                'cache': False,
                'amp': False,  # 禁用混合精度
                'verbose': True,
                'project': 'yolo_training',
                'name': 'ndt_detection_cpu'
            }

            print(f"\n🔄 使用保守配置重新训练:")
            print(f"  设备: CPU")
            print(f"  批次大小: 2")
            print(f"  图像尺寸: 320")

            results = model.train(**conservative_args)
            return results
        else:
            print(f"❌ 训练错误: {e}")
            raise e


def show_training_results(results):
    """显示训练结果"""
    if results:
        print("\n📊 训练结果摘要:")
        print(f"最佳模型保存在: runs/detect/train/")

        # 尝试获取关键指标
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
                print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
                print(f"精确度: {metrics.get('metrics/precision(B)', 'N/A'):.3f}")
                print(f"召回率: {metrics.get('metrics/recall(B)', 'N/A'):.3f}")
        except:
            print("无法获取详细指标，请查看训练日志")


if __name__ == "__main__":
    # 运行训练
    results = train_yolov8()

    # 显示结果
    show_training_results(results)

    # 使用说明
    print("\n💡 使用说明:")
    print("1. 最佳模型: runs/detect/ndt_detection/weights/best.pt")
    print("2. 训练日志: runs/detect/ndt_detection/")
    print("3. 使用模型进行预测:")
    print("   model = YOLO('runs/detect/ndt_detection/weights/best.pt')")
    print("   results = model.predict('your_image.jpg')")