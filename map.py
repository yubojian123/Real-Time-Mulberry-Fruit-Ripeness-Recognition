import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch, cv2, numpy as np
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import os

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 只返回 YOLO 模型输出的第一个元素
        output = self.model(x)[0]
        # 启用 requires_grad
        output.requires_grad = True
        return output

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, conf_threshold=0.25, show_box=True, renormalize=True):
        device = torch.device(device)
        self.original_model = attempt_load_weights(weight, device)
        self.original_model.eval()
        self.model = YOLOWrapper(self.original_model)  # 使用包装器
        self.device = device

        self.target_layers = [self.original_model.model[l] for l in layer]
        self.method = method(self.model, self.target_layers)
        self.conf_threshold = conf_threshold
        self.show_box = show_box
        self.renormalize = renormalize

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        grayscale_cam = self.method(tensor)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.original_model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], (0, 255, 0), f'{data[4]:.2f}', cam_image)

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                name = img_path_.rsplit('.')[0]
                end_name = img_path_.rsplit('.')[-1]
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{name}_heatmap.{end_name}')
        else:
            self.process(img_path, f'{save_path}/result_heatmap.png')

def get_params():
    return {
        'weight': 'yolov8n.pt',  # Path to YOLOv8 weight file
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',  # Device: cuda or cpu
        'method': GradCAMPlusPlus,  # Grad-CAM method
        'layer': [10, 12, 14, 16],  # YOLO model layers to use for Grad-CAM
        'conf_threshold': 0.25,  # Confidence threshold for object detection
        'show_box': True,  # Whether to show bounding boxes
        'renormalize': True,  # Renormalize CAM output for better visualization
    }

if __name__ == '__main__':
    params = get_params()
    heatmap_generator = yolov8_heatmap(**params)
    heatmap_generator(r'C:\Users\1\Downloads\yolo\R-C.jpg', r'C:\Users\1\Desktop\heatmaps')
