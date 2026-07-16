# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO

sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        self.conf = 0.25
        self.iou = 0.7

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)  # 添加批量图片检测方法

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)), device=self.device)  # 预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 90)
        self.ui.tableWidget.setColumnWidth(4, 230)
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 设置标签属性，确保可以显示图片
        self.ui.label_show.setScaledContents(False)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        self.ui.label_show.setMinimumSize(100, 100)

    def detact_batch_imgs(self):
        """批量图片检测方法"""
        try:
            if self.cap:
                # 打开图片前关闭摄像头
                self.video_stop()
                self.is_camera_open = False
                self.ui.CaplineEdit.setText('摄像头未开启')
                self.cap = None

            directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
            if not directory:
                return

            self.org_path = directory
            img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
            image_files = []

            # 先收集所有图片文件
            for file_name in os.listdir(directory):
                full_path = os.path.join(directory, file_name)
                if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                    image_files.append(full_path)

            if not image_files:
                QMessageBox.warning(self, "提示", "文件夹中没有找到图片文件！")
                return

            # 处理每个图片文件
            for img_path in image_files:
                try:
                    print(f"处理图片: {img_path}")
                    self.org_img = cv2.imread(img_path)
                    if self.org_img is None:
                        print(f"无法读取图片: {img_path}")
                        continue

                    # 目标检测
                    t1 = time.time()
                    results = self.model(img_path, conf=self.conf, iou=self.iou, classes=range(0, 14))[0]
                    t2 = time.time()
                    take_time_str = '{:.3f} s'.format(t2 - t1)
                    self.ui.time_lb.setText(take_time_str)

                    # 安全地获取检测结果
                    if results.boxes is not None:
                        location_list = results.boxes.xyxy.tolist()
                        self.location_list = [list(map(int, e)) for e in location_list]
                        cls_list = results.boxes.cls.tolist()
                        self.cls_list = [int(i) for i in cls_list]
                        self.conf_list = results.boxes.conf.tolist()
                        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                    else:
                        self.location_list = []
                        self.cls_list = []
                        self.conf_list = []

                    now_img = results.plot()
                    self.draw_img = now_img

                    # 显示图片
                    self.show_detection_image(now_img)

                    # 设置路径显示
                    self.ui.PiclineEdit.setText(img_path)

                    # 目标数目
                    target_nums = len(self.cls_list)
                    self.ui.label_nums.setText(str(target_nums))

                    # 设置目标选择下拉框
                    choose_list = ['全部']
                    if target_nums > 0:
                        # 安全地创建目标名称列表
                        target_names = []
                        for index, type_id in enumerate(self.cls_list):
                            # 确保Config.names存在且是列表或字典
                            if hasattr(Config, 'names') and Config.names is not None:
                                if isinstance(Config.names, (list, tuple)) and 0 <= type_id < len(Config.names):
                                    target_name = f"{Config.names[type_id]}_{index}"
                                elif isinstance(Config.names, dict) and type_id in Config.names:
                                    target_name = f"{Config.names[type_id]}_{index}"
                                else:
                                    target_name = f"类型{type_id}_{index}"
                            else:
                                target_name = f"类型{type_id}_{index}"
                            target_names.append(target_name)
                        choose_list.extend(target_names)

                    self.ui.comboBox.clear()
                    self.ui.comboBox.addItems(choose_list)

                    # 更新目标信息
                    self.update_target_info()

                    # 添加到表格
                    if target_nums > 0:
                        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                    self.ui.tableWidget.scrollToBottom()
                    QApplication.processEvents()  # 刷新页面

                except Exception as e:
                    print(f"处理图片 {img_path} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"批量处理图片错误: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"批量处理图片时发生错误：{str(e)}")

    def open_img(self):
        try:
            if self.cap:
                # 打开图片前关闭摄像头
                self.video_stop()
                self.is_camera_open = False
                self.ui.CaplineEdit.setText('摄像头未开启')
                self.cap = None

            # 弹出的窗口名称：'打开图片'
            file_path, _ = QFileDialog.getOpenFileName(
                self.ui.centralwidget,
                '打开图片',
                './',
                "Image files (*.jpg *.jpeg *.png *.bmp *.gif)"
            )

            if not file_path:
                return

            print(f"选择的文件: {file_path}")
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "错误", "图片文件不存在！")
                return

            self.ui.comboBox.setDisabled(False)
            self.org_path = file_path

            # 使用OpenCV读取图片
            self.org_img = cv2.imread(self.org_path)
            if self.org_img is None:
                QMessageBox.warning(self, "错误", "无法读取图片文件！")
                return

            print(f"图片读取成功，尺寸: {self.org_img.shape}")

            # 目标检测
            t1 = time.time()
            self.results = self.model(self.org_path, conf=self.conf, iou=self.iou, classes=range(0, 14))[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            # 安全地获取检测结果
            if self.results.boxes is not None:
                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
            else:
                # 如果没有检测到目标，初始化空列表
                self.location_list = []
                self.cls_list = []
                self.conf_list = []

            # 使用YOLO的plot方法绘制结果
            now_img = self.results.plot()
            print(f"检测后图片尺寸: {now_img.shape}")

            self.draw_img = now_img

            # 显示图片
            self.show_detection_image(now_img)

            # 设置路径显示
            self.ui.PiclineEdit.setText(self.org_path)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框 - 修复set相关错误
            choose_list = ['全部']
            if target_nums > 0:
                # 安全地创建目标名称列表
                target_names = []
                for index, type_id in enumerate(self.cls_list):
                    # 确保Config.names存在且是列表或字典
                    if hasattr(Config, 'names') and Config.names is not None:
                        if isinstance(Config.names, (list, tuple)) and 0 <= type_id < len(Config.names):
                            target_name = f"{Config.names[type_id]}_{index}"
                        elif isinstance(Config.names, dict) and type_id in Config.names:
                            target_name = f"{Config.names[type_id]}_{index}"
                        else:
                            target_name = f"类型{type_id}_{index}"
                    else:
                        target_name = f"类型{type_id}_{index}"
                    target_names.append(target_name)
                choose_list.extend(target_names)  # 使用extend而不是直接相加

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            # 更新目标信息显示
            self.update_target_info()

            # 删除表格所有行并显示新信息
            self.ui.tableWidget.setRowCount(0)
            self.ui.tableWidget.clearContents()
            if target_nums > 0:
                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

            print("图片显示完成")

        except Exception as e:
            print(f"打开图片错误: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            QMessageBox.warning(self, "错误", f"处理图片时发生错误：{str(e)}")

    def update_target_info(self):
        """更新目标信息显示"""
        target_nums = len(self.cls_list)

        if target_nums >= 1:
            # 安全地获取类别名称
            if hasattr(Config, 'CH_names') and Config.CH_names is not None:
                if isinstance(Config.CH_names, (list, tuple)) and 0 <= self.cls_list[0] < len(Config.CH_names):
                    type_name = Config.CH_names[self.cls_list[0]]
                elif isinstance(Config.CH_names, dict) and self.cls_list[0] in Config.CH_names:
                    type_name = Config.CH_names[self.cls_list[0]]
                else:
                    type_name = f"类型{self.cls_list[0]}"
            else:
                type_name = f"类型{self.cls_list[0]}"

            self.ui.type_lb.setText(type_name)
            if self.conf_list:
                self.ui.label_conf.setText(str(self.conf_list[0]))
            # 设置坐标位置值
            if self.location_list:
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

    def show_detection_image(self, cv_img):
        """显示检测后的图片"""
        try:
            # 确保图片是3通道的
            if len(cv_img.shape) == 3:
                # 转换颜色空间 BGR to RGB
                if cv_img.shape[2] == 3:
                    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                else:
                    cv_img_rgb = cv_img
            else:
                # 如果是灰度图，转换为3通道
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

            height, width = cv_img_rgb.shape[:2]
            print(f"显示图片尺寸: {width}x{height}")

            # 计算缩放尺寸
            img_width, img_height = self.get_resize_size(cv_img_rgb)

            # 调整图片尺寸
            resize_cvimg = cv2.resize(cv_img_rgb, (img_width, img_height))

            # 转换为QImage
            bytes_per_line = 3 * img_width
            q_image = QImage(resize_cvimg.data, img_width, img_height, bytes_per_line, QImage.Format_RGB888)

            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self.ui.label_show.setPixmap(pixmap)

        except Exception as e:
            print(f"显示图片错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def combox_change(self):
        try:
            com_text = self.ui.comboBox.currentText()
            if com_text == '全部':
                # 显示全部检测结果
                if hasattr(self, 'results'):
                    cur_img = self.results.plot()
                    self.show_detection_image(cur_img)
                    if self.cls_list:
                        self.update_target_info()
            else:
                # 显示单个目标
                try:
                    index = int(com_text.split('_')[-1])
                    if (hasattr(self, 'results') and
                            hasattr(self, 'location_list') and
                            index < len(self.location_list)):

                        # 获取单个目标的图像
                        if hasattr(self.results, '__getitem__'):
                            single_result = self.results[index]
                            if hasattr(single_result, 'plot'):
                                cur_img = single_result.plot()
                                self.show_detection_image(cur_img)

                        # 更新信息显示
                        if (index < len(self.cls_list) and
                                index < len(self.conf_list) and
                                index < len(self.location_list)):

                            # 更新类型
                            if hasattr(Config, 'CH_names') and Config.CH_names is not None:
                                if (isinstance(Config.CH_names, (list, tuple)) and
                                        0 <= self.cls_list[index] < len(Config.CH_names)):
                                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
                                elif (isinstance(Config.CH_names, dict) and
                                      self.cls_list[index] in Config.CH_names):
                                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
                                else:
                                    self.ui.type_lb.setText(f"类型{self.cls_list[index]}")
                            else:
                                self.ui.type_lb.setText(f"类型{self.cls_list[index]}")

                            # 更新置信度
                            self.ui.label_conf.setText(str(self.conf_list[index]))

                            # 更新坐标
                            location = self.location_list[index]
                            self.ui.label_xmin.setText(str(location[0]))
                            self.ui.label_ymin.setText(str(location[1]))
                            self.ui.label_xmax.setText(str(location[2]))
                            self.ui.label_ymax.setText(str(location[3]))

                except (ValueError, IndexError, AttributeError) as e:
                    print(f"解析下拉框选项错误: {str(e)}")
                    return

        except Exception as e:
            print(f"下拉框切换错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def tabel_info_show(self, locations, clses, confs, path=None):
        """在表格中显示检测信息"""
        try:
            for i, (location, cls, conf) in enumerate(zip(locations, clses, confs)):
                row_count = self.ui.tableWidget.rowCount()
                self.ui.tableWidget.insertRow(row_count)

                # 序号
                item_id = QTableWidgetItem(str(row_count + 1))
                item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                # 路径
                item_path = QTableWidgetItem(str(path))

                # 类别名称 - 安全地获取
                cls_name = f"类型{cls}"  # 默认值
                if hasattr(Config, 'CH_names') and Config.CH_names is not None:
                    if isinstance(Config.CH_names, (list, tuple)) and 0 <= cls < len(Config.CH_names):
                        cls_name = Config.CH_names[cls]
                    elif isinstance(Config.CH_names, dict) and cls in Config.CH_names:
                        cls_name = Config.CH_names[cls]

                item_cls = QTableWidgetItem(cls_name)
                item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                # 置信度
                item_conf = QTableWidgetItem(str(conf))
                item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                # 位置
                item_location = QTableWidgetItem(str(location))

                self.ui.tableWidget.setItem(row_count, 0, item_id)
                self.ui.tableWidget.setItem(row_count, 1, item_path)
                self.ui.tableWidget.setItem(row_count, 2, item_cls)
                self.ui.tableWidget.setItem(row_count, 3, item_conf)
                self.ui.tableWidget.setItem(row_count, 4, item_location)

            self.ui.tableWidget.scrollToBottom()

        except Exception as e:
            print(f"表格显示错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Video files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(30)
        self.timer_camera.timeout.connect(self.open_frame)

    def video_stop(self):
        if self.cap:
            self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        try:
            ret, now_img = self.cap.read()
            if ret:
                # 目标检测
                t1 = time.time()
                results = self.model(now_img, conf=self.conf, iou=self.iou, vid_stride=1, classes=range(0, 14))[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                # 安全地获取检测结果
                if results.boxes is not None:
                    location_list = results.boxes.xyxy.tolist()
                    self.location_list = [list(map(int, e)) for e in location_list]
                    cls_list = results.boxes.cls.tolist()
                    self.cls_list = [int(i) for i in cls_list]
                    self.conf_list = results.boxes.conf.tolist()
                    self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                else:
                    self.location_list = []
                    self.cls_list = []
                    self.conf_list = []

                # 视频框绘制
                now_img = results.plot()

                # 显示图片
                self.show_detection_image(now_img)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框
                choose_list = ['全部']
                if target_nums > 0:
                    target_names = []
                    for index, type_id in enumerate(self.cls_list):
                        if hasattr(Config, 'names') and Config.names is not None:
                            if isinstance(Config.names, (list, tuple)) and 0 <= type_id < len(Config.names):
                                target_name = f"{Config.names[type_id]}_{index}"
                            elif isinstance(Config.names, dict) and type_id in Config.names:
                                target_name = f"{Config.names[type_id]}_{index}"
                            else:
                                target_name = f"类型{type_id}_{index}"
                        else:
                            target_name = f"类型{type_id}_{index}"
                        target_names.append(target_name)
                    choose_list.extend(target_names)

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                # 更新目标信息
                self.update_target_info()

                # 添加到表格
                if target_nums > 0:
                    self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

            else:
                if self.cap:
                    self.cap.release()
                self.timer_camera.stop()

        except Exception as e:
            print(f"处理视频帧错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头！")
                self.is_camera_open = False
                self.ui.CaplineEdit.setText('摄像头未开启')
                return
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            if self.cap:
                self.cap.release()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        try:
            _img = img.copy()
            img_height, img_width = _img.shape[:2]
            ratio = img_width / img_height
            if ratio >= self.show_width / self.show_height:
                img_width = self.show_width
                img_height = int(img_width / ratio)
            else:
                img_height = self.show_height
                img_width = int(img_height * ratio)
            return img_width, img_height
        except Exception as e:
            print(f"计算缩放尺寸错误: {str(e)}")
            return self.show_width, self.show_height

    def save_detect_video(self):
        try:
            if self.cap is None and not self.org_path:
                QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
                return

            if self.is_camera_open:
                QMessageBox.about(self, '提示', '摄像头视频无法保存!')
                return

            if self.cap:
                res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if res == QMessageBox.Yes:
                    self.video_stop()
                    com_text = self.ui.comboBox.currentText()
                    self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text, self.conf, self.iou)
                    self.btn2Thread_object.start()
                    self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
                else:
                    return
            else:
                if os.path.isfile(self.org_path):
                    fileName = os.path.basename(self.org_path)
                    name, end_name = fileName.rsplit(".", 1)
                    save_name = name + '_detect_result.' + end_name
                    save_img_path = os.path.join(Config.save_path, save_name)
                    # 确保保存目录存在
                    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                    # 保存图片
                    cv2.imwrite(save_img_path, self.draw_img)
                    QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
                else:
                    img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                    # 确保保存目录存在
                    os.makedirs(Config.save_path, exist_ok=True)
                    for file_name in os.listdir(self.org_path):
                        full_path = os.path.join(self.org_path, file_name)
                        if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                            name, end_name = file_name.rsplit(".", 1)
                            save_name = name + '_detect_result.' + end_name
                            save_img_path = os.path.join(Config.save_path, save_name)
                            results = self.model(full_path, conf=self.conf, iou=self.iou)[0]
                            now_img = results.plot()
                            # 保存图片
                            cv2.imwrite(save_img_path, now_img)

                    QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))

        except Exception as e:
            print(f"保存检测结果错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_process_bar(self, cur_num, total):
        try:
            if cur_num == 1:
                self.progress_bar = ProgressBar(self)
                self.progress_bar.show()
            if cur_num >= total:
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.close()
                QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
                return
            if hasattr(self, 'progress_bar') and not self.progress_bar.isVisible():
                # 点击取消保存时，终止进程
                self.btn2Thread_object.stop()
                return
            value = int(cur_num / total * 100)
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(cur_num, total, value)
            QApplication.processEvents()
        except Exception as e:
            print(f"更新进度条错误: {str(e)}")


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, com_text, conf, iou):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.conf = conf
        self.iou = iou
        self.colors = tools.Colors()
        self.is_running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.org_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"视频尺寸: {size}")

            fileName = os.path.basename(self.org_path)
            name, end_name = fileName.split('.')
            save_name = name + '_detect_result.avi'
            save_video_path = os.path.join(Config.save_path, save_name)
            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

            prop = cv2.CAP_PROP_FRAME_COUNT
            total = int(cap.get(prop))
            print("[INFO] 视频总帧数：{}".format(total))
            cur_num = 0

            while cap.isOpened() and self.is_running:
                cur_num += 1
                print('当前第{}帧，总帧数{}'.format(cur_num, total))
                ret, frame = cap.read()
                if ret:
                    results = self.model(frame, conf=self.conf, iou=self.iou)[0]
                    frame = results.plot()
                    out.write(frame)
                    self.update_ui_signal.emit(cur_num, total)
                else:
                    break

            cap.release()
            out.release()

        except Exception as e:
            print(f"视频保存线程错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())