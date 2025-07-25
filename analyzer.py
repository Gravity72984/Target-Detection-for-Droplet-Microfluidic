import torch
import numpy as np
from ultralytics import YOLOv10
from queue import Queue
from threading import Thread
import time
import yaml
from camera import CameraManager
import math

class DropletAnalyzer:
    def __init__(self, data_queue: Queue, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化模型
        self.model = YOLOv10(self.config['model']['weights_path'])
        self.model.to(torch.device(self.config['model']['device']))
        
        # 数据管道
        self.data_queue = data_queue
        self._running = False
        
        # 历史数据缓存
        self.diameter_history = []

    # 在 DropletAnalyzer 类中添加方法
    def _process_frame(self, frame, roi=None):
        """执行单帧分析（支持ROI）"""
        results = self.model(frame, 
                            conf=self.config['model']['confidence'],
                            imgsz=640,
                            verbose=False)
        
        # 获取图像尺寸
        h, w = frame.shape[:2]
        
        # 过滤重复液滴
        valid_boxes = self._filter_duplicate_droplets(results[0].boxes, self.config['processing'].get('duplicate_threshold', 15))
        
        valid_diameters = []
        for box in valid_boxes:
            if int(box.cls) == 0:  # 仅处理液滴类别
                box_data = box.xywh[0].cpu().detach().numpy()
                x_center, y_center, w_box, h_box = box_data
                
                # 计算检测框的边界坐标
                x1 = x_center - w_box/2
                y1 = y_center - h_box/2
                x2 = x_center + w_box/2
                y2 = y_center + h_box/2
                
                # ROI过滤：只保留整个检测框都在ROI区域内的检测结果
                if roi is not None:
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi
                    if not (x1 >= roi_x1 and x2 <= roi_x2 and y1 >= roi_y1 and y2 <= roi_y2):
                        continue  # 跳过不完全在ROI区域内的检测

                # 边界检查
                boundary_threshold = self.config['processing'].get('boundary_threshold', 5)
                if (x1 < boundary_threshold or
                    x2 > w - boundary_threshold or
                    y1 < boundary_threshold or
                    y2 > h - boundary_threshold):
                    continue
                
                diameter = ((w_box + h_box) / 2) * self.config['hardware']['pixel_ratio']
                if (self.config['processing']['min_diameter'] < diameter 
                    < self.config['processing']['max_diameter']):
                    valid_diameters.append(diameter)
        
        return valid_diameters



    def _filter_duplicate_droplets(self, detections, threshold=1):
        """过滤重复检测的液滴"""
        filtered_detections = []
        centers = []

        for box in detections:
            x_center, y_center = box.xywh[0].cpu().numpy()[:2]
            center = (x_center, y_center)

            is_duplicate = False
            for existing_center in centers:
                if self._euclidean_distance(center, existing_center) < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(box)
                centers.append(center)

        return filtered_detections
    

    def _euclidean_distance(self, point1, point2):
        """计算两点间的欧几里得距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _analysis_loop(self):
        """分析线程主循环"""
        cam = CameraManager()
        try:
            while self._running:
                start_time = time.time()
                
                try:
                    frame = cam.get_frame()
                except RuntimeError as e:
                    print(f"视频输入异常: {str(e)}")
                    break
                diameters = self._process_frame(frame)
                
                if diameters:
                    # 计算滑动平均
                    self.diameter_history = (self.diameter_history + diameters)[-self.config['processing']['history_size']:]
                    avg_diameter = np.mean(self.diameter_history)
                    
                    # 推送数据到主程序
                    self.data_queue.put({
                        'timestamp': time.time(),
                        'value': avg_diameter,
                        'unit': 'μm',
                        'frame': frame  # 可选传递视频帧
                    })
                else:
                     print("未检测到有效液滴")
                # 频率控制
                elapsed = time.time() - start_time
                sleep_time = self.config['processing']['detection_interval'] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cam.release()

    def start(self):
        """启动分析线程"""
        if not self._running:
            self._running = True
            self.thread = Thread(target=self._analysis_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """停止分析"""
        self._running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)


    def release(self):
        """释放模型资源"""
        if hasattr(self, 'model'):
            try:
                # 根据使用的深度学习框架进行资源释放
                # 例如PyTorch: del self.model
                pass
            except:
                pass