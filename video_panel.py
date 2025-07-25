# video_panel.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class VideoPanel(tk.Frame):
    """实时视频显示组件（支持区域选择）"""
    def __init__(self, master):
        super().__init__(master)
        
        # 创建控制按钮区域
        control_frame = tk.Frame(self)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.roi_enabled = tk.BooleanVar(value=False)
        self.roi_checkbox = tk.Checkbutton(
            control_frame, 
            text="启用区域识别", 
            variable=self.roi_enabled,
            command=self.toggle_roi_mode
        )
        self.roi_checkbox.pack(side=tk.LEFT)
        
        self.confirm_btn = tk.Button(
            control_frame, 
            text="确认区域", 
            command=self.confirm_roi,
            state=tk.DISABLED
        )
        self.confirm_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(
            control_frame, 
            text="重新选择", 
            command=self.reset_roi,
            state=tk.DISABLED
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建视频显示区域
        self.video_label = tk.Label(self, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ROI相关变量
        self.roi_start = None
        self.roi_end = None
        self.roi_confirmed = False
        self.drawing_roi = False
        self.current_roi = None  # (x1, y1, x2, y2)
        
        # 图像相关
        self.image = None
        self.last_frame = None
        self.display_scale = 1.0
        self.original_size = None
        
        # 绑定鼠标事件
        self.video_label.bind("<Button-1>", self.on_mouse_down)
        self.video_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def toggle_roi_mode(self):
        """切换ROI模式"""
        if self.roi_enabled.get():
            self.confirm_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
            self.roi_confirmed = False
            self.current_roi = None
        else:
            self.confirm_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.DISABLED)
            self.roi_confirmed = False
            self.current_roi = None
            self.roi_start = None
            self.roi_end = None
    
    def on_mouse_down(self, event):
        """鼠标按下事件"""
        if not self.roi_enabled.get() or self.roi_confirmed:
            return
        
        self.roi_start = (event.x, event.y)
        self.drawing_roi = True
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件"""
        if not self.roi_enabled.get() or self.roi_confirmed or not self.drawing_roi:
            return
        
        self.roi_end = (event.x, event.y)
    
    def on_mouse_up(self, event):
        """鼠标释放事件"""
        if not self.roi_enabled.get() or self.roi_confirmed or not self.drawing_roi:
            return
        
        self.roi_end = (event.x, event.y)
        self.drawing_roi = False
    
    def confirm_roi(self):
        """确认选择的ROI区域"""
        if self.roi_start and self.roi_end and self.original_size:
            # 转换显示坐标到原始图像坐标
            x1 = int(min(self.roi_start[0], self.roi_end[0]) / self.display_scale)
            y1 = int(min(self.roi_start[1], self.roi_end[1]) / self.display_scale)
            x2 = int(max(self.roi_start[0], self.roi_end[0]) / self.display_scale)
            y2 = int(max(self.roi_start[1], self.roi_end[1]) / self.display_scale)
            
            # 确保坐标在图像范围内
            h, w = self.original_size[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                self.current_roi = (x1, y1, x2, y2)
                self.roi_confirmed = True
                print(f"ROI confirmed: {self.current_roi}")
    
    def reset_roi(self):
        """重置ROI选择"""
        self.roi_start = None
        self.roi_end = None
        self.roi_confirmed = False
        self.current_roi = None
        self.drawing_roi = False
    
    def get_roi(self):
        """获取当前ROI区域"""
        if self.roi_enabled.get() and self.roi_confirmed and self.current_roi:
            return self.current_roi
        return None
    
    def crop_frame_to_roi(self, frame):
        """将帧裁剪到ROI区域"""
        roi = self.get_roi()
        if roi:
            x1, y1, x2, y2 = roi
            return frame[y1:y2, x1:x2]
        return frame
    
    def update_frame(self, frame):
        """更新显示帧"""
        if self.last_frame is not None and self.last_frame.shape == frame.shape:
            if np.array_equal(self.last_frame, frame):
                return
        
        self.last_frame = frame.copy()
        self.original_size = frame.shape
        
        # 转换OpenCV BGR格式为RGB
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 绘制ROI选择框
        if self.roi_enabled.get():
            display_frame = self._draw_roi_overlay(display_frame)
        
        # 调整尺寸以适应窗口
        display_frame, self.display_scale = self._resize_image_with_scale(
            display_frame, target_width=self.video_label.winfo_width()
        )
        
        # 转换为Tkinter兼容格式
        self.image = ImageTk.PhotoImage(image=Image.fromarray(display_frame))
        self.video_label.configure(image=self.image)
    
    def _draw_roi_overlay(self, img):
        """在图像上绘制ROI覆盖层"""
        overlay = img.copy()
        
        # 如果已确认ROI，绘制绿色框
        if self.roi_confirmed and self.current_roi:
            x1, y1, x2, y2 = self.current_roi
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, "ROI Active", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 如果正在绘制，绘制红色框
        elif self.roi_start and self.roi_end:
            # 转换到原始图像坐标
            x1 = int(min(self.roi_start[0], self.roi_end[0]) / self.display_scale)
            y1 = int(min(self.roi_start[1], self.roi_end[1]) / self.display_scale)
            x2 = int(max(self.roi_start[0], self.roi_end[0]) / self.display_scale)
            y2 = int(max(self.roi_start[1], self.roi_end[1]) / self.display_scale)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay, "Drawing ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return overlay
    
    def _resize_image_with_scale(self, img, target_width=640):
        """保持比例调整图像尺寸并返回缩放比例"""
        h, w = img.shape[:2]
        
        if target_width <= 0 or w <= 0:
            return img, 1.0
            
        scale = target_width / w
        new_height = int(h * scale)
        
        if new_height <= 0:
            return img, 1.0
            
        resized = cv2.resize(img, (target_width, new_height))
        return resized, scale
