import tkinter as tk
from tkinter import ttk
import time
import threading
import socket
import numpy as np
from queue import Queue
import cv2
import yaml
from video_panel import VideoPanel
# from line_profiler import LineProfiler #导入线程分析

class PIDController:
    """PID控制器（带抗积分饱和）"""
    def __init__(self, Kp, Ki, Kd, deadband=1, output_limits=(0, 2000)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.lock = threading.Lock()
        self.deadband = deadband
        self.output_limits = output_limits  # 添加输出限制范围
        self.reset()

    def reset(self):
        with self.lock:
            self.last_error = 0.0
            self.integral = 0.0
            self.last_time = time.time()
            self.saturated = False  # 添加饱和状态标志

    def compute(self, setpoint, measured_value):
        with self.lock:
            now = time.time()
            dt = now - self.last_time
            
            error = setpoint - measured_value
            if abs(error) < self.deadband:
                self.last_time = now  # 更新时间防止dt过大
                return 0.0
            
            # 比例项
            P = self.Kp * error
            
            # 积分项（带抗饱和）
            if not self.saturated:  # 仅在非饱和状态更新积分
                self.integral += error * dt
            I = self.Ki * self.integral
            
            # 微分项
            if dt > 0:
                derivative = (error - self.last_error) / dt
            else:
                derivative = 0
            D = self.Kd * derivative
            
            # 计算原始输出
            raw_output = P + I + D
            
            # 检测饱和状态
            min_output, max_output = self.output_limits
            is_saturated = raw_output <= min_output or raw_output >= max_output
            
            # 当从饱和状态转为非饱和状态时重置积分
            if self.saturated and not is_saturated:
                self.integral = 0  # 重置积分项
                self.saturated = False
            
            # 保存饱和状态
            self.saturated = is_saturated
            
            # 保存状态
            self.last_error = error
            self.last_time = now
            
            # 限制输出
            return max(min_output, min(max_output, raw_output))

class VideoInputManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.input_type = self.config['input']['type']
        self._init_input_source()
        
    def _init_input_source(self):
        """初始化输入源"""
        if self.input_type == "video":
            self.cap = cv2.VideoCapture(self.config['input']['video_path'])
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self.config['input']['video_path']}")
            self.loop = self.config['input']['loop']
        elif self.input_type == "camera":
            api_name = self.config['hardware']['capture_api'].upper()
            api_code = getattr(cv2, f"CAP_{api_name}")
            self.cap = cv2.VideoCapture(
                self.config['hardware']['camera_index'],
                api_code
            )
        else:
            raise ValueError(f"不支持的输入类型: {self.input_type}")

        self.frame_skip = self.config['input'].get('frame_skip', 0)
        self._skip_counter = 0

    def get_frame(self):
        """获取视频帧（自动处理循环和跳帧）"""
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.input_type == "video" and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    raise RuntimeError("视频播放结束" if self.input_type == "video" 
                                      else "摄像头读取失败")
            
            if not ret and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # 验证重置是否成功
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                 raise RuntimeError("视频循环失败")
                continue



            # 跳帧处理
            if self._skip_counter < self.frame_skip:
                self._skip_counter += 1
                continue
                
            self._skip_counter = 0
            return frame

    def release(self):
        self.cap.release()



class ControlSystem(tk.Tk):
    """主控制系统(集成GUI和通信接口)"""
    def __init__(self, data_queue: Queue):
        super().__init__()
        self.data_queue = data_queue
        self.pid_enabled = False
        self.title("微滴智能控制系统beta0.5")
        self.geometry("1024x768")
        self.video_source = VideoInputManager()
        self.show_video = tk.BooleanVar(value=True)  # 默认显示视频
        
        # 延后初始化非核心组件
        #self.after(100, self._delayed_init)
        # 初始化参数
        self.target_size = 100.0     # 目标粒径（μm）
        self.continuous_pressure = 100.0  # 连续相压力（mbar）
        self.dispersed_pressure = 100.0   # 分散相压力（mbar）

        # 初始化性能变量
        self.fps = 0.0  
        self.processing_latency = 0.0  

        # 获取输入源类型
        self.input_type = self.video_source.input_type
        print(f"输入源类型: {self.input_type}")

        # 分析器初始化
        from analyzer import DropletAnalyzer  # 确保模块正确导入
        self.analyzer = DropletAnalyzer(
            data_queue=self.data_queue,
            config_path="config.yaml"
        )
        # 初始化性能分析器
        #self.lp = LineProfiler()
        #self.lp.add_function(self.control_loop_wrapper)
        #self.lp.add_function(self.analyzer._process_frame)
        
        # 初始化PID控制器
        self.continuous_pid = PIDController(
            0.00, 0.01, 0.00, 
            deadband=1,
            output_limits=(-100, 100)  # 压力调整范围设为±100
        )
        self.dispersed_pid = PIDController(
            0.00, 0.01, 0.00,
            deadband=1,
            output_limits=(-100, 100)
        )
        
        # 创建UI组件
        self.create_widgets()
        
        # 压力安全范围
        self.pressure_limits = (0.0, 2000.0)
        # 初始化窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    #def _delayed_init(self):
        #"""延迟加载视频面板等重型组件"""
        #self.video_panel = VideoPanel(self)
        #self.video_panel.pack()

    def on_close(self):
        """窗口关闭事件处理"""
        print("\n正在关闭程序...")
        # 停止控制循环
        self.pid_enabled = False
        # 等待控制线程完成
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        # 输出最终分析报告
        #self.lp.print_stats()
        self.destroy()

    def control_loop_wrapper(self):
        """用于性能分析的包装函数"""
        self.control_loop()

    def start_profiling(self):
        """启动带性能分析的控制循环"""
        self.control_loop()  # 直接调用控制循环，不使用性能分析
        # self.lp.runcall(self.control_loop_wrapper)


    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建主框架（左右分栏）
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 右侧视频面板
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========================
        # 左侧控制区域
        # ========================
        
        # 显示设置
        display_frame = ttk.LabelFrame(left_panel, text="显示设置")
        display_frame.pack(padx=5, pady=5, fill=tk.X)
        
        ttk.Checkbutton(
            display_frame,
            text="显示实时视频",
            variable=self.show_video,
            command=self._toggle_video_display
        ).pack(side=tk.LEFT)
        
        # 参数输入区
        control_frame = ttk.LabelFrame(left_panel, text="控制参数")
        control_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 目标粒径
        ttk.Label(control_frame, text="目标粒径 (μm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.target_entry = ttk.Entry(control_frame, width=10)
        self.target_entry.insert(0, "150.0")
        self.target_entry.grid(row=0, column=1, pady=2)
        
        # PID参数
        pid_params = [
            ("Kp", "0"), ("Ki", "0.01"), ("Kd", "0")
        ]
        for i, (label, default) in enumerate(pid_params, start=1):
            ttk.Label(control_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(control_frame, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=2)
            setattr(self, f"{label.lower()}_entry", entry)
        
        # 控制按钮
        self.toggle_btn = ttk.Button(
            control_frame, 
            text="启动智能控制", 
            command=self.toggle_pid
        )
        self.toggle_btn.grid(row=4, columnspan=2, pady=5)
        
        # 压力控制区域
        pressure_frame = ttk.LabelFrame(left_panel, text="压力控制")
        pressure_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 连续相压力控制
        ttk.Label(pressure_frame, text="连续相压力 (mbar):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.continuous_pressure_entry = ttk.Entry(pressure_frame, width=10)
        self.continuous_pressure_entry.insert(0, "100.0")
        self.continuous_pressure_entry.grid(row=0, column=1, pady=2)
        
        # 分散相压力控制
        ttk.Label(pressure_frame, text="分散相压力 (mbar):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dispersed_pressure_entry = ttk.Entry(pressure_frame, width=10)
        self.dispersed_pressure_entry.insert(0, "100.0")
        self.dispersed_pressure_entry.grid(row=1, column=1, pady=2)
        
        # 设置压力按钮
        self.set_pressure_btn = ttk.Button(
            pressure_frame, 
            text="设置压力", 
            command=self.set_pressures
        )
        self.set_pressure_btn.grid(row=2, columnspan=2, pady=5)
        
        # 实时状态区域
        status_frame = ttk.LabelFrame(left_panel, text="实时状态")
        status_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 压力显示
        ttk.Label(status_frame, text="连续相压力:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.continuous_label = ttk.Label(status_frame, text="100.0 mbar")
        self.continuous_label.grid(row=0, column=1, pady=2)
        
        ttk.Label(status_frame, text="分散相压力:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dispersed_label = ttk.Label(status_frame, text="100.0 mbar")
        self.dispersed_label.grid(row=1, column=1, pady=2)
        
        # 通信状态
        self.comm_status = ttk.Label(status_frame, text="通信状态: 未连接", foreground="red")
        self.comm_status.grid(row=2, columnspan=2, pady=2)
        
        # ========================
        # 右侧视频区域
        # ========================
        self.right_panel = ttk.Frame(main_frame)  # 添加 self. 前缀
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 视频显示区域
        self.video_panel = VideoPanel(right_panel)
        self.video_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 启动控制线程
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def set_pressures(self):
        """设置新的压力值"""
        try:
            # 获取输入的压力值
            new_continuous = float(self.continuous_pressure_entry.get())
            new_dispersed = float(self.dispersed_pressure_entry.get())
            
            # 确保在安全范围内
            min_p, max_p = self.pressure_limits
            if min_p <= new_continuous <= max_p and min_p <= new_dispersed <= max_p:
                self.continuous_pressure = new_continuous
                self.dispersed_pressure = new_dispersed
                
                # 更新UI显示
                self.update_display()
                
                # 发送新的压力命令
                self.send_pressure_command()
                
                self.comm_status.config(text="压力设置成功", foreground="green")
            else:
                self.comm_status.config(text="压力超出安全范围", foreground="red")
        except ValueError:
            self.comm_status.config(text="请输入有效的数字", foreground="red")

    def _toggle_video_display(self):
        """切换视频显示状态"""
        if self.show_video.get():
           self.video_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        else:
            self.video_panel.pack_forget()

    def toggle_pid(self):
        """切换PID控制状态"""
        self.pid_enabled = not self.pid_enabled
        if self.pid_enabled:
            # 获取参数
            self.target_size = float(self.target_entry.get())
            kp = float(self.kp_entry.get())
            ki = float(self.ki_entry.get())
            kd = float(self.kd_entry.get())
        
            # 重置控制器 (添加必要的参数)
            self.continuous_pid = PIDController(
                kp, ki, kd, 
                deadband=1,
                output_limits=(-100, 100)  # 添加输出限制
            )
            self.dispersed_pid = PIDController(
                kp, ki, kd,
                deadband=1,
                output_limits=(-100, 100)
            )
        
            self.toggle_btn.config(text="停止智能控制")
            self.comm_status.config(text="通信状态: 模拟模式", foreground="green")
        else:
            self.toggle_btn.config(text="启动智能控制")
            self.comm_status.config(text="通信状态: 已断开", foreground="red")
    

    def control_loop(self):
     """主控制循环（支持视频/摄像头输入）"""
     from analyzer import DropletAnalyzer
     last_update_time = time.time()
     frame_count = 0
     self.running = True  # 新增运行状态标志
     try:
        while self.running:
            # 对于摄像头输入源，无论PID是否开启，都处理帧
            if self.input_type == "camera" or self.pid_enabled:
                # ========================
                # 1. 获取输入帧
                # ========================
                try:
                    frame = self.video_source.get_frame()
                    frame_count += 1
                except RuntimeError as e:
                    print(f"输入源异常: {str(e)}")
                    if self.input_type == "camera":
                            time.sleep(0.5)  # 摄像头异常时稍作等待
                            continue
                    else:
                        self.toggle_pid()
                        continue
                
                # ========================
                # 2. 执行液滴检测（仅在启用PID时）
                # ========================
                if self.pid_enabled:
                        start_process = time.time()
                        diameters = self.analyzer._process_frame(frame)
                        process_time = time.time() - start_process
                else:
                        diameters = None
                        process_time = 0
                
                # ========================
                # 3. 计算PID控制量（仅在启用PID时）
                # ========================
                if self.pid_enabled and diameters:
                    current_size = np.mean(diameters)
                    
                    # 连续相PID计算（反比关系）
                    cont_output = self.continuous_pid.compute(
                        self.target_size, current_size
                    )
                    # 分散相PID计算（正比关系）
                    disp_output = self.dispersed_pid.compute(
                        self.target_size, current_size
                    )
                    
                    # ========================
                    # 4. 更新压力值
                    # ========================
                    self.continuous_pressure = np.clip(
                        self.continuous_pressure - cont_output,
                        *self.pressure_limits
                    )
                    self.dispersed_pressure = np.clip(
                        self.dispersed_pressure + disp_output,
                        *self.pressure_limits
                    )
                    
                    # ========================
                    # 5. 发送控制指令
                    # ========================
                    self.send_pressure_command()
                
                # ========================
                # 6. 更新性能监控数据
                # ========================
                current_time = time.time()
                time_diff = current_time - last_update_time
                if time_diff >= 1.0:  # 每秒更新一次性能数据
                    self.fps = frame_count / time_diff
                    self.processing_latency = process_time * 1000  # ms
                    frame_count = 0
                    last_update_time = current_time
                    
                    # 更新UI显示
                    self.after(0, self.update_display)
                
                # ========================
                # 7. 更新实时显示
                # ========================
                if self.show_video.get():
                    display_frame = self._annotate_frame(frame.copy(), diameters)
                    self.after(0, lambda: self.video_panel.update_frame(display_frame))
            else:
                time.sleep(0.1)

     except Exception as e:
        print(f"控制循环异常终止: {str(e)}")
        self.toggle_pid()
     #finally: 
         #self.lp.print_stats()
         #pass

    def _annotate_frame(self, frame, diameters):
        """在视频帧上添加标注信息"""
        # 确保所有属性都已初始化
        fps_display = getattr(self, 'fps', 0.0)
        latency_display = getattr(self, 'processing_latency', 0.0)
        continuous_pressure = getattr(self, 'continuous_pressure', 100.0)
        dispersed_pressure = getattr(self, 'dispersed_pressure', 100.0)
        
        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 左上角显示性能数据
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        if self.pid_enabled:  # 仅在PID启用时显示延迟
            cv2.putText(frame, f"Latency: {latency_display:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
        # 右上角显示压力信息
        cv2.putText(frame, f"Continuous: {continuous_pressure:.1f}mbar", (width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Dispersed: {dispersed_pressure:.1f}mbar", (width - 300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
        # 左下角显示检测结果
        if diameters is not None and diameters:  # 检查diameters不为None且不为空
            avg_size = np.mean(diameters)
            cv2.putText(frame, f"Avg Size: {avg_size:.2f}μm", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Droplets: {len(diameters)}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        elif self.input_type == "camera" and not self.pid_enabled:
            cv2.putText(frame, "Camera: Streaming", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # 右下角显示状态信息
        cv2.putText(frame, "PID: " + ("ON" if self.pid_enabled else "OFF"), 
                   (width - 150, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.pid_enabled else (0, 0, 255), 2)
        
        return frame
    
    def update_perf_display(self):
        """更新性能监控显示"""
        self.status_bar.config(text=
            f"处理性能 | FPS: {self.fps:.1f} "
            f"延迟: {self.processing_latency:.1f}ms "
            f"压力范围: {self.continuous_pressure:.1f}-{self.dispersed_pressure:.1f}mbar"
        )

    
    
    def update_display(self):
        """更新界面显示"""
        # 获取饱和状态
        cont_saturated = self.continuous_pid.saturated
        disp_saturated = self.dispersed_pid.saturated
    
        # 更新标签（添加颜色指示）
        cont_text = f"{self.continuous_pressure:.1f} mbar"
        disp_text = f"{self.dispersed_pressure:.1f} mbar"
        if cont_saturated:
            self.continuous_label.config(text=cont_text, foreground="red")
        else:
            self.continuous_label.config(text=cont_text, foreground="black")
    
        if disp_saturated:
            self.dispersed_label.config(text=disp_text, foreground="red")
        else:
            self.dispersed_label.config(text=disp_text, foreground="black")
    
    def send_pressure_command(self):
        """压力值发送接口（模拟实现）"""
        try:
            # 实际通信时可在此处实现具体协议
            command = f"CP{self.continuous_pressure:.1f},DP{self.dispersed_pressure:.1f}\n"
            
            # 模拟网络发送
            # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
               #s.connect(('192.168.1.100', 502))
               #s.sendall(command.encode())
                
            self.comm_status.config(text="通信状态: 已发送", foreground="blue")
        except Exception as e:
            self.comm_status.config(text=f"通信失败: {str(e)}", foreground="red")


if __name__ == "__main__":
    # 初始化数据管道
    data_queue = Queue(maxsize=10)
    
    # 启动控制系统
    app = ControlSystem(data_queue)
    
    # 模拟数据输入（实际应与检测模块对接）
    def mock_data_input():
        while True:
            data_queue.put({
                'timestamp': time.time(),
                'value': 150 + 20 * np.sin(time.time()),
                'unit': 'μm'
            })
            time.sleep(1)
    
    #threading.Thread(target=mock_data_input, daemon=True).start()
    
    app.mainloop()


