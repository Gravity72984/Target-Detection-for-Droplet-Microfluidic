import cv2
import yaml

class CameraManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        input_type = config['input']['type']
        self.loop = config['input'].get('video_loop', False)
        self.frame_skip = config['input'].get('frame_skip', 0)
        self._skip_counter = 0


        if input_type == "video":
            # 视频文件模式
            video_path = config['input']['video_path']
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {video_path}")
        

        elif input_type == "camera":
            # 原有摄像头模式
         #camera_index = int(config['hardware']['camera_index'])  # 确保转换为整数
         api_name = config['hardware']['capture_api'].strip().upper()
         api_code = getattr(cv2, f"CAP_{api_name}")
         self.cap = cv2.VideoCapture(
                config['hardware']['camera_index'],
                api_code
         )
        else:
            raise ValueError(f"不支持的输入类型: {input_type}")


    def get_frame(self):
        """获取视频帧（自动处理循环和跳帧）"""
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    raise RuntimeError("视频播放结束")
            
            # 跳帧处理
            if self._skip_counter < self.frame_skip:
                self._skip_counter += 1
                continue
                
            self._skip_counter = 0
            return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

         
      #  try:
            # 动态获取CV2的API常量
       #     api_code = getattr(cv2, f"CAP_{api_name}")
        #except AttributeError:
         #   raise ValueError(f"不支持的视频API: {api_name}")
        
        # 初始化摄像头
        #self.cap = cv2.VideoCapture(camera_index, api_code)
        
        #if not self.cap.isOpened():
         #   raise RuntimeError(f"摄像头初始化失败 | 设备号: {camera_index} | API: {api_name}")

    #def get_frame(self):
     #   ret, frame = self.cap.read()
      #  if not ret:
         #   raise RuntimeError("帧获取失败")
       # return frame

    #def release(self):
     #   self.cap.release()
      #  cv2.destroyAllWindows()
