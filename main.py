
# main.py
import torch
import cv2
import time
import os
import pyfiglet
import numpy as np
from typing import Optional, Tuple

from core.detector import YOLODetector
from core.result_handler import ResultHandler
from core.logger import DetectionLogger
from core.config import DetectionConfig
from core.utils import ImageUtils, DetectionResults
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import scale_boxes
from MvImport.MvCameraControl_class import *
from MVS_camera_control import MVSCamera

class YOLOInference:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化 YOLO 推論系統
        
        Args:
            config_path: 配置文件路徑
        """
        self.logger = DetectionLogger()
        self.config = DetectionConfig.from_yaml(config_path)
        self.model = self._load_model()
        self.detector = YOLODetector(self.model, self.config)
        self.result_handler = ResultHandler(self.config)
        self.camera = MVSCamera(self.config)
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(self.config)

    def _load_model(self) -> DetectMultiBackend:
        """
        載入 YOLO 模型
        
        Returns:
            DetectMultiBackend: 載入的模型實例
        """
        try:
            model = DetectMultiBackend(self.config.weights, device=self.config.device)
            self.logger.logger.info("模型載入成功")
            return model
        except Exception as e:
            self.logger.logger.error(f"模型載入錯誤: {str(e)}")
            raise

    def print_large_text(self, text: str) -> None:
        """
        使用 ASCII art 打印大型文字
        
        Args:
            text: 要打印的文字
        """
        ascii_art = pyfiglet.figlet_format(text)
        print(ascii_art)

    def handle_detection(self, frame: np.ndarray, detections: list, 
                        elapsed_time: float) -> Tuple[str, Optional[np.ndarray]]:
        """
        處理檢測結果
        
        Args:
            frame: 原始幀
            detections: 檢測結果列表
            elapsed_time: 已經過的時間
            
        Returns:
            Tuple[str, Optional[np.ndarray]]: 結果文字和標注後的幀
        """
        result, _ = self.detection_results.evaluate_detection(detections)
        
        if result == "PASS" or elapsed_time >= self.config.timeout:
            status = "PASS" if result == "PASS" else "FAIL"
            result_frame = self.result_handler.save_results(
                frame=frame,
                detections=detections,
                status=status,
                detector=self.detector
            )
            self.print_large_text(status)
            return status, result_frame
            
        return "", None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        處理單個幀
        
        Args:
            frame: 輸入幀
            
        Returns:
            Tuple[np.ndarray, list]: 處理後的幀和檢測結果
        """
        im = self.detector.preprocess_image(frame)
        with torch.no_grad():
            pred = self.model(im)
        return self.detector.process_detections(pred, im, frame)

    def run_inference(self) -> None:
        """執行推論主循環"""
        try:
            # 初始化相機
            if not self.camera.enum_devices():
                raise IOError("無法找到MVS相機")
            if not self.camera.connect_to_camera():
                raise IOError("無法連接MVS相機")
            
            # 狀態變量
            detecting = False  # 是否正在檢測
            paused = False    # 是否暫停
            result_display_start = None  # 結果顯示開始時間
            result_text = ""
            last_result_frame = None
            start_time = None  # 添加 start_time 變量
            
            while True:
                # 獲取並處理幀
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                frame = cv2.resize(frame,self.config.imgsz)
                original_frame = frame.copy()
                
                # 處理按鍵輸入
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                
                # 處理空白鍵
                if key == ord(' '):
                    paused = not paused  # 切換暫停狀態
                    detecting = not paused  # 如果取消暫停，立即開始檢測
                    if detecting:
                        start_time = time.time()  # 開始檢測時設置開始時間
                    result_display_start = None
                    result_text = ""
                    last_result_frame = None
                    continue
                
                # 如果暫停，顯示最後一幀
                if paused:
                    cv2.imshow('YOLOv5 檢測',
                            last_result_frame if last_result_frame is not None else frame)
                    continue
                
                # 自動檢測邏輯
                current_time = time.time()
                
                # 如果有結果正在顯示，檢查是否已經顯示超過3秒
                if result_display_start is not None:
                    if current_time - result_display_start >= 3.0:  # 顯示3秒
                        detecting = True  # 自動開始下一次檢測
                        start_time = time.time()  # 設置新的開始時間
                        result_display_start = None
                        result_text = ""
                        last_result_frame = None
                    else:
                        cv2.imshow('YOLOv5 檢測', last_result_frame)
                        continue
                
                # 執行檢測
                if detecting:
                    if start_time is None:  # 安全檢查
                        start_time = time.time()
                        
                    result_frame, detections = self.process_frame(frame)
                    
                    result_text, new_result_frame = self.handle_detection(
                        original_frame, detections, current_time - start_time)
                    
                    if result_text:
                        last_result_frame = new_result_frame
                        detecting = False
                        result_display_start = current_time  # 設置結果顯示開始時間
                    
                    display_frame = self.detector.draw_results(
                        frame, result_text, detections) if result_text else frame
                    cv2.imshow('YOLOv5 檢測', display_frame)
                else:
                    cv2.imshow('YOLOv5 檢測', frame)
        
        except Exception as e:
            self.logger.logger.error(f"執行過程中發生錯誤: {str(e)}")
            raise
        
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        # 初始化 YOLO 推論
        inference = YOLOInference(r"config.yaml")
        inference.run_inference()
    except Exception as e:
        # 紀錄錯誤日誌
        import traceback
        with open("error.log", "w") as f:
            f.write(traceback.format_exc())
        input(f"程序執行出錯，請檢查 error.log 文件。按 Enter 鍵退出...")
