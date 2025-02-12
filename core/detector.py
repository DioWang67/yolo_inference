
import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
from .utils import ImageUtils

class YOLODetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.colors = colors
        self.image_utils = ImageUtils()
        self.scale_boxes = scale_boxes
        self.non_max_suppression = non_max_suppression
        self.Annotator = Annotator

    def preprocess_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(frame_rgb).to(self.config.device).permute(2, 0, 1).float().unsqueeze(0)
        im /= 255.0
        return im
    
    @staticmethod
    def iou(box1, box2):
        """計算兩個框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection_area / float(box1_area + box2_area - intersection_area)
    

    def check_color_in_range(self, image, box, color_name):

        """
        針對線條特徵進行顏色檢測。
        """
        if color_name not in self.config.color_ranges:
            raise ValueError(f"未知的顏色: {color_name}")

        x1, y1, x2, y2 = box
        region = image[y1:y2, x1:x2]
        
        # 裁切成更窄的條狀
        h, w = region.shape[:2]
        narrow_width = w // 10 
        match_ratios = []

        for i in range(0, w, narrow_width):
            narrow_region = region[:, i:i + narrow_width]
            # HSV 檢測
            hsv_region = cv2.cvtColor(narrow_region, cv2.COLOR_BGR2HSV)
            hsv_mask = np.zeros(narrow_region.shape[:2], dtype=np.uint8)
            for hsv_range in self.config.color_ranges[color_name]['hsv_range']:
                lower = np.array(hsv_range['lower'])
                upper = np.array(hsv_range['upper'])
                mask = cv2.inRange(hsv_region, lower, upper)
                hsv_mask = cv2.bitwise_or(hsv_mask, mask)

            # LAB 檢測
            lab_region = cv2.cvtColor(narrow_region, cv2.COLOR_BGR2LAB)
            lab_lower = np.array(self.config.color_ranges[color_name]['lab_range']['lower'])
            lab_upper = np.array(self.config.color_ranges[color_name]['lab_range']['upper'])
            lab_mask = cv2.inRange(lab_region, lab_lower, lab_upper)

            # 結合結果
            combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            match_ratio = cv2.countNonZero(combined_mask) / combined_mask.size
            match_ratios.append(match_ratio)

        # 返回最高的匹配比例
        best_match_ratio = max(match_ratios) if match_ratios else 0
        # print(f"{color_name} Best Match Ratio: {best_match_ratio:.2f}")
        return best_match_ratio > self.config.color_ranges[color_name].get('confidence', 0) / 100


    def process_detections(self, pred, im, frame):
        """處理檢測結果"""
        pred = self.non_max_suppression(pred, self.config.conf_thres, self.config.iou_thres)
        detections = []
        annotator = self.Annotator(frame, line_width=3, example=str(self.model.names))

        detected_colors = set()
        overlapping_threshold = 0.3

        for det in pred:
            if len(det):
                det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    color_name = self.model.names[int(cls)]
                    if color_name in detected_colors or color_name not in self.config.expected_color_order:
                        continue

                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    if any(self.iou(box, det_box['box']) > overlapping_threshold for det_box in detections):
                        continue

                    if not self.check_color_in_range(frame, box, color_name):
                        continue

                    detections.append({
                        'label': color_name,
                        'confidence': conf.item(),
                        'box': box
                    })
                    
                    color = self.colors(self.config.expected_color_order.index(color_name), True)
                    annotator.box_label(xyxy, f"{color_name} {conf:.2f}", color=color)
                    detected_colors.add(color_name)

        detections = sorted(detections, 
                          key=lambda d: self.config.expected_color_order.index(d['label']))
        return annotator.result(), detections

    def draw_fixed_labels(self, frame: np.ndarray, avg_scores: Dict[str, float]) -> None:
        """在固定位置繪製標籤"""
        label_offset_y = 30
        label_start_y = 30
        label_x_pos = 10

        for i, color_name in enumerate(self.config.expected_color_order):
            label_y_pos = label_start_y + i * label_offset_y
            score = avg_scores.get(color_name, 0)
            label_text = f"{color_name}: {score:.2f}" if score > 0 else f"{color_name}: 0"
            color = self.colors(i, True)
            
            self.image_utils.draw_label(frame, label_text, 
                                      (label_x_pos, label_y_pos), color)
            
    def draw_results(self, frame, result_text, detections):
        result_frame = frame.copy()
        avg_scores = {det['label']: det['confidence'] for det in detections}
        self.draw_fixed_labels(result_frame, avg_scores)
        
        cv2.putText(result_frame, result_text, (230, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 255, 0) if result_text == "PASS" else (0, 0, 255), 3)
        
            
        return result_frame
