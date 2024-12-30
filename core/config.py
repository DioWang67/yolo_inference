# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: str
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    color_match_threshold: float = 0.2
    imgsz: Tuple[int, int] = (1280, 1280)
    expected_color_order: List[str] = None
    color_ranges: Dict[str, List[List[Tuple[int, int, int]]]] = None
    timeout: int = 2
    use_hsv_detection: bool = True
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 640  # 設定解析度寬度
    height: int = 640 # 設定解析度高度
    MV_CC_GetImageBuffer_nMsec : int = 10000
    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            print("Loaded YAML:", config_dict)  # 調試輸出
        return cls(**config_dict)

