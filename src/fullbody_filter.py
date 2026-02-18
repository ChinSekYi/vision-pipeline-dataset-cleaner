"""Phase 3: Validate full body visibility using YOLOv8-Pose."""
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from .base import BaseFilter, FilterResult


class FullBodyFilter(BaseFilter):
    """Keep only images with full body (head + legs) visible."""

    name = "fullbody_filter"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = None

    def setup(self, input_dir: Path) -> None:
        """Load YOLOv8-Pose model once."""
        model_path = self.config["models"]["pose"]
        self.model = YOLO(model_path)

    def apply(self, image_path: Path) -> FilterResult:
        """Check if image shows full body (head + legs visible)."""
        img = cv2.imread(str(image_path))
        if img is None:
            return FilterResult(keep=False)

        # Run pose detection
        results = self.model.predict(source=img, verbose=False)
        result = results[0]

        if not result.keypoints or len(result.keypoints) == 0:
            return FilterResult(keep=False)

        # Get keypoints for first person
        keypoints = result.keypoints[0]
        keypoint_threshold = self.config["pose"]["keypoint_threshold"]

        # COCO keypoint indices:
        # 0=nose, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
        nose_conf = keypoints.conf[0, 0].item() if keypoints.conf is not None else 0
        left_knee_conf = (
            keypoints.conf[0, 13].item() if keypoints.conf is not None else 0
        )
        right_knee_conf = (
            keypoints.conf[0, 14].item() if keypoints.conf is not None else 0
        )
        left_ankle_conf = (
            keypoints.conf[0, 15].item() if keypoints.conf is not None else 0
        )
        right_ankle_conf = (
            keypoints.conf[0, 16].item() if keypoints.conf is not None else 0
        )

        # Check pose: Head + Legs visible
        has_head = nose_conf > keypoint_threshold
        has_legs = (
            left_knee_conf > keypoint_threshold
            or right_knee_conf > keypoint_threshold
            or left_ankle_conf > keypoint_threshold
            or right_ankle_conf > keypoint_threshold
        )
        is_fullbody = has_head and has_legs

        return FilterResult(keep=is_fullbody)
