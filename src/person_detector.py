"""Phase 2: Detect persons in images using YOLOv8."""

from pathlib import Path

import yaml
from ultralytics import YOLO

from .base import BaseFilter, FilterResult


class PersonDetector(BaseFilter):
    """Keep only images with at least one person detected."""

    name = "person_detector"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = None

    def setup(self, input_dir: Path) -> None:
        """Load model once."""
        model_path = self.config["models"]["yolo"]
        self.model = YOLO(model_path)

    def apply(self, image_path: Path) -> FilterResult:
        """Check if image contains at least one person."""
        # Run YOLO inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.config["yolo"]["confidence_threshold"],
            verbose=False,
        )
        result = results[0]

        # Filter to person class only (class 0 in COCO)
        person_boxes = result.boxes[
            result.boxes.cls == self.config["yolo"]["person_class_id"]
        ]
        has_person = len(person_boxes) > 0

        return FilterResult(keep=has_person)
