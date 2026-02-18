"""Phase 4: Filter by age >= min_age using InsightFace."""

import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import yaml

from insightface.app import FaceAnalysis

from .base import BaseFilter, FilterResult

# Suppress InsightFace internal warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


class AgeFilter(BaseFilter):
    """Keep only images with detected face aged >= min_age."""

    name = "age_filter"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = None

    def setup(self, input_dir: Path) -> None:
        """Load InsightFace model once."""
        if FaceAnalysis is None:
            raise ImportError("insightface not installed")
        
        # Suppress InsightFace verbose output
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.model = FaceAnalysis(providers=["CPUExecutionProvider"])
            self.model.prepare(ctx_id=0, det_size=(640, 640))

    def apply(self, image_path: Path) -> FilterResult:
        """Check if image has face with age >= min_age."""
        img = cv2.imread(str(image_path))
        if img is None:
            return FilterResult(keep=False)

        try:
            # InsightFace detects faces and extracts age/gender in one call
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.model.get(img_rgb)

            if len(faces) == 0:
                return FilterResult(keep=False)

            # Get largest face
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

            # Extract age (in years)
            estimated_age = float(face.age)
            min_age = self.config["age"]["min_age"]

            passes_check = estimated_age >= min_age

            return FilterResult(keep=passes_check)
        except Exception:
            return FilterResult(keep=False)
