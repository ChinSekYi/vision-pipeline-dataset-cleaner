"""Phase 5: Filter advertisements using CLIP visual-semantic matching."""

import warnings
from pathlib import Path

import torch
import yaml
from PIL import Image

try:
    import clip
except ImportError:
    clip = None

from .base import BaseFilter, FilterResult

# Suppress PyTorch MPS pin_memory warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class AdvertisementFilter(BaseFilter):
    """Remove images that are advertisements based on CLIP image-text similarity."""

    name = "advertisement_filter"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        ad_config = self.config.get("advertisement", {})
        self.model = None
        self.preprocess = None
        self.device = ad_config.get("clip_device", "cpu")
        self.model_name = ad_config.get("clip_model", "ViT-B/32")
        self.prompts = [
            ad_config.get(
                "ad_prompt", "a promotional advertisement or marketing image"
            ),
            ad_config.get("natural_prompt", "a candid photo of a person"),
        ]

    def setup(self, input_dir: Path) -> None:
        """Load CLIP model once."""
        if clip is None:
            raise ImportError(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def apply(self, image_path: Path) -> FilterResult:
        """Check if image is NOT an advertisement (keep if not ad)."""
        try:
            # Load and preprocess image
            image = Image.open(str(image_path)).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Encode image and text
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(
                    clip.tokenize(self.prompts).to(self.device)
                )

            # Normalize and compute similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

            # Classify: advertisement if first prompt (ad) has higher similarity
            is_advertisement = similarity[0] > similarity[1]

            # Keep if NOT an advertisement
            return FilterResult(keep=not is_advertisement)

        except Exception:
            # On error, keep the image (don't filter it out)
            return FilterResult(keep=True)
