"""Phase 5: Filter advertisements using EasyOCR."""

import warnings
from pathlib import Path

import yaml

try:
    import easyocr
except ImportError:
    easyocr = None

from .base import BaseFilter, FilterResult

# Suppress PyTorch MPS pin_memory warnings from EasyOCR
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class AdvertisementFilter(BaseFilter):
    """Remove images that are advertisements based on OCR text detection."""

    name = "advertisement_filter"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.reader = None
        self.promotional_keywords = [
            "sale",
            "offer",
            "discount",
            "off",
            "limited",
            "buy",
            "price",
            "promotion",
            "deal",
            "promo",
            "save",
            "free",
            "new",
            "now",
            "shop",
            "order",
            "collection",
            "exclusive",
            "%",
            "$",
            "€",
            "£",
        ]

    def setup(self, input_dir: Path) -> None:
        """Load EasyOCR reader once."""
        if easyocr is None:
            raise ImportError("easyocr not installed")
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def apply(self, image_path: Path) -> FilterResult:
        """Check if image is NOT an advertisement (keep if not ad)."""
        try:
            # Read image and extract text
            results = self.reader.readtext(str(image_path))

            if not results:
                # No text detected - not an ad
                return FilterResult(keep=True)

            # Extract all text
            full_text = " ".join([text for (bbox, text, conf) in results])
            full_text_upper = full_text.upper()

            # Check for promotional keywords with confidence threshold
            ocr_confidence_threshold = 0.3

            for keyword in self.promotional_keywords:
                if keyword.upper() in full_text_upper:
                    # Find confidence for this keyword match
                    for bbox, text, conf in results:
                        if keyword.upper() in text.upper():
                            # If high-confidence keyword found, it's an ad
                            if conf >= ocr_confidence_threshold:
                                return FilterResult(keep=False)

            # No high-confidence promotional keywords - not an ad
            return FilterResult(keep=True)

        except Exception:
            # On error, keep the image (don't filter it out)
            return FilterResult(keep=True)
