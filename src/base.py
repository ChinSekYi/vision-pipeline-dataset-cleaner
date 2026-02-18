"""
Runs the pipeline by applying each filter to a batch of images.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilterResult:
    """Result for a single image after a filter is applied."""

    keep: bool


class BaseFilter:  # per image
    name: str = "base"

    def apply(self, image_path: Path) -> FilterResult:
        """Apply the filter to one image."""
        raise NotImplementedError
