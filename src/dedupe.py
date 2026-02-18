"""
Phase 1: Remove duplicate images using imagededup (PHash).
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import logging
from typing import Set

import yaml
from imagededup.methods import PHash

from .base import BaseFilter, FilterResult


class Dedupe(BaseFilter):
    """Remove exact duplicate images from dataset."""

    name = "dedupe"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.images_to_remove: Set[str] = set()

    def setup(self, input_dir: Path) -> None:
        """Run imagededup once on entire folder to find and mark duplicates for removal."""
        try:
            logging.getLogger("imagededup").setLevel(logging.CRITICAL)
            logging.getLogger("imagededup").propagate = False
            phasher = PHash()
            max_distance = self.config.get("imagededup", {}).get(
                "max_distance_threshold", 10
            )
            prev_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    duplicates = phasher.find_duplicates(
                        image_dir=str(input_dir),
                        max_distance_threshold=max_distance,
                        scores=True,
                    )
            finally:
                logging.disable(prev_disable)

            # Keep first image from each duplicate group, mark rest for removal
            duplicate_groups = []
            seen = set()
            for img, dup_list in duplicates.items():
                if not dup_list:
                    continue
                group = sorted({Path(img).name, *[Path(d[0]).name for d in dup_list]})
                key = tuple(group)
                if key not in seen:
                    duplicate_groups.append(group)
                    seen.add(key)

            if duplicate_groups:
                for dup_group in duplicate_groups:
                    if len(dup_group) > 1:
                        for dup_name in dup_group[1:]:
                            self.images_to_remove.add(Path(dup_name).name)

        except Exception as e:
            print(f"  Warning: Could not process duplicates - {e}")

    def apply(self, image_path: Path) -> FilterResult:
        """Keep image if it's NOT marked for removal as a duplicate."""
        is_duplicate = image_path.name in self.images_to_remove
        return FilterResult(keep=not is_duplicate)
