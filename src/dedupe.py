"""
Phase 1: Remove duplicate images using CleanVision.
"""

import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Set

import yaml
from cleanvision import Imagelab

from .base import BaseFilter, FilterResult


class Dedupe(BaseFilter):
    """Remove exact duplicate images from dataset."""

    name = "dedupe"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.images_to_remove: Set[str] = set()

    def setup(self, input_dir: Path) -> None:
        """Run CleanVision once on entire folder to find and mark duplicates for removal."""
        imagelab = Imagelab(data_path=str(input_dir))

        issue_types = {issue: {} for issue in self.config["cleanvision"]["issue_types"]}

        # Suppress CleanVision output
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            imagelab.find_issues(issue_types)

        # Get duplicate groups from CleanVision
        # Strategy: Keep first image from each duplicate group, remove the rest
        try:
            duplicate_sets = imagelab.info.get("exact_duplicates", {}).get("sets", [])

            if duplicate_sets:
                print(f"  Found {len(duplicate_sets)} duplicate groups")

                # Keep first image from each group, mark rest for removal
                for dup_group in duplicate_sets:
                    if len(dup_group) > 1:
                        for dup_path in dup_group[1:]:
                            self.images_to_remove.add(Path(dup_path).name)

                print(f"  Removing {len(self.images_to_remove)} duplicate images")
            else:
                print(f"  No duplicate groups found")

        except Exception as e:
            print(f"  Warning: Could not process duplicates - {e}")

    def apply(self, image_path: Path) -> FilterResult:
        """Keep image if it's NOT marked for removal as a duplicate."""
        is_duplicate = image_path.name in self.images_to_remove
        return FilterResult(keep=not is_duplicate)
