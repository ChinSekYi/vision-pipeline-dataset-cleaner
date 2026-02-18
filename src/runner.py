"""
Runs the pipeline by applying each filter to a batch of images.
"""

from pathlib import Path
from typing import List

from .base import BaseFilter, FilterResult


class PipelineRunner:
    """Batch-style filter runner for a folder of images."""

    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters

    def run(self, input_dir: Path, output_dir: Path) -> None:
        """Apply filters in order and copy final kept images to output."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Call setup() on filters that have it
        for f in self.filters:
            if hasattr(f, "setup"):
                f.setup(input_dir)

        images = sorted(input_dir.glob("*.png"))
        current_images = images  # working list of images still in the pipeline
        print(f"\nPhase 0: Original")
        print(f"  Total images: {len(current_images)}\n")

        # batch-style: each filter processes all images in sequence
        for i, f in enumerate(self.filters, 1):
            print(f"Running Phase {i}: {f.name}...")
            kept: List[Path] = []
            for img_path in current_images:
                result: FilterResult = f.apply(img_path)
                if result.keep:
                    kept.append(img_path)

            filtered_count = len(current_images) - len(kept)
            print(f"Phase {i}: {f.name}")
            print(f"  Images after filter: {len(kept)}")
            print(f"  Filtered out: {filtered_count}\n")
            current_images = kept

        # only images that passed all filters get copied to the final folder
        for img_path in current_images:
            target = output_dir / img_path.name
            if not target.exists():
                target.write_bytes(img_path.read_bytes())

        print(f"Final: {len(current_images)} images copied to {output_dir}")
