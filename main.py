"""
Main entry point

Usage:
  make run
  make run-help
"""

import argparse
import sys
from pathlib import Path

from src.age_filter import AgeFilter
from src.dedupe import Dedupe
from src.fullbody_filter import FullBodyFilter
from src.person_detector import PersonDetector

# from src.face_filter import FaceFilter
# from src.age_filter import AgeFilter
# from src.advertisement_filter import AdvertisementFilter
from src.runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(description="Run vision pipeline dataset cleaner")
    parser.add_argument(
        "--input", type=str, default="data/original_raw", help="Input directory"
    )
    parser.add_argument(
        "--output", type=str, default="data/final", help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file path"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = args.config

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Initialize all filters
    filters = [
        Dedupe(config_path),
        PersonDetector(config_path),
        FullBodyFilter(config_path),
        AgeFilter(config_path),
        # AdvertisementFilter(config_path),
    ]

    # Run pipeline
    print(f"Starting pipeline: {input_dir} -> {output_dir}")
    print(f"Filters: {' -> '.join(f.name for f in filters)}")

    runner = PipelineRunner(filters)
    runner.run(input_dir, output_dir)

    print(f"Pipeline complete. Check {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
