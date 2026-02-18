from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path


@dataclass
class FilterResult:
    # every filter returns a FilterResult
    keep: bool
    metadata: Dict[str, Any]
    reason: str | None = None


class BaseFilter: # per image
    name: str = "base"

    def apply(self, image_path: Path, metadata: Dict[str, Any]) -> FilterResult:
        raise NotImplementedError