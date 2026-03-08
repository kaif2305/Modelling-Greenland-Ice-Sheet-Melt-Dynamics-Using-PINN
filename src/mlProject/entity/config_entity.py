from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    base_url: str
    stations: List[str]
    local_data_dir: Path
