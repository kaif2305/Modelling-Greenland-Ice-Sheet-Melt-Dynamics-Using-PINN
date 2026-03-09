from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    base_url: str
    stations: List[str]
    local_data_dir: Path

@dataclass(frozen=True)
class DataAnalysisConfig:
    root_dir: Path
    input_data_dir: Path
    reports_dir: Path
    target_variables: List[str]
