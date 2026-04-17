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

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path      # Folder containing the daily CSVs
    transformed_path: Path # Where to save the final merged CSV
    winter_albedo: float   # 0.85
    interpolation_limit: int # 7 days
    stations: list

@dataclass(frozen=True)
class ModelTrainerConfig:
    # 1. Directory and File Paths (from config.yaml)
    root_dir: Path
    clean_data_path: Path
    trained_model_path: Path
    loss_log_path: Path
    
    # 2. PINN Architecture (from params.yaml)
    input_dim: int
    output_dim: int
    hidden_layers: list
    activation: str
    seq_length: int       # NEW: For the LSTM temporal memory window
    
    # 3. Training Hyperparameters (from params.yaml)
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    
    # Note: Physics lambdas removed because the Advanced PINN learns them dynamically!
    
    # 4. Data Schema Setup (from schema.yaml / config)
    target_feature: str
    collocation_flag: str
    input_features: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_dir: Path
    scaler_X_path: Path
    scaler_y_path: Path
    metric_file_name: Path
    mlflow_uri: str
    
    # We need the architecture params to rebuild the model before loading the weights
    input_dim: int
    output_dim: int
    hidden_layers: list
    seq_length: int
    target_feature: str
    collocation_flag: str
    input_features: list