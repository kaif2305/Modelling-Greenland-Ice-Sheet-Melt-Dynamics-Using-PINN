from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig, 
                                            DataAnalysisConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Create both the root dir and the specific folder for the daily CSVs
        create_directories([config.root_dir, config.local_data_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            base_url=config.base_url,
            stations=list(config.stations),
            local_data_dir=Path(config.local_data_dir)
        )

        return data_ingestion_config
    
    def get_data_analysis_config(self) -> DataAnalysisConfig:
        config = self.config.data_analysis
        schema = self.schema

        # 1. Dynamically build the target variables list from schema.yaml
        feature_columns = list(schema.columns.keys())
        target_column = schema.target_column.name
        
        # Combine and de-duplicate
        target_variables = list(set(feature_columns + [target_column]))

        # 2. Create the directories for analysis reports
        create_directories([Path(config.root_dir), Path(config.reports_dir)])

        # 3. Build the config object
        data_analysis_config = DataAnalysisConfig(
            root_dir=Path(config.root_dir),
            input_data_dir=Path(config.input_data_dir),
            reports_dir=Path(config.reports_dir),
            target_variables=target_variables
        )

        return data_analysis_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(self.config.data_ingestion.local_data_dir),
            transformed_path=Path(config.transformed_path),
            winter_albedo=self.params.data_processing.winter_albedo_fill_value,
            interpolation_limit=self.params.data_processing.max_gap_limit_days,
            stations=list(self.config.data_ingestion.stations)
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_training
        schema = self.schema
        
        create_directories([config.root_dir])

        # Dynamically extract input features from the schema
        target_col = schema.target_column.name
        all_columns = list(schema.columns.keys())
        input_feats = [col for col in all_columns if col not in [target_col, "station_name"]]

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            clean_data_path=Path(config.clean_data_path),
            trained_model_path=Path(config.trained_model_path),
            loss_log_path=Path(config.loss_log_path),
            input_dim=params.input_dim,
            output_dim=params.output_dim,
            hidden_layers=list(params.hidden_layers),
            activation=params.activation,
            seq_length=params.seq_length,
            epochs=params.epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            optimizer=params.optimizer,
            target_feature=target_col,
            collocation_flag="t_surf_is_collocation",
            input_features=input_feats
        )

        return model_trainer_config