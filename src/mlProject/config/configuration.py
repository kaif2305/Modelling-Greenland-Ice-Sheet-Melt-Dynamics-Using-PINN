from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig, 
                                            DataAnalysisConfig)

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