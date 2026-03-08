from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import DataIngestionConfig

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