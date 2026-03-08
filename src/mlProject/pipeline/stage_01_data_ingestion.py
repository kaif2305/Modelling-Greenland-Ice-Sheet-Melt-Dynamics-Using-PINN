from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_ingestion import DataIngestion
from mlProject import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        # 1. Initialize the configuration manager
        config = ConfigurationManager()
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # 2. Get the specific config for data ingestion
        data_ingestion_config = config.get_data_ingestion_config()

        # 3. Initialize the ingestion component
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # 4. Execute the actual data fetching and resampling
        # This replaces the old zip download and extraction steps
        data_ingestion.download_and_resample_to_daily()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

if __name__ == "__main__":
    try:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e