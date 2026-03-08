from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionPipeline
# import dagshub
import mlflow

# dagshub.init(
#     repo_owner="23kaif05",
#     repo_name="Modelling-Greenland-Ice-Sheet-Melt-Dynamics-Using-PINN",
#     mlflow=True
# )

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e