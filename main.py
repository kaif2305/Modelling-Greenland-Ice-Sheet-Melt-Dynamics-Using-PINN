from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from mlProject.pipeline.stage_01_1_data_analysis import DataAnalysisTrainingPipeline
from mlProject.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_03_model_training import ModelTrainerTrainingPipeline

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

STAGE_NAME = "Data Analysis Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_analysis_pipeline = DataAnalysisTrainingPipeline()
    data_analysis_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training Stage (LSTM-PINN)"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer_pipeline = ModelTrainerTrainingPipeline()
    model_trainer_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e