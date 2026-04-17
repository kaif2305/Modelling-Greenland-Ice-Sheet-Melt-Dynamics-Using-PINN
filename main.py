import argparse
import dagshub
from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from mlProject.pipeline.stage_01_1_data_analysis import DataAnalysisTrainingPipeline
from mlProject.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_03_model_training import ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline

if __name__ == '__main__':
    # 1. Setup Arguments for Ensemble & Supercomputer Control
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["prepare", "train", "evaluate", "all"], 
                        help="Which part of the pipeline to run: 'prepare' (Stages 1-3), 'train' (Stage 4), or 'evaluate' (Stage 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ensemble training")
    args = parser.parse_args()

    # 2. Connect to DagsHub
    dagshub.init(
        repo_owner="23kaif05",
        repo_name="Modelling-Greenland-Ice-Sheet-Melt-Dynamics-Using-PINN",
        mlflow=True
    )

    try:
        # ==========================================
        # PHASE 1: PREPARE (Data Ingestion, Analysis, Transformation)
        # Run this ONCE before submitting the SLURM array
        # ==========================================
        if args.mode in ["prepare", "all"]:
            STAGE_NAME = "Data Ingestion Stage"
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            data_ingestion = DataIngestionPipeline()
            data_ingestion.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

            STAGE_NAME = "Data Analysis Stage"
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            data_analysis = DataAnalysisTrainingPipeline()
            data_analysis.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

            STAGE_NAME = "Data Transformation Stage"
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            data_transformation = DataTransformationTrainingPipeline()
            data_transformation.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # ==========================================
        # PHASE 2: TRAIN (Model Training)
        # Run this via SLURM Array (5 simultaneous jobs)
        # ==========================================
        if args.mode in ["train", "all"]:
            STAGE_NAME = "Model Training Stage (LSTM-PINN)"
            logger.info(f">>>>>> stage {STAGE_NAME} (Seed: {args.seed}) started <<<<<<")
            model_trainer = ModelTrainerTrainingPipeline()
            # Ensure your stage_03_model_training.py main() accepts a seed argument!
            model_trainer.main(seed=args.seed) 
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # ==========================================
        # PHASE 3: EVALUATE (Model Evaluation)
        # Run this ONCE after all training jobs finish
        # ==========================================
        if args.mode in ["evaluate", "all"]:
            STAGE_NAME = "Model Evaluation Stage"
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            model_evaluation = ModelEvaluationTrainingPipeline()
            model_evaluation.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e