from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_training import ModelTrainer
from mlProject import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    # ADD THE SEED PARAMETER HERE
    def main(self, seed=42):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        # PASS THE SEED INTO THE TRAIN FUNCTION
        model_trainer.train(seed=seed)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e