from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger

class DataTransformationTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.clean_and_combine()

if __name__ == '__main__':
    try:
        logger.info(">>>>>> stage Data Transformation started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(">>>>>> stage Data Transformation completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e