from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_analysis import DataAnalysis
from mlProject import logger

STAGE_NAME = "Data Analysis Stage"

class DataAnalysisTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_analysis_config = config.get_data_analysis_config()
        data_analysis = DataAnalysis(config=data_analysis_config)
        # 1. This generates the JSON report
        data_analysis.analyze_missing_values()
        
        # 2. ADD THIS LINE to generate and save the graphs!
        data_analysis.generate_research_graphs()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataAnalysisTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e