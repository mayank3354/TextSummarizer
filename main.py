from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f"Initiating {STAGE_NAME}")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"{STAGE_NAME} completed successfully")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f"Initiating {STAGE_NAME}")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"{STAGE_NAME} completed successfully")

except Exception as e:
    logger.exception(e)
    raise e
