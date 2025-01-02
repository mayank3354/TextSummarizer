from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f"Initiating {STAGE_NAME}")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"{STAGE_NAME} completed successfully")

except Exception as e:
    logger.exception(e)
    raise e