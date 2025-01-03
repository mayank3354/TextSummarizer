from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainerConfig
from src.textSummarizer.logging import logger

from src.textSummarizer.components.model_trainer import ModelTrainer


class ModelTrainerTrainingPipeline:
    def __init__(self,):
        pass
    def initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config)
        model_trainer.train()
        logger.info("Model Training process completed.")