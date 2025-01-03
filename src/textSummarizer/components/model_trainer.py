from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.config = model_trainer_config
        

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        dataset = load_from_disk(self.config.data_path)
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            warmup_steps=self.config.warmup_step,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.root_dir,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["test"],
            eval_dataset=dataset["validation"]
        )

        trainer.train()

        ## Save model
        model.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))