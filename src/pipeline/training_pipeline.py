# Pipeline for training
# src/pipeline/train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logger
import numpy as np
from sklearn.model_selection import train_test_split

def run_training_pipeline():
    ingestion = DataIngestion()
    transformation = DataTransformation()
    trainer = ModelTrainer()
    evaluator = ModelEvaluation()

    train_path, test_path = ingestion.initiate_data_ingestion()
    import pandas as pd
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = np.log1p(train_df['price'])
    X, label_encoders, tfidf = transformation.preprocess(train_df, fit=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    rmse, model_path = trainer.train(X_tr, y_tr, X_val, y_val)
    evaluator.evaluate(model_path, X_val, y_val)

    logger.info(f"✅ Pipeline completed successfully. RMSE: {rmse}")
