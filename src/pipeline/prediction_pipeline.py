import os
import pandas as pd
from src.components.model_trainer import MODEL_TRAINER_FOLDER
from datetime import datetime
import joblib

class PredictionPipeline:
    def __init__(self):
        self.model_folder = MODEL_TRAINER_FOLDER
        self.dataset_folder = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\Dataset"
        self.artifacts_folder = os.path.join(os.path.dirname(self.dataset_folder), "artifacts")

    def run(self):
        """Run predictions for all models on the latest CSV"""
        # Load latest CSV
        csv_files = [f for f in os.listdir(self.dataset_folder) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found for prediction.")
        latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(self.dataset_folder, x)))
        df = pd.read_csv(os.path.join(self.dataset_folder, latest_csv))

        # Load models and predict
        predictions = {}
        for model_file in os.listdir(self.model_folder):
            if model_file.endswith(".pkl"):
                model_path = os.path.join(self.model_folder, model_file)
                model = joblib.load(model_path)
                y_pred = model.predict(df.select_dtypes(include=['number']))  # use numeric columns
                predictions[model_file] = y_pred

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_file = os.path.join(self.artifacts_folder, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pred_df.to_csv(pred_file, index=False)
        return pred_file
