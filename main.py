import os
import shutil
import logging
from datetime import datetime
import pandas as pd
import joblib
from src.components.data_ingestion import DataIngestion, DataIngestionException
from src.components.data_transformation import DataTransformation, DataTransformationException
from src.components.model_trainer import ModelTrainer, ModelTrainerException
from src.components.model_evaluation import ModelEvaluator, ModelEvaluationException

# =======================
# Folder paths
# =======================
BASE_DIR = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)"
DATASET_FOLDER = os.path.join(BASE_DIR, "Dataset")
ARTIFACTS_FOLDER = os.path.join(BASE_DIR, "artifacts")
MODEL_TRAINER_FOLDER = os.path.join(ARTIFACTS_FOLDER, "3_model_trainer")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(MODEL_TRAINER_FOLDER, exist_ok=True)

# =======================
# Logging configuration
# =======================
log_file = os.path.join(LOGS_FOLDER, f"main_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================
# Main pipeline
# =======================
if __name__ == "__main__":
    try:
        logger.info("===== Pipeline started =====")
        print("🚀 Starting the full pipeline...")

        # ----------------------
        # Step 0: Data Ingestion
        # ----------------------
        logger.info("Step 0: Data Ingestion")
        print("📥 Data Ingestion started...")

        gdrive_folder_url = "https://drive.google.com/drive/folders/1MRbFiEYG68_vwFaAtkLofQPlm-g0-ask?usp=sharing"
        ingestion = DataIngestion(
            dataset_folder=DATASET_FOLDER,
            artifacts_folder=os.path.join(ARTIFACTS_FOLDER, "1_data_ingestion"),
            gdrive_folder_url=gdrive_folder_url
        )

        ingestion.download_folder_from_drive()
        df_raw = ingestion.load_csv()
        print(f"✅ Data ingestion completed. CSV: {df_raw.shape}")

        # ----------------------
        # Step 1: Data Transformation
        # ----------------------
        logger.info("Step 1: Data Transformation")
        transformer = DataTransformation(
            dataset_folder=DATASET_FOLDER,
            artifacts_folder=os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")
        )
        df_transformed, encoders = transformer.transform(df_raw)
        transformed_csv_path = transformer.save_transformed_csv(df_transformed, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        transformer.save_encoders(encoders)
        print(f"✅ Data transformed and saved: {transformed_csv_path}")

        # ----------------------
        # Step 2: Model Training
        # ----------------------
        logger.info("Step 2: Model Training")
        trainer = ModelTrainer(transformed_csv_path)
        X_train, X_test, y_train, y_test = trainer.load_data()

        trainer.train_models()

        # Save feature columns used for training
        feature_cols = X_train.columns.tolist()
        with open(os.path.join(MODEL_TRAINER_FOLDER, "feature_columns.txt"), "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        print("✅ Model training completed and feature columns saved.")

        # ----------------------
        # Step 3: Model Evaluation
        # ----------------------
        logger.info("Step 3: Model Evaluation")
        evaluator = ModelEvaluator()
        evaluator.evaluate_models()
        print("✅ Model evaluation completed.")

        # ----------------------
        # Step 4: Predictions
        # ----------------------
        logger.info("Step 4: Generating Predictions")
        # Load a trained model (example: RandomForest)
        model_path = os.path.join(MODEL_TRAINER_FOLDER, "RandomForest_model.pkl")
        model = joblib.load(model_path)

        # Load transformed data for prediction
        df_pred = pd.read_csv(transformed_csv_path)

        # Load feature columns
        with open(os.path.join(MODEL_TRAINER_FOLDER, "feature_columns.txt"), "r") as f:
            feature_cols = [line.strip() for line in f.readlines()]

        # Use only training features
        X_pred = df_pred[feature_cols]

        # Predict
        y_pred = model.predict(X_pred)
        df_pred["Predicted_Churn"] = y_pred

        predictions_file = os.path.join(
            ARTIFACTS_FOLDER, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_pred.to_csv(predictions_file, index=False)
        print(f"✅ Predictions completed. File saved at: {predictions_file}")

        logger.info("===== Pipeline finished successfully =====")
        print("🎉 Pipeline finished successfully!")

    except (DataIngestionException, DataTransformationException, ModelTrainerException, ModelEvaluationException, Exception) as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
