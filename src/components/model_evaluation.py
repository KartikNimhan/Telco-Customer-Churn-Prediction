import os
import json
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# ================================
# Folder paths
# ================================
BASE_DIR = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

DATA_TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, "2_data_transformation")
MODEL_TRAINER_DIR = os.path.join(ARTIFACTS_DIR, "3_model_trainer")
MODEL_EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "4_model_evaluation")

os.makedirs(MODEL_EVALUATION_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ================================
# Logging Configuration
# ================================
log_file = os.path.join(LOGS_DIR, f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# Custom Exception
# ================================
class ModelEvaluationException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

# ================================
# Model Evaluation Class
# ================================
class ModelEvaluator:
    def __init__(self):
        self.transformed_data_path = os.path.join(DATA_TRANSFORMATION_DIR, "transformed_WA_Fn-UseC_-Telco-Customer-Churn.csv")
        self.model_dir = MODEL_TRAINER_DIR
        self.metrics_path = os.path.join(MODEL_EVALUATION_DIR, "metrics.json")

    def load_data(self):
        """Load transformed dataset and select exactly the same features as training"""
        try:
            df = pd.read_csv(self.transformed_data_path)
            logger.info(f"Loaded transformed data with shape {df.shape}")

            # Keep only numeric/encoded columns, exclude original 'Churn' and target 'Churn_encoded'
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ["Churn", "Churn_encoded"]]

            X = df[feature_cols]
            y = df["Churn_encoded"]

            logger.info(f"Using features: {feature_cols}")

            # Train-test split same as training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error loading transformed dataset: {e}")
            raise ModelEvaluationException("Failed to load transformed data", e)

    def evaluate_models(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]
            if not model_files:
                raise ModelEvaluationException("No trained models found in model trainer directory.")

            results = {}

            mlflow.set_tracking_uri(f"file:///{os.path.join(BASE_DIR, 'mlruns').replace(os.sep, '/')}")
            mlflow.set_experiment("Telco_Customer_Churn_Evaluation")

            for model_file in model_files:
                model_path = os.path.join(self.model_dir, model_file)
                model_name = os.path.splitext(model_file)[0]

                with mlflow.start_run(run_name=model_name):
                    logger.info(f"Evaluating model: {model_name}")

                    model = joblib.load(model_path)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Log metrics to MLflow
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)
                    mlflow.log_metric("f1_score", f1)

                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred, output_dict=True)

                    results[model_name] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1,
                        "confusion_matrix": cm.tolist(),
                        "classification_report": cr
                    }

                    logger.info(f"✅ {model_name} evaluated. Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Save all metrics
            with open(self.metrics_path, "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"Evaluation completed. Metrics saved at: {self.metrics_path}")
            print(f"✅ Model evaluation completed. Metrics saved at: {self.metrics_path}")

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            print(f"❌ Model evaluation failed: {e}")
            raise ModelEvaluationException("Model evaluation failed", e)

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        evaluator.evaluate_models()
    except ModelEvaluationException as e:
        logger.error(f"Model evaluation failed: {e}")
        print(f"❌ Error: {e}")
