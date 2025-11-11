# import os
# import pandas as pd
# import logging
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# import xgboost as xgb
# import joblib
# import mlflow
# import mlflow.sklearn
# from imblearn.over_sampling import SMOTE

# # =======================
# # Folder paths
# # =======================
# ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
# TRANSFORMED_DATA_FOLDER = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")
# MODEL_TRAINER_FOLDER = os.path.join(ARTIFACTS_FOLDER, "3_model_trainer")
# LOGS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\logs"

# os.makedirs(MODEL_TRAINER_FOLDER, exist_ok=True)
# os.makedirs(LOGS_FOLDER, exist_ok=True)

# # =======================
# # Logging configuration
# # =======================
# log_file = os.path.join(LOGS_FOLDER, f"model_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # =======================
# # Custom Exception
# # =======================
# class ModelTrainerException(Exception):
#     def __init__(self, message, errors=None):
#         super().__init__(message)
#         self.errors = errors

# # =======================
# # Model Trainer Class
# # =======================
# class ModelTrainer:
#     def __init__(self, transformed_csv_path):
#         self.transformed_csv_path = transformed_csv_path
#         self.models = {
#             "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced'),
#             "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
#             "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
#             "GradientBoosting": GradientBoostingClassifier(random_state=42),
#             "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1)  # Will calculate later
#         }

#     def load_data(self):
#         """Load transformed dataset and prepare train-test split"""
#         try:
#             df = pd.read_csv(self.transformed_csv_path)
#             logger.info(f"Loaded transformed data from {self.transformed_csv_path} with shape {df.shape}")

#             if "Churn_encoded" not in df.columns:
#                 raise ModelTrainerException("Target column 'Churn_encoded' not found in transformed dataset")

#             X = df.drop(columns=["Churn", "Churn_encoded"], errors='ignore')
#             y = df["Churn_encoded"]

#             # Train-test split with stratify to maintain class distribution
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#             logger.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

#             # Apply SMOTE to handle imbalance
#             sm = SMOTE(random_state=42)
#             X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
#             logger.info(f"Applied SMOTE. Resampled train shape: {X_train_res.shape}")

#             return X_train_res, X_test, y_train_res, y_test
#         except Exception as e:
#             logger.error(f"Error loading or splitting data: {e}")
#             raise ModelTrainerException("Failed to load or prepare data", e)

#     def train_models(self):
#         """Train multiple models and log results to MLflow"""
#         try:
#             X_train, X_test, y_train, y_test = self.load_data()

#             # MLflow tracking
#             tracking_dir = os.path.join(LOGS_FOLDER, "mlruns").replace("\\", "/")
#             mlflow.set_tracking_uri(f"file:///{tracking_dir}")
#             mlflow.set_experiment("Telco_Customer_Churn_MultiModels")

#             results = []

#             for model_name, model in self.models.items():
#                 with mlflow.start_run(run_name=model_name):
#                     logger.info(f"Training model: {model_name}")

#                     # Special handling for XGBoost scale_pos_weight
#                     if model_name == "XGBoost":
#                         ratio = (y_train == 0).sum() / (y_train == 1).sum()
#                         model.set_params(scale_pos_weight=ratio)

#                     model.fit(X_train, y_train)

#                     # Predict
#                     y_pred = model.predict(X_test)
#                     y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

#                     # Evaluate
#                     acc = accuracy_score(y_test, y_pred)
#                     f1 = f1_score(y_test, y_pred)
#                     cm = confusion_matrix(y_test, y_pred)
#                     roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

#                     # Log to MLflow
#                     mlflow.log_param("model_name", model_name)
#                     mlflow.log_metric("accuracy", acc)
#                     mlflow.log_metric("f1_score", f1)
#                     if roc_auc:
#                         mlflow.log_metric("roc_auc", roc_auc)
#                     mlflow.sklearn.log_model(model, model_name)

#                     # Save locally
#                     model_path = os.path.join(MODEL_TRAINER_FOLDER, f"{model_name}_model.pkl")
#                     joblib.dump(model, model_path)
#                     logger.info(f"{model_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc}")
#                     logger.info(f"Model saved at: {model_path}")

#                     results.append({
#                         "model": model_name,
#                         "accuracy": acc,
#                         "f1_score": f1,
#                         "roc_auc": roc_auc,
#                         "model_path": model_path
#                     })

#             # Save summary CSV
#             results_df = pd.DataFrame(results)
#             summary_path = os.path.join(MODEL_TRAINER_FOLDER, "model_training_summary.csv")
#             results_df.to_csv(summary_path, index=False)
#             logger.info(f"Model training summary saved at {summary_path}")
#             print(f"✅ Model training completed. Summary saved at: {summary_path}")

#         except Exception as e:
#             logger.error(f"Model training failed: {e}")
#             raise ModelTrainerException("Model training failed", e)


# # =======================
# # Main execution
# # =======================
# if __name__ == "__main__":
#     try:
#         transformed_csv_path = os.path.join(
#             TRANSFORMED_DATA_FOLDER,
#             "transformed_WA_Fn-UseC_-Telco-Customer-Churn.csv"
#         )

#         trainer = ModelTrainer(transformed_csv_path)
#         trainer.train_models()

#     except Exception as e:
#         import traceback
#         logger.error(f"Model training failed: {e}")
#         print(f"❌ Error: {e}")
#         traceback.print_exc()

import os
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE

# =======================
# Folder paths
# =======================
BASE_FOLDER = os.getenv("PROJECT_BASE", os.getcwd())  # fallback to current working dir
ARTIFACTS_FOLDER = os.path.join(BASE_FOLDER, "artifacts")
TRANSFORMED_DATA_FOLDER = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")
MODEL_TRAINER_FOLDER = os.path.join(ARTIFACTS_FOLDER, "3_model_trainer")
LOGS_FOLDER = os.path.join(BASE_FOLDER, "logs")

# Ensure folders exist
os.makedirs(MODEL_TRAINER_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(LOGS_FOLDER, "mlruns"), exist_ok=True)

# =======================
# Logging configuration
# =======================
log_file = os.path.join(LOGS_FOLDER, f"model_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================
# Custom Exception
# =======================
class ModelTrainerException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

# =======================
# Model Trainer Class
# =======================
class ModelTrainer:
    def __init__(self, transformed_csv_path):
        self.transformed_csv_path = transformed_csv_path
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced'),
            "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1)
        }

    def load_data(self):
        """Load transformed dataset and prepare train-test split"""
        try:
            df = pd.read_csv(self.transformed_csv_path)
            logger.info(f"Loaded transformed data from {self.transformed_csv_path} with shape {df.shape}")

            if "Churn_encoded" not in df.columns:
                raise ModelTrainerException("Target column 'Churn_encoded' not found in transformed dataset")

            X = df.drop(columns=["Churn", "Churn_encoded"], errors='ignore')
            y = df["Churn_encoded"]

            # Train-test split with stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Data split into train and test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # Apply SMOTE
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            logger.info(f"Applied SMOTE. Resampled train shape: {X_train_res.shape}")

            return X_train_res, X_test, y_train_res, y_test
        except Exception as e:
            logger.error(f"Error loading or splitting data: {e}")
            raise ModelTrainerException("Failed to load or prepare data", e)

    def train_models(self):
        """Train multiple models and log results to MLflow"""
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            # MLflow tracking
            mlflow.set_tracking_uri(f"file://{os.path.abspath(os.path.join(LOGS_FOLDER, 'mlruns'))}")
            mlflow.set_experiment("Telco_Customer_Churn_MultiModels")

            results = []

            for model_name, model in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    logger.info(f"Training model: {model_name}")

                    # XGBoost scale_pos_weight
                    if model_name == "XGBoost":
                        ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                        model.set_params(scale_pos_weight=ratio)

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    if roc_auc:
                        mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.sklearn.log_model(model, model_name)

                    model_path = os.path.join(MODEL_TRAINER_FOLDER, f"{model_name}_model.pkl")
                    joblib.dump(model, model_path)
                    logger.info(f"{model_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc}")
                    logger.info(f"Model saved at: {model_path}")

                    results.append({
                        "model": model_name,
                        "accuracy": acc,
                        "f1_score": f1,
                        "roc_auc": roc_auc,
                        "model_path": model_path
                    })

            # Save summary CSV
            results_df = pd.DataFrame(results)
            summary_path = os.path.join(MODEL_TRAINER_FOLDER, "model_training_summary.csv")
            results_df.to_csv(summary_path, index=False)
            logger.info(f"Model training summary saved at {summary_path}")
            print(f"✅ Model training completed. Summary saved at: {summary_path}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelTrainerException("Model training failed", e)


# =======================
# Main execution
# =======================
if __name__ == "__main__":
    try:
        transformed_csv_path = os.path.join(
            TRANSFORMED_DATA_FOLDER,
            "transformed_WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

        trainer = ModelTrainer(transformed_csv_path)
        trainer.train_models()

    except Exception as e:
        import traceback
        logger.error(f"Model training failed: {e}")
        print(f"❌ Error: {e}")
        traceback.print_exc()
