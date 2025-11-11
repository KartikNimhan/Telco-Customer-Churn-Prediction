# import os
# import pandas as pd
# import logging
# from datetime import datetime
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # =======================
# # Folder paths
# # =======================
# DATASET_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\Dataset"
# ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
# LOGS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\logs"

# # Subfolder for data transformation artifacts
# DATA_TRANSFORMATION_ARTIFACTS = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")

# # Ensure all folders exist
# os.makedirs(LOGS_FOLDER, exist_ok=True)
# os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
# os.makedirs(DATA_TRANSFORMATION_ARTIFACTS, exist_ok=True)

# # =======================
# # Logging configuration
# # =======================
# log_file = os.path.join(LOGS_FOLDER, f"data_transformation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # =======================
# # Custom Exception
# # =======================
# class DataTransformationException(Exception):
#     def __init__(self, message, errors=None):
#         super().__init__(message)
#         self.errors = errors


# # =======================
# # Data Transformation Class
# # =======================
# class DataTransformation:
#     def __init__(self, dataset_folder: str, artifacts_folder: str):
#         self.dataset_folder = dataset_folder
#         self.artifacts_folder = artifacts_folder

#     def load_raw_csv(self, filename=None):
#         """Load CSV from dataset folder"""
#         try:
#             if filename is None:
#                 csv_files = [f for f in os.listdir(self.dataset_folder) if f.lower().endswith('.csv')]
#                 if not csv_files:
#                     raise FileNotFoundError("No CSV files found in dataset folder.")
#                 filename = csv_files[0]
            
#             file_path = os.path.join(self.dataset_folder, filename)
#             df = pd.read_csv(file_path)
#             logger.info(f"Raw CSV loaded: {file_path} with shape {df.shape}")
#             return df, filename
#         except Exception as e:
#             logger.error(f"Error loading CSV: {e}")
#             raise DataTransformationException("Failed to load raw CSV", e)

#     def transform(self, df: pd.DataFrame):
#         """Perform all transformations and encoding"""
#         try:
#             # Drop customerID column if it exists
#             if 'customerID' in df.columns:
#                 df.drop(columns=['customerID'], inplace=True)
#                 logger.info("Dropped 'customerID' column.")

#             # Convert TotalCharges to numeric (coerce errors)
#             if 'TotalCharges' in df.columns:
#                 df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#                 df['TotalCharges'] = df['TotalCharges'].fillna(0)

#             # Identify categorical columns
#             cat_cols = df.select_dtypes(include='object').columns.tolist()
#             label_encoders = {}

#             # Encode categorical columns
#             for col in cat_cols:
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col].astype(str))
#                 label_encoders[col] = le
#             logger.info(f"Categorical columns encoded in place: {cat_cols}")

#             # ✅ Encode target column 'Churn' separately if it exists
#             if 'Churn' in df.columns:
#                 le_churn = LabelEncoder()
#                 df['Churn_encoded'] = le_churn.fit_transform(df['Churn'])
#                 label_encoders['Churn'] = le_churn
#                 logger.info("✅ Target column 'Churn' encoded as 'Churn_encoded'.")

#             # Handle missing values in numeric columns
#             num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#             df[num_cols] = df[num_cols].fillna(0)

#             logger.info(f"✅ Final transformed data shape: {df.shape}")
#             return df, label_encoders

#         except Exception as e:
#             logger.error(f"Error during transformation: {e}")
#             raise DataTransformationException("Failed during data transformation", e)

#     def save_transformed_csv(self, df: pd.DataFrame, filename: str):
#         """Save transformed data to artifacts subfolder"""
#         try:
#             transformed_file_path = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "transformed_" + filename)
#             df.to_csv(transformed_file_path, index=False)
#             logger.info(f"Transformed CSV saved: {transformed_file_path}")
#             return transformed_file_path
#         except Exception as e:
#             logger.error(f"Error saving transformed CSV: {e}")
#             raise DataTransformationException("Failed to save transformed CSV", e)

#     def save_encoders(self, encoders):
#         """Save label encoders for inference"""
#         encoder_path = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "label_encoders.pkl")
#         with open(encoder_path, "wb") as f:
#             pickle.dump(encoders, f)
#         logger.info(f"✅ Label encoders saved at: {encoder_path}")
#         return encoder_path


# # =======================
# # Example usage
# # =======================
# if __name__ == "__main__":
#     try:
#         transformer = DataTransformation(dataset_folder=DATASET_FOLDER, artifacts_folder=DATA_TRANSFORMATION_ARTIFACTS)

#         # Step 1: Load raw CSV
#         df, raw_filename = transformer.load_raw_csv()

#         # Step 2: Transform data
#         df_transformed, encoders = transformer.transform(df)

#         # Step 3: Save transformed CSV
#         transformed_path = transformer.save_transformed_csv(df_transformed, raw_filename)

#         # Step 4: Save encoders for inference
#         transformer.save_encoders(encoders)

#         print(f"✅ Transformed data saved at: {transformed_path}")

#     except DataTransformationException as e:
#         logger.error(f"Data transformation failed: {e}")
#         print(f"❌ Error: {e}")



import os
import pandas as pd
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle

# =======================
# Folder paths
# =======================
DATASET_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\Dataset"
ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
LOGS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\logs"

# Subfolder for data transformation artifacts
DATA_TRANSFORMATION_ARTIFACTS = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")

# Ensure all folders exist
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
os.makedirs(DATA_TRANSFORMATION_ARTIFACTS, exist_ok=True)

# =======================
# Logging configuration
# =======================
log_file = os.path.join(LOGS_FOLDER, f"data_transformation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================
# Custom Exception
# =======================
class DataTransformationException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

# =======================
# Data Transformation Class
# =======================
class DataTransformation:
    def __init__(self, dataset_folder: str, artifacts_folder: str):
        self.dataset_folder = dataset_folder
        self.artifacts_folder = artifacts_folder

    def load_raw_csv(self, filename=None):
        """Load CSV from dataset folder"""
        try:
            if filename is None:
                csv_files = [f for f in os.listdir(self.dataset_folder) if f.lower().endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in dataset folder.")
                filename = csv_files[0]
            
            file_path = os.path.join(self.dataset_folder, filename)
            df = pd.read_csv(file_path)
            logger.info(f"Raw CSV loaded: {file_path} with shape {df.shape}")
            return df, filename
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise DataTransformationException("Failed to load raw CSV", e)

    def transform(self, df: pd.DataFrame):
        """Perform all transformations and encoding"""
        try:
            # Drop customerID column if it exists
            if 'customerID' in df.columns:
                df.drop(columns=['customerID'], inplace=True)
                logger.info("Dropped 'customerID' column.")

            # Convert TotalCharges to numeric (coerce errors)
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(0)

            # Identify categorical columns
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            label_encoders = {}

            # Encode categorical columns
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            logger.info(f"Categorical columns encoded in place: {cat_cols}")

            # Encode target column 'Churn' separately if it exists
            if 'Churn' in df.columns:
                le_churn = LabelEncoder()
                df['Churn_encoded'] = le_churn.fit_transform(df['Churn'])
                label_encoders['Churn'] = le_churn
                logger.info("✅ Target column 'Churn' encoded as 'Churn_encoded'.")

            # Handle missing values in numeric columns
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            df[num_cols] = df[num_cols].fillna(0)

            # Save feature order (excluding target column)
            feature_order = df.drop(columns=['Churn_encoded'] if 'Churn_encoded' in df.columns else []).columns.tolist()
            feature_order_path = os.path.join(self.artifacts_folder, "feature_order.pkl")
            with open(feature_order_path, "wb") as f:
                pickle.dump(feature_order, f)
            logger.info(f"✅ Feature order saved at: {feature_order_path}")

            logger.info(f"✅ Final transformed data shape: {df.shape}")
            return df, label_encoders

        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise DataTransformationException("Failed during data transformation", e)

    def save_transformed_csv(self, df: pd.DataFrame, filename: str):
        """Save transformed data to artifacts subfolder"""
        try:
            transformed_file_path = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "transformed_" + filename)
            df.to_csv(transformed_file_path, index=False)
            logger.info(f"Transformed CSV saved: {transformed_file_path}")
            return transformed_file_path
        except Exception as e:
            logger.error(f"Error saving transformed CSV: {e}")
            raise DataTransformationException("Failed to save transformed CSV", e)

    def save_encoders(self, encoders):
        """Save label encoders for inference"""
        encoder_path = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "label_encoders.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(encoders, f)
        logger.info(f"✅ Label encoders saved at: {encoder_path}")
        return encoder_path

# =======================
# Example usage
# =======================
if __name__ == "__main__":
    try:
        transformer = DataTransformation(dataset_folder=DATASET_FOLDER, artifacts_folder=DATA_TRANSFORMATION_ARTIFACTS)

        # Step 1: Load raw CSV
        df, raw_filename = transformer.load_raw_csv()

        # Step 2: Transform data
        df_transformed, encoders = transformer.transform(df)

        # Step 3: Save transformed CSV
        transformed_path = transformer.save_transformed_csv(df_transformed, raw_filename)

        # Step 4: Save encoders for inference
        transformer.save_encoders(encoders)

        print(f"✅ Transformed data saved at: {transformed_path}")

    except DataTransformationException as e:
        logger.error(f"Data transformation failed: {e}")
        print(f"❌ Error: {e}")
