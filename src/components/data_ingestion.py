import os
import pandas as pd
import logging
from datetime import datetime
import gdown  # pip install gdown

# =======================
# Folder paths
# =======================
LOGS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\logs"
ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
DATASET_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\Dataset"

# Subfolder for data ingestion artifacts
DATA_INGESTION_ARTIFACTS = os.path.join(ARTIFACTS_FOLDER, "1_data_ingestion")

# Ensure all folders exist
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(DATA_INGESTION_ARTIFACTS, exist_ok=True)

# =======================
# Logging configuration
# =======================
log_file = os.path.join(LOGS_FOLDER, f"data_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================
# Custom Exception
# =======================
class DataIngestionException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

# =======================
# Data Ingestion Class
# =======================
class DataIngestion:
    def __init__(self, dataset_folder: str, artifacts_folder: str, gdrive_folder_url: str):
        self.dataset_folder = dataset_folder
        self.artifacts_folder = artifacts_folder
        self.gdrive_folder_url = gdrive_folder_url
        os.makedirs(self.dataset_folder, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)

    def download_folder_from_drive(self):
        """Download entire folder from Google Drive into dataset folder"""
        try:
            logger.info(f"Starting download of folder from Google Drive: {self.gdrive_folder_url}")
            gdown.download_folder(
                self.gdrive_folder_url,
                output=self.dataset_folder,
                quiet=False,
                use_cookies=False
            )
            logger.info(f"Folder downloaded successfully into: {self.dataset_folder}")
        except Exception as e:
            logger.error(f"Error downloading folder from Google Drive: {e}")
            raise DataIngestionException("Failed to download folder from Google Drive", e)

    def get_csv_files(self):
        """List all CSV files in dataset folder"""
        try:
            csv_files = [f for f in os.listdir(self.dataset_folder) if f.lower().endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in dataset folder.")
            logger.info(f"CSV files found: {csv_files}")
            return csv_files
        except Exception as e:
            logger.error(f"Error detecting CSV files: {e}")
            raise DataIngestionException("Failed to detect CSV files in dataset folder", e)

    def load_csv(self, filename=None):
        """Load CSV file from dataset folder and save a copy to artifacts subfolder"""
        try:
            if filename is None:
                csv_files = self.get_csv_files()
                filename = csv_files[0]  # default to first CSV
            csv_path = os.path.join(self.dataset_folder, filename)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} does not exist.")
            
            df = pd.read_csv(csv_path)
            logger.info(f"CSV loaded successfully: {csv_path} with shape {df.shape}")

            # Save intermediate copy to data_ingestion artifacts subfolder
            artifacts_path = os.path.join(DATA_INGESTION_ARTIFACTS, filename)
            df.to_csv(artifacts_path, index=False)
            logger.info(f"Intermediate copy saved to artifacts: {artifacts_path}")

            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise DataIngestionException("Failed to load CSV from dataset folder", e)


# =======================
# Example usage
# =======================
if __name__ == "__main__":
    try:
        gdrive_folder_url = "https://drive.google.com/drive/folders/1MRbFiEYG68_vwFaAtkLofQPlm-g0-ask?usp=sharing"
        ingestion = DataIngestion(
            dataset_folder=DATASET_FOLDER,
            artifacts_folder=DATA_INGESTION_ARTIFACTS,
            gdrive_folder_url=gdrive_folder_url
        )

        # Step 1: Download raw folder
        ingestion.download_folder_from_drive()

        # Step 2: Detect CSVs and load first CSV
        df = ingestion.load_csv()  # automatically picks first CSV
        print(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Data ingestion completed successfully. Data shape: {df.shape}")

    except DataIngestionException as e:
        logger.error(f"Data ingestion failed: {e}")
        print(f"Error: {e}")
