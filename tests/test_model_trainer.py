# Unit tests for model trainer
# tests/test_model_trainer.py
import pytest
import pandas as pd
from src.components.model_trainer import ModelTrainer

def test_load_data_returns_dataframe(tmp_path):
    # Create a small dummy CSV
    data = {
        "tenure": [1, 2, 3, 4],
        "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
        "TotalCharges": [10.0, 20.0, 30.0, 40.0],
        "Churn_encoded": [0, 0, 1, 1]  # 2 samples per class
    }

    
    csv_file = tmp_path / "dummy.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    trainer = ModelTrainer(str(csv_file))
    X_train, X_test, y_train, y_test = trainer.load_data()

    assert not X_train.empty
    assert not X_test.empty
    assert len(X_train) + len(X_test) == 2
    assert all(col in X_train.columns for col in ["tenure", "MonthlyCharges", "TotalCharges"])

def test_train_models_runs_without_error(tmp_path):
    # Minimal dummy CSV to pass training
    data = {
        "tenure": [1, 2, 3, 4],
        "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
        "TotalCharges": [10.0, 20.0, 30.0, 40.0],
        "Churn_encoded": [0, 1, 0, 1]
    }
    csv_file = tmp_path / "dummy.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    trainer = ModelTrainer(str(csv_file))

    try:
        trainer.train_models()
    except Exception:
        pytest.fail("train_models() raised an exception unexpectedly")
