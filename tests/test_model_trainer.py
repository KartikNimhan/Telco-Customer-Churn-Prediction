# Unit tests for ModelTrainer
# tests/test_model_trainer.py

import pytest
import pandas as pd
from src.components.model_trainer import ModelTrainer

def test_load_data_returns_dataframe(tmp_path):
    """
    Test that load_data() correctly loads CSV, splits, and returns train/test sets
    with expected columns.
    """
    # Create a dummy CSV with enough rows per class for stratified split
    data = {
        "tenure": list(range(1, 11)),  # 10 rows
        "MonthlyCharges": [10, 20, 30, 40, 15, 25, 35, 45, 50, 60],
        "TotalCharges": [10, 20, 30, 40, 15, 25, 35, 45, 50, 60],
        "Churn_encoded": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]  # 5 samples per class
    }

    csv_file = tmp_path / "dummy.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    trainer = ModelTrainer(str(csv_file))
    X_train, X_test, y_train, y_test = trainer.load_data()

    # Check that data is loaded correctly
    assert not X_train.empty
    assert not X_test.empty
    assert len(X_train) + len(X_test) == 10  # total rows
    assert all(col in X_train.columns for col in ["tenure", "MonthlyCharges", "TotalCharges"])
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}


def test_train_models_runs_without_error(tmp_path):
    """
    Test that train_models() runs without raising an exception
    using a small dummy dataset.
    """
    data = {
        "tenure": list(range(1, 11)),
        "MonthlyCharges": [10, 20, 30, 40, 15, 25, 35, 45, 50, 60],
        "TotalCharges": [10, 20, 30, 40, 15, 25, 35, 45, 50, 60],
        "Churn_encoded": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    }

    csv_file = tmp_path / "dummy.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    trainer = ModelTrainer(str(csv_file))

    try:
        trainer.train_models()
    except Exception as e:
        pytest.fail(f"train_models() raised an exception unexpectedly: {e}")
