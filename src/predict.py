from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline


TARGET_COL = "Onshore/Offshore"

MODEL_PATH = Path("models/best_model.pkl")
TEST_DATA_PATH = Path("data/raw/oil_test.csv")
PREDICTIONS_OUTPUT_PATH = Path("outputs/predictions/submission_oil.csv")

LABEL_MAPPING = {
    "OFFSHORE": 0,
    "ONSHORE": 1,
}


def load_model(model_path: Path) -> Pipeline:
    """Load trained model pipeline from disk."""
    return joblib.load(model_path)


def load_test_data(data_path: Path) -> pd.DataFrame:
    """Load test data from csv file."""
    return pd.read_csv(data_path)


def predict(model_pipeline: Pipeline, test_df: pd.DataFrame) -> pd.Series:
    """Generate predictions for test data."""
    predictions = model_pipeline.predict(test_df)
    return pd.Series(predictions, name=TARGET_COL)


def save_submission(predictions: pd.Series, output_path: Path) -> None:
    """Save submission file in Kaggle format with numeric target labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_predictions = predictions.map(LABEL_MAPPING)

    if encoded_predictions.isna().any():
        unexpected_labels = predictions[encoded_predictions.isna()].unique().tolist()
        raise ValueError(
            f"Unexpected prediction labels found: {unexpected_labels}"
        )

    submission = pd.DataFrame(
        {
            "index": range(len(encoded_predictions)),
            TARGET_COL: encoded_predictions.astype(int),
        }
    )

    submission.to_csv(output_path, index=False)


def main() -> None:
    model_pipeline = load_model(MODEL_PATH)
    test_df = load_test_data(TEST_DATA_PATH)

    predictions = predict(model_pipeline, test_df)
    save_submission(predictions, PREDICTIONS_OUTPUT_PATH)

    print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()