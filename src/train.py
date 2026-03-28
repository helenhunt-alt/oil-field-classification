import json
from pathlib import Path

import joblib
import pandas as pd

from typing import Any

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import get_feature_types, build_preprocessor


TARGET_COL = "Onshore/Offshore"
RARE_CLASS = "ONSHORE-OFFSHORE"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TRAIN_DATA_PATH = Path("data/raw/train_oil.csv")
MODEL_OUTPUT_PATH = Path("models/best_model.pkl")
RESULTS_OUTPUT_PATH = Path("outputs/model_comparison.csv")
METRICS_OUTPUT_PATH = Path("outputs/holdout_metrics.json")
HOLDOUT_PREDICTIONS_OUTPUT_PATH = Path("outputs/holdout_predictions.csv")


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from csv file."""
    return pd.read_csv(data_path)


def remove_rare_class(
    df: pd.DataFrame,
    target_col: str,
    rare_class: str,
) -> pd.DataFrame:
    """Remove rows with a rare target class."""
    return df[df[target_col] != rare_class].copy()


def build_model_pipeline(
    preprocessor,
    model: Any,
) -> Pipeline:
    """Build full sklearn pipeline from preprocessor and model."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_candidate_models() -> dict[str, Any]:
    """Return candidate models for comparison."""
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    cv,
    scoring: dict[str, str],
    candidate_models: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate candidate models with cross-validation."""
    cv_rows = []

    for model_name, model in candidate_models.items():
        model_pipeline = build_model_pipeline(preprocessor, clone(model))

        cv_scores = cross_validate(
            estimator=model_pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        cv_rows.append(
            {
                "model": model_name,
                "balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
                "balanced_accuracy_std": cv_scores["test_balanced_accuracy"].std(),
                "f1_macro_mean": cv_scores["test_f1_macro"].mean(),
                "f1_macro_std": cv_scores["test_f1_macro"].std(),
                "accuracy_mean": cv_scores["test_accuracy"].mean(),
                "accuracy_std": cv_scores["test_accuracy"].std(),
            }
        )

    results_df = (
        pd.DataFrame(cv_rows)
        .sort_values(
            by=["balanced_accuracy_mean", "f1_macro_mean", "accuracy_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    return results_df


def select_best_model(results_df: pd.DataFrame) -> str:
    """Return the name of the best model based on CV results."""
    return results_df.loc[0, "model"]


def fit_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
) -> Pipeline:
    """Fit preprocessing + model pipeline on provided data."""
    num_cols, cat_cols = get_feature_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    model_pipeline = build_model_pipeline(preprocessor, clone(model))
    model_pipeline.fit(X, y)

    return model_pipeline


def save_holdout_predictions(
    y_true: pd.Series,
    y_pred: Any,
    output_path: Path,
) -> None:
    """Save holdout predictions for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    holdout_df = pd.DataFrame(
        {
            "index": y_true.index,
            "y_true": y_true.values,
            "y_pred": y_pred,
        }
    )
    holdout_df.to_csv(output_path, index=False)


def main() -> None:
    train_df = load_training_data(TRAIN_DATA_PATH)
    train_df = remove_rare_class(train_df, TARGET_COL, RARE_CLASS)

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = get_candidate_models()

    num_cols, cat_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    results_df = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        cv=cv,
        scoring=scoring,
        candidate_models=candidate_models,
    )

    best_model_name = select_best_model(results_df)
    best_model = candidate_models[best_model_name]

    holdout_pipeline = fit_pipeline(
        X=X_train,
        y=y_train,
        model=best_model,
    )
    holdout_pred = holdout_pipeline.predict(X_holdout)

    holdout_metrics = {
        "best_model": best_model_name,
        "balanced_accuracy": balanced_accuracy_score(y_holdout, holdout_pred),
        "f1_macro": f1_score(y_holdout, holdout_pred, average="macro"),
        "accuracy": accuracy_score(y_holdout, holdout_pred),
    }

    final_pipeline = fit_pipeline(
        X=X,
        y=y,
        model=best_model,
    )

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(holdout_metrics, file, ensure_ascii=False, indent=4)

    save_holdout_predictions(
        y_true=y_holdout,
        y_pred=holdout_pred,
        output_path=HOLDOUT_PREDICTIONS_OUTPUT_PATH,
    )

    print(f"Best model: {best_model_name}")
    print("Holdout metrics:")
    for metric_name, metric_value in holdout_metrics.items():
        if metric_name != "best_model":
            print(f"{metric_name}: {metric_value:.3f}")

    print(f"Final inference model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()