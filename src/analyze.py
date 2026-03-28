from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/best_model.pkl")
HOLDOUT_PREDICTIONS_PATH = Path("outputs/holdout_predictions.csv")
FIGURES_OUTPUT_DIR = Path("outputs/figures")


def load_model(model_path: Path) -> Pipeline:
    """Load trained model pipeline from disk."""
    return joblib.load(model_path)


def load_holdout_predictions(data_path: Path) -> pd.DataFrame:
    """Load saved holdout predictions from disk."""
    holdout_df = pd.read_csv(data_path)

    required_columns = {"y_true", "y_pred"}
    missing_columns = required_columns - set(holdout_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in holdout predictions file: {missing_columns}"
        )

    return holdout_df


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_logreg_coefficients(
    model_pipeline: Pipeline,
    output_path: Path,
) -> None:
    """Plot and save the most pronounced logistic regression coefficients."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fitted_model = model_pipeline.named_steps["model"]
    transformed_feature_names = (
        model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    coef_df = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "coefficient": fitted_model.coef_[0],
        }
    )

    coef_df["feature"] = (
        coef_df["transformed_feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    coef_plot_df = (
        pd.concat(
            [
                coef_df.sort_values(by="coefficient", ascending=True).head(5),
                coef_df.sort_values(by="coefficient", ascending=False).head(5),
            ]
        )
        .drop_duplicates(subset="feature")
        .sort_values(by="coefficient")
    )

    plt.figure(figsize=(9, 6))
    plt.barh(coef_plot_df["feature"], coef_plot_df["coefficient"])
    plt.title("Most Pronounced Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model_pipeline: Pipeline,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Plot and save top feature importances for tree-based models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fitted_model = model_pipeline.named_steps["model"]
    transformed_feature_names = (
        model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    importance_df = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "importance": fitted_model.feature_importances_,
        }
    )

    importance_df["feature"] = (
        importance_df["transformed_feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    importance_plot_df = (
        importance_df.sort_values(by="importance", ascending=False)
        .head(top_n)
        .sort_values(by="importance")
    )

    plt.figure(figsize=(9, 6))
    plt.barh(importance_plot_df["feature"], importance_plot_df["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_interpretation(
    model_pipeline: Pipeline,
    output_dir: Path,
) -> str | None:
    """Save model-specific interpretation plot if supported."""
    fitted_model = model_pipeline.named_steps["model"]

    if isinstance(fitted_model, LogisticRegression):
        output_path = output_dir / "logreg_coefficients.png"
        plot_logreg_coefficients(model_pipeline, output_path)
        return str(output_path)

    if hasattr(fitted_model, "feature_importances_"):
        output_path = output_dir / "feature_importance.png"
        plot_feature_importance(model_pipeline, output_path)
        return str(output_path)

    return None


def main() -> None:
    holdout_df = load_holdout_predictions(HOLDOUT_PREDICTIONS_PATH)
    model_pipeline = load_model(MODEL_PATH)

    model_name = model_pipeline.named_steps["model"].__class__.__name__

    confusion_matrix_path = FIGURES_OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=holdout_df["y_true"],
        y_pred=holdout_df["y_pred"],
        model_name=model_name,
        output_path=confusion_matrix_path,
    )

    interpretation_path = save_model_interpretation(model_pipeline, FIGURES_OUTPUT_DIR)

    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    if interpretation_path is not None:
        print(f"Model interpretation plot saved to: {interpretation_path}")
    else:
        print(f"No interpretation plot created for model: {model_name}")


if __name__ == "__main__":
    main()