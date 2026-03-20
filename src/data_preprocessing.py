import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in df.columns if col not in numeric_features]

    if not numeric_features and not categorical_features:
        raise ValueError("No feature columns were found.")

    return numeric_features, categorical_features


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""
    if not num_cols and not cat_cols:
        raise ValueError("Both num_cols and cat_cols are empty.")

    transformers = []

    if num_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, num_cols))

    if cat_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers)