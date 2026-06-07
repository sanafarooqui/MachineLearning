"""Feature/target split, train/test split, and the preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data_prep import BINARY_TARGET_COLUMN, TARGET_COLUMN

ORDINAL_COLUMNS = ["age"]
AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
    "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_features_target(df):
    """Split a cleaned dataframe into features ``X`` and binary target ``y``."""
    X = df.drop(columns=[TARGET_COLUMN, BINARY_TARGET_COLUMN])
    y = df[BINARY_TARGET_COLUMN]
    return X, y


def get_column_groups(X):
    """Infer numeric / categorical / ordinal column groups from ``X``.

    ``age`` is treated as ordinal (it has a natural order) rather than as an
    unordered categorical.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    cat_cols.remove("age")
    return num_cols, cat_cols, ORDINAL_COLUMNS


def build_preprocessor(X):
    """Build the ColumnTransformer used by every model pipeline.

    - Numeric columns: median impute + standard scale.
    - Categorical columns: constant impute ('Unknown') + one-hot encode.
    - Age: ordinal encode using its natural bucket order.
    """
    num_cols, cat_cols, ord_cols = get_column_groups(X)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
    ])
    ord_pipe = Pipeline([
        ("encoder", OrdinalEncoder(categories=[AGE_ORDER])),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
        ("ord", ord_pipe, ord_cols),
    ])
    return preprocessor


def make_train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified train/test split — preserves the (heavily imbalanced) class ratio."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
