import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


def create_preprocessing_pipeline(feature_metadata, df):
    """Create a preprocessing pipeline based on feature metadata."""
    numerical_features = []
    categorical_features = []
    excluded_features = []

    # Combine given and derived features
    all_features = {**feature_metadata['given_features'],
                   **feature_metadata['derived_features']}

    # Categorize features based on their preprocessing
    for feature, metadata in all_features.items():
        if metadata['preprocessing'] == 'standard_scaler':
            numerical_features.append(feature)
        elif metadata['preprocessing'] == 'onehot':
            categorical_features.append(feature)
        elif metadata['preprocessing'] == 'none':
            excluded_features.append(feature)

    # Validate presence of metadata columns in dataframe
    metadata_cols = set(numerical_features + categorical_features)
    missing_cols = sorted(metadata_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns from data: {', '.join(missing_cols)}")

    # Intersect features with actual dataframe columns
    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    try:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
    except TypeError:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def train_and_evaluate():
    # Load metadata and data
    with open('Data/feature_metadata.json', 'r') as f:
        feature_metadata = json.load(f)

    df = pd.read_csv('Data/loan_data_processed.csv')
    X = df.drop('credit_approved', axis=1)
    y = df['credit_approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = create_preprocessing_pipeline(feature_metadata, X_train)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    param_grid = {'classifier__C': [0.1, 1.0, 10.0]}

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return grid_search


if __name__ == '__main__':
    train_and_evaluate()
