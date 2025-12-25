import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.data.load_data import load_data
from src.preprocess import build_preprocessor, ALL_FEATURES


def train():
    df = load_data()

    X = df[ALL_FEATURES]
    y = df["poaching_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

    # Save model
    joblib.dump(pipeline, "models/baseline_model.joblib")
    print("\nModel saved to models/baseline_model.joblib")


if __name__ == "__main__":
    train()
