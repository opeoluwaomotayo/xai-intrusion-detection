"""Train a Random Forest IDS intended for explainability demonstrations."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio"]


def train_model(dataset="data/network_traffic.csv", output="models/xai_ids.joblib", metrics_path="models/metrics.json"):
    df = pd.read_csv(dataset)
    missing = set(FEATURES + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }
    reference = X_train.sample(min(100, len(X_train)), random_state=42)
    bundle = {"model": model, "features": FEATURES, "reference": reference}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


if __name__ == "__main__":
    result = train_model()
    print("XAI IDS model trained.")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key.title()}: {result[key]:.3f}")
    print(f"Confusion matrix: {result['confusion_matrix']}")
    print("Results are from synthetic teaching data, not a real-world benchmark.")
