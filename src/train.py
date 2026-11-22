"""Train a leakage-safe logistic-regression voice classifier."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def train(csv_path, output_dir):
    df = pd.read_csv(csv_path).dropna(subset=["label"])
    X = df.drop(columns=["label"])
    y = df["label"].astype(str)
    if X.empty or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("Expected numeric acoustic features")
    if y.nunique() != 2:
        raise ValueError("Expected two classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=1, stratify=y
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=1)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "test_samples": len(y_test),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close()
    print(f"Held-out accuracy: {metrics['accuracy']:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    train(args.data, args.output)
