"""
train.py — Train the Student Performance Prediction Model
Uses Random Forest + Gradient Boosting ensemble on UCI Student Performance data.
Run: python backend/train.py
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "age", "studytime", "failures", "absences", "G1", "G2",
    "traveltime", "freetime", "goout", "health",
    "internet", "paid", "activities", "nursery", "higher",
    "romantic", "famrel", "schoolsup", "famsup"
]

def load_or_generate_data():
    csv_path = DATA_DIR / "student_data.csv"

    if csv_path.exists():
        print(f"📂 Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    print("🔧 Generating synthetic dataset (1200 students)...")
    np.random.seed(42)
    n = 1200

    data = {
        "age": np.random.randint(15, 22, n),
        "studytime": np.random.randint(1, 5, n),
        "failures": np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.20, 0.10, 0.05]),
        "absences": np.random.randint(0, 30, n),
        "G1": np.random.randint(4, 20, n),
        "G2": np.random.randint(4, 20, n),
        "traveltime": np.random.randint(1, 5, n),
        "freetime": np.random.randint(1, 5, n),
        "goout": np.random.randint(1, 5, n),
        "health": np.random.randint(1, 6, n),
        "internet": np.random.randint(0, 2, n),
        "paid": np.random.randint(0, 2, n),
        "activities": np.random.randint(0, 2, n),
        "nursery": np.random.randint(0, 2, n),
        "higher": np.random.randint(0, 2, n),
        "romantic": np.random.randint(0, 2, n),
        "famrel": np.random.randint(1, 6, n),
        "schoolsup": np.random.randint(0, 2, n),
        "famsup": np.random.randint(0, 2, n),
    }

    df = pd.DataFrame(data)

    # Compute G3 with realistic correlation
    g3_raw = (
        0.45 * df["G1"]
        + 0.45 * df["G2"]
        + 0.5 * df["studytime"]
        - 1.5 * df["failures"]
        - 0.05 * df["absences"]
        + 0.3 * df["higher"]
        + 0.2 * df["internet"]
        + np.random.normal(0, 1.5, n)
    )
    df["G3"] = np.clip(g3_raw.round().astype(int), 0, 20)

    # Map G3 to letter grades
    def to_grade(g3):
        if g3 >= 17: return "A"
        elif g3 >= 14: return "B"
        elif g3 >= 10: return "C"
        elif g3 >= 7:  return "D"
        else:           return "F"

    df["grade"] = df["G3"].apply(to_grade)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved dataset: {csv_path} ({len(df)} rows)")
    return df


def train():
    df = load_or_generate_data()

    X = df[FEATURE_NAMES].values
    y = df["grade"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n🚀 Training ensemble model...")

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft"
    )
    ensemble.fit(X_train_s, y_train)

    y_pred = ensemble.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(ensemble, scaler.transform(X), y_encoded, cv=5)

    print(f"\n📊 Test Accuracy:  {accuracy:.4f}")
    print(f"📊 CV Mean Score:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Grade distribution
    grade_dist = dict(zip(*np.unique(y, return_counts=True)))
    grade_dist = {k: int(v) for k, v in grade_dist.items()}

    # Save artifacts
    joblib.dump(ensemble, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)

    stats = {
        "accuracy": round(float(accuracy), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": len(FEATURE_NAMES),
        "grade_distribution": grade_dist,
        "classes": list(le.classes_)
    }

    with open(MODELS_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Model artifacts saved to: {MODELS_DIR}")
    print(f"   model.pkl, scaler.pkl, label_encoder.pkl, feature_names.json, stats.json")


if __name__ == "__main__":
    train()
