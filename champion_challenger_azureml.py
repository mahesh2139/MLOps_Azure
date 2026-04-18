"""
Champion/Challenger selection for Azure ML using the current Model_Training.ipynb and model_registery.ipynb pattern.
This script registers a challenger model, compares it to the current champion, and promotes the challenger if it wins.
The current champion is exported to a local directory for batch inference.
"""

import json
import os
import shutil
from pathlib import Path

from azureml.core import Workspace, Model

MODEL_NAME = "CreditCard_Fraud_Detection"
METRICS = ["accuracy", "precision", "recall", "f1_score"]
THRESHOLDS = {
    "accuracy": 0.60,
    "precision": 0.60,
    "recall": 0.60,
    "f1_score": 0.60,
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUTS_DIR / "model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
EXPORT_DIR = OUTPUTS_DIR / "champion_model"


def load_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def validate_metrics(metrics: dict):
    print("\n📊 Validating model metrics...")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric)
        if value is None:
            print(f"❌ Missing metric {metric}")
            passed = False
        elif value < threshold:
            print(f"❌ {metric}: {value:.4f} < {threshold}")
            passed = False
        else:
            print(f"✅ {metric}: {value:.4f} >= {threshold}")
    return passed


def pick_latest_model(models):
    if not models:
        return None
    return max(models, key=lambda m: int(m.version))


def list_models(ws, role=None, status=None):
    models = Model.list(ws, name=MODEL_NAME)
    selected = []
    for model in models:
        tags = model.tags or {}
        if role is not None and tags.get("role") != role:
            continue
        if status is not None and tags.get("status") != status:
            continue
        selected.append(model)
    return selected


def get_candidate_model(ws, role_value):
    return pick_latest_model(list_models(ws, role=role_value))


def parse_metric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_models(challenger: Model, champion: Model):
    challenger_tags = challenger.tags or {}
    champion_tags = champion.tags or {}
    challenger_metrics = {m: parse_metric(challenger_tags.get(m)) for m in METRICS}
    champion_metrics = {m: parse_metric(champion_tags.get(m)) for m in METRICS}

    print("\n📊 Champion vs Challenger metrics:")
    print(f"{'Metric':<15} {'Challenger':<15} {'Champion':<15}")
    print("-" * 45)
    wins = 0
    for metric in METRICS:
        c_val = challenger_metrics.get(metric)
        champ_val = champion_metrics.get(metric)
        print(f"{metric:<15} {str(c_val):<15} {str(champ_val):<15}")
        if c_val is not None and champ_val is not None and c_val > champ_val:
            wins += 1
    return wins > len(METRICS) / 2


def update_model_tags(model: Model, extra_tags: dict):
    tags = model.tags or {}
    tags.update(extra_tags)
    model.update(tags=tags)


def register_challenger(ws, metrics: dict):
    print("\n🚀 Registering challenger model...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    local_model = Path.cwd() / "model.pkl"
    shutil.copy2(str(MODEL_PATH), str(local_model))
    registered = Model.register(
        workspace=ws,
        model_path=str(local_model),
        model_name=MODEL_NAME,
        tags={
            "role": "challenger",
            "status": "staging",
            "framework": "sklearn",
            "accuracy": str(metrics.get("accuracy", 0)),
            "precision": str(metrics.get("precision", 0)),
            "recall": str(metrics.get("recall", 0)),
            "f1_score": str(metrics.get("f1_score", 0)),
        },
        description="Credit Card Fraud Detection challenger model for Azure ML batch inference",
    )
    local_model.unlink(missing_ok=True)
    print(f"✅ Challenger registered: {registered.name} v{registered.version}")
    return registered


def export_champion_model(model: Model):
    print("\n📥 Exporting champion model for batch inference...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.download(target_dir=str(EXPORT_DIR), exist_ok=True)
    print(f"✅ Champion model downloaded to: {EXPORT_DIR}")


def main():
    print("Azure ML Champion/Challenger selection starting...")
    ws = Workspace.from_config()
    print(f"Connected to workspace: {ws.name}")

    metrics = load_metrics(METRICS_PATH)
    if not validate_metrics(metrics):
        print("❌ Metrics validation failed. Aborting champion selection.")
        return

    challenger_model = register_challenger(ws, metrics)
    print(f"ℹ️ Challenger registered as version {challenger_model.version}")

    champion_model = get_candidate_model(ws, "champion")
    if champion_model is None:
        print("⚠️ No champion exists. Promoting challenger as champion.")
        update_model_tags(challenger_model, {"role": "champion", "status": "production"})
        export_champion_model(challenger_model)
        return

    print(f"ℹ️ Current champion: version {champion_model.version}")
    if compare_models(challenger_model, champion_model):
        print("🚀 Challenger outperforms champion. Promoting challenger...")
        update_model_tags(champion_model, {"role": "archived", "status": "archived"})
        update_model_tags(challenger_model, {"role": "champion", "status": "production"})
        export_champion_model(challenger_model)
        print(f"✅ Challenger version {challenger_model.version} promoted to champion.")
    else:
        print("⚠️ Challenger did not outperform the champion. No promotion performed.")


if __name__ == '__main__':
    main()
