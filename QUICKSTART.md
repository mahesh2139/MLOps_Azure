# MLOps Quick Start Guide

Get your production MLOps pipeline running in 15 minutes.

## Prerequisites

- Python 3.8+
- Azure subscription with ML workspace
- GitHub repository (optional, for CI/CD)

## 5-Minute Setup

### 1. Clone & Install (2 min)

```bash
# Navigate to project
cd MLOps_Azure

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs outputs/batch_results
```

### 2. Azure ML Authentication (1 min)

```bash
# Login to Azure
az login

# (If in workspace) Download config
az ml workspace download-config -w YOUR_WORKSPACE -g YOUR_RG
```

### 3. Run First Pipeline (2 min)

```bash
# Run data preparation
python -m src.pipelines.data_prep

# Run training
python -m src.pipelines.train

# Check results
ls -la outputs/
cat outputs/metrics.json
```

## Common Tasks

### 📊 Train New Model

```bash
python -m src.pipelines.train \
  --prepared-data-dir "./outputs/prepared_data" \
  --output-dir "./outputs"
```

**Output**: `outputs/model.pkl`, `outputs/metrics.json`

### ✅ Evaluate Model

```bash
python -m src.pipelines.evaluate \
  --model-path "./outputs/model.pkl" \
  --test-data-dir "./outputs" \
  --output-dir "./outputs"
```

**Output**: `outputs/evaluation_report.json` (includes governance gates ✓)

### 🏆 Compare & Promote

```bash
python -m src.pipelines.champion_challenger \
  --model-path "./outputs/model.pkl" \
  --metrics-path "./outputs/metrics.json"
```

**Output**: Model promoted to champion (if better) → appears in registry

### 🔮 Run Batch Predictions

```bash
python -m src.pipelines.batch_inference \
  --batch-input-path "UI/2026-04-17_180302_UTC/creditcard_batch_input.csv" \
  --output-dir "./outputs/batch_results"
```

**Output**: `outputs/batch_results/batch_predictions.csv` (134K rows)

### 📈 Monitor for Drift

```bash
python -m src.pipelines.model_monitoring \
  --batch-results-path "./outputs/batch_results/batch_predictions.csv" \
  --reference-data-dir "./outputs" \
  --output-dir "./outputs/batch_results/model_monitoring"
```

**Output**: `outputs/batch_results/model_monitoring/data_drift_report.html` 🎯

## Full Pipeline (Automated)

```bash
# Run everything end-to-end
./scripts/run_pipeline.sh

# Checks all 6 steps:
# ✓ Data Prep
# ✓ Training
# ✓ Evaluation
# ✓ Champion/Challenger
# ✓ Batch Inference
# ✓ Monitoring
```

## View Results

### Model Metrics
```bash
cat outputs/metrics.json
```

### Governance Report
```bash
cat outputs/evaluation_report.json | jq .governance_gates
```

### Batch Predictions
```bash
head -20 outputs/batch_results/batch_predictions.csv
wc -l outputs/batch_results/batch_predictions.csv  # Count rows
```

### Drift Report
```bash
# Open in browser
open outputs/batch_results/model_monitoring/data_drift_report.html
```

## Model Registry

### See All Models

```bash
az ml model list --name "CreditCard_Fraud_Detection"
```

### Download Champion

```python
from src.components.model_registry import ModelRegistry
from azureml.core import Workspace

ws = Workspace.from_config()
registry = ModelRegistry(ws, "CreditCard_Fraud_Detection")
champion = registry.get_champion_model()
registry.download_model(champion, "./champion_model")
```

### Compare Models

```python
model1 = registry.list_models()[0]
model2 = registry.list_models()[1]
comparison = registry.compare_models(model1, model2)
print(comparison)
```

## GitHub Actions (CI/CD)

### Enable Workflows

```bash
# Copy workflows to .github/workflows/
# Add GitHub Secrets:

AZURE_CREDENTIALS          # Full JSON from Service Principal
AZURE_SUBSCRIPTION_ID      # Your subscription
AZURE_RESOURCE_GROUP       # Your resource group
AZURE_WORKSPACE_NAME       # Your workspace
```

### Trigger Workflows

```bash
# Option 1: Push to main branch
git add .
git commit -m "Update training config"
git push origin main
→ Automatically runs model_training.yml

# Option 2: Manual trigger
# Go to GitHub → Actions → Choose workflow → Run workflow
```

### Monitor Workflows

Visit: `https://github.com/YOUR_ORG/YOUR_REPO/actions`

## Troubleshooting

### Issue: "config.json not found"

```bash
# Download from Azure ML
az ml workspace download-config -w YOUR_WORKSPACE -g YOUR_RG
# Or use local CSV instead of Azure
```

### Issue: "No champion model found"

```bash
# First time? Register initial model:
az ml model register \
  --name "CreditCard_Fraud_Detection" \
  --path "./outputs/model.pkl" \
  --tags role=champion status=production
```

### Issue: "Batch file not found"

```bash
# Check datastore
az storage blob list --container-name ui \
  --account-name YOUR_STORAGE

# Or use local CSV
python -m src.pipelines.batch_inference \
  --batch-input-path "./data/batch_input.csv"
```

### Issue: "Drift detection failed"

```bash
# Check if Evidently is installed
pip install evidently

# Or run monitoring without drift
python -m src.pipelines.model_monitoring \
  # (Gracefully skips Evidently if not available)
```

## Configuration

### Key Parameters

**config/config.yaml**
```yaml
model:
  hyperparameters:
    n_estimators: 100      # Change for model tuning
    max_depth: 10          # Control model complexity
```

**config/thresholds.yaml**
```yaml
model_validation:
  accuracy:
    min_threshold: 0.60    # Minimum acceptable accuracy
```

## Next Steps

1. ✅ Run first pipeline: `./scripts/run_pipeline.sh`
2. 📊 Check metrics: `cat outputs/evaluation_report.json`
3. 🏆 Promote model: `python -m src.pipelines.champion_challenger`
4. 🔮 Run inference: `python -m src.pipelines.batch_inference`
5. 📈 Monitor drift: `python -m src.pipelines.model_monitoring`
6. 🚀 Setup GitHub Actions (optional)

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/pipelines/*.py` | Pipeline steps |
| `src/components/*.py` | Reusable components |
| `config/config.yaml` | Configuration |
| `config/thresholds.yaml` | Governance policies |
| `.github/workflows/*.yml` | CI/CD definitions |

## Support

- 📖 Full docs: `MLOPS_README.md`
- 🏗️ Architecture: `ARCHITECTURE.md`
- 🧪 Tests: `./scripts/run_tests.sh`
- 📝 Logs: `logs/`

---

**Ready to use!** 🚀

Next: `python -m src.pipelines.data_prep`
