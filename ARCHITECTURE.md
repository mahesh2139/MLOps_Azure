# MLOps Architecture & Integration Guide

## System Architecture

### Cloud Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                        GITHUB REPOSITORY                         │
│  (Code versioning, workflow definitions, governance policies)   │
└──────────────────────────┬──────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
           ┌────▼─────┐      ┌─────▼─────┐
       GitHub Actions │      │  Azure ML  │
       (CI/CD Pipeline)      │ Workspace  │
           └────┬─────┘      └─────┬─────┘
                │                   │
    ┌───────────┴───────────┐       │
    │                       │       │
    ▼                       ▼       ▼
┌──────────┐  ┌──────────┐  ┌─────────────────┐
│ Training │  │ Inference│  │ Azure Blob      │
│ Pipeline │  │ Pipeline │  │ Storage         │
└────┬─────┘  └────┬─────┘  └─────────────────┘
     │             │              │
     ├─────────────┼──────────────┤
     │             │              │
     ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Registry (Azure ML)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Champion v5: [accuracy: 0.85, role: champion, ...]     │   │
│  │ Archive v4:  [accuracy: 0.82, role: archived, ...]    │   │
│  │ Archive v3:  [accuracy: 0.79, role: archived, ...]    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    ┌────────────┐  ┌──────────────┐
    │  Endpoints │  │  Monitoring  │
    │ (Real-time)│  │  & Analytics │
    └────────────┘  └──────────────┘
```

## Component Integration

### 1. GitHub Actions ↔ Azure ML

**Authentication Flow**
```
GitHub Secrets
    ↓
AZURE_CREDENTIALS (Service Principal)
    ↓
az login (Azure CLI)
    ↓
workspace = Workspace.from_config()
    ↓
Model Registry Access
```

**Configuration Files**
- `.github/workflows/` - CI/CD definitions
- `config/config.yaml` - Application settings
- `config/thresholds.yaml` - Governance policies

### 2. Data Flow

```
Raw Data (Blob Storage)
    ↓
Data Preparation (data_prep.py)
    ↓ Prepared Data
Training (train.py)
    ├→ metrics.json
    ├→ model.pkl
    └→ feature_importance.json
    ↓
Evaluation (evaluate.py)
    ├→ evaluation_report.json
    └→ governance_gates.json
    ↓
Champion/Challenger (champion_challenger.py)
    ├→ Model Registry (if promoted)
    └→ decision.json
    ↓
Batch Data → Batch Inference (batch_inference.py)
    ├→ batch_predictions.csv
    └→ batch_summary.json
    ↓
Monitoring (model_monitoring.py)
    ├→ data_drift_report.html
    └→ monitoring_summary.json
```

### 3. Governance Gate Implementation

```python
# Governance Gate Workflow
┌──────────────────────────────────┐
│ Training Complete                 │
└────────────┬─────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Metrics Validation  │
    │ (accuracy ≥ 0.60)   │
    └────────┬────────────┘
             │ Pass/Fail
    ┌────────▼────────┐
    │ Explainability  │
    │ (Top 10 features)
    └────────┬────────┘
             │ Pass/Fail
    ┌────────▼──────────────────┐
    │ Governance Gates          │
    │ (all checks must pass)     │
    └────────┬───────────────────┘
             │ Pass/Fail
    ┌────────▼───────────┐
    │ Promotion Eligible │
    │ (Ready for review) │
    └───────────────────┘
```

## Step-by-Step Integration

### Phase 1: Setup Azure ML Workspace

```bash
# 1. Create resource group
az group create --name mlops-rg --location eastus

# 2. Create workspace
az ml workspace create \
  --name mlops-ws \
  --resource-group mlops-rg

# 3. Create datastore
az ml data create \
  --name fraud-data \
  --type uri_file \
  --path wasbs://data@storage.blob.core.windows.net/

# 4. Download config
az ml workspace download-config \
  --resource-group mlops-rg \
  --name mlops-ws \
  --output-dir ./
```

### Phase 2: Set Up GitHub Actions

```bash
# 1. Create service principal
az ad sp create-for-rbac \
  --name mlops-github-action \
  --role Contributor

# 2. Add to GitHub Secrets
AZURE_CREDENTIALS (entire JSON output)
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_WORKSPACE_NAME

# 3. Commit and push
git add .github/workflows/
git commit -m "Add MLOps CI/CD pipelines"
git push origin main
```

### Phase 3: Deploy Pipelines

```bash
# Option 1: Local testing
./scripts/setup.sh
./scripts/run_pipeline.sh

# Option 2: GitHub Actions
git push origin main  # Triggers model_training.yml
# View: https://github.com/yourorg/repo/actions
```

## Model Registry Strategy

### Tagging System

```python
Model Tags Structure:
{
  'role': 'champion|challenger|archived',
  'status': 'production|staging|archived',
  'framework': 'sklearn',
  'accuracy': '0.85',
  'precision': '0.82',
  'recall': '0.87',
  'f1_score': '0.85',
  'features_count': '31',
  'training_date': '2026-04-18',
  'promotion_reason': 'outperformed_champion_v4'
}
```

### Promotion Workflow

```
New Model Training
    ↓
Validation Pass?
    ├─ No → Stop
    └─ Yes ↓
Governance Gates Pass?
    ├─ No → Stop
    └─ Yes ↓
Champion Exists?
    ├─ No → Register as champion
    └─ Yes ↓
        Compare Metrics
            ├─ Challenger Better → Promote
            │   ├ Archive Champion (v_old)
            │   └ Promote Challenger (v_new)
            └─ Champion Better → Keep
                Archive Challenger
```

## Batch Inference Integration

### Scheduled Execution

```yaml
# GitHub Actions Schedule
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC

# Or triggered by data arrival
on:
  push:
    paths:
      - 'data/batch_input/**'
```

### Data Preparation

```
1. Load batch data from blob storage
2. Download champion model from registry
3. Prepare features (same preprocessing as training)
4. Generate predictions + probabilities
5. Save results with timestamp
6. Upload to blob storage
```

## Monitoring Integration

### Drift Detection Setup

```python
# Reference: Training data (30K samples)
reference_df = pd.read_csv('outputs/x_test.pkl')

# Current: Batch predictions (134K samples)
current_df = pd.read_csv('outputs/batch_results/batch_predictions.csv')

# Comparison (via Evidently)
drift_report.run(reference_data=reference_df, 
                 current_data=current_df,
                 column_mapping=column_mapping)
```

### Alert Configuration

```python
# Thresholds (config/thresholds.yaml)
data_drift:
  feature_drift_threshold: 0.1
  allowed_drift_columns: 3

monitoring:
  performance_degradation_threshold: 0.05
  data_distribution_shift_threshold: 0.15
```

## Security Best Practices

### 1. Authentication

```python
# Service Principal Authentication
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace

credential = DefaultAzureCredential()
workspace = Workspace.from_config()
```

### 2. Secrets Management

```bash
# GitHub Secrets (encrypted)
AZURE_CREDENTIALS
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_WORKSPACE_NAME
AZURE_STORAGE_KEY

# Local .env (gitignored)
MLFLOW_TRACKING_URI=...
```

### 3. RBAC Configuration

```bash
# Grant minimal required permissions
az role assignment create \
  --role "Azure Machine Learning Operator" \
  --assignee <service-principal-id>
```

## Performance Optimization

### 1. Distributed Training

```python
# For large datasets (> 1GB)
from azureml.core.compute import AmlCompute

compute_target = AmlCompute.create(
    workspace=workspace,
    name="gpu-cluster",
    vm_size="STANDARD_NC6",
    min_nodes=1,
    max_nodes=4
)
```

### 2. Parallel Batch Inference

```bash
# Process batches in parallel
for batch in data/batches/*.csv; do
    python -m src.pipelines.batch_inference \
      --batch-input-path "$batch" &
done
wait
```

### 3. Caching

```python
# Cache model downloads
MODEL_CACHE_DIR = "./model_cache"
model.download(target_dir=MODEL_CACHE_DIR, exist_ok=True)
```

## Deployment Strategies

### Strategy 1: Blue-Green Deployment

```
Blue (Current Champion)
    ↓
    → Serves 100% traffic
    
Green (New Challenger)
    ↓
    → 10% traffic (shadow)
    ↓
    → 50% traffic (canary)
    ↓
    → 100% traffic (if no issues)
```

### Strategy 2: Shadow Mode

```python
# New model runs alongside current but doesn't affect predictions
prediction_champion = champion_model.predict(X)
prediction_challenger = challenger_model.predict(X)

# Log both for comparison
log_predictions(champion=prediction_champion, 
               challenger=prediction_challenger)

# Return champion prediction to user
return prediction_champion
```

## Scaling the Pipeline

### For Increased Data Volume

1. **Streaming Data**: Use Azure Event Hubs
2. **Distributed Processing**: Use Apache Spark
3. **Data Versioning**: Implement DVC (Data Version Control)

### For Model Complexity

1. **Ensemble Methods**: Combine multiple models
2. **Deep Learning**: Migrate to PyTorch/TensorFlow
3. **AutoML**: Use Azure AutoML

### For Real-time Requirements

1. **Online Inference**: Deploy as Azure ML endpoint
2. **Latency Optimization**: Use model quantization
3. **Load Balancing**: Use Azure Load Balancer

## Troubleshooting Guide

### Issue: Model not found in registry

```bash
# Check registry status
az ml model list --name CreditCard_Fraud_Detection

# Manually register
az ml model create \
  --name CreditCard_Fraud_Detection \
  --path ./outputs/model.pkl
```

### Issue: Batch inference timeout

```python
# Increase timeout in workflow
timeout-minutes: 30

# Or split batch into smaller chunks
for chunk in split_dataframe(batch_df, chunk_size=10000):
    batch_inference.run_on_chunk(chunk)
```

### Issue: Data drift alerts

```python
# Review drift report
cat outputs/batch_results/model_monitoring/monitoring_summary.json

# Investigate feature changes
python -c "
import pandas as pd
batch = pd.read_csv('batch_predictions.csv')
ref = pd.read_csv('x_test.pkl')
print(batch.describe())
print(ref.describe())
"
```

## Compliance & Governance

### Audit Trail

```python
# Every model action logged
{
  'timestamp': '2026-04-18T10:30:00Z',
  'action': 'promotion',
  'model_version': '5',
  'reason': 'outperformed_v4_on_f1',
  'metrics_delta': {'f1_score': +0.03},
  'user': 'mlops_pipeline',
  'approver': 'data_team'
}
```

### Explainability Tracking

```python
# Store feature importance for every model
{
  'model_version': 5,
  'top_features': {
    'V14': 0.156,
    'V10': 0.142,
    'V12': 0.128,
    ...
  }
}
```

---

**Document Version**: 1.0
**Last Updated**: April 2026
**Maintained By**: MLOps Team
