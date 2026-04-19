# Production MLOps Pipeline - Credit Card Fraud Detection

A production-grade, modular MLOps pipeline for credit card fraud detection using Azure ML and GitHub Actions with governance controls.

## Architecture Overview

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Prep    │ → │ Train Model  │ → │ Evaluate     │
└──────────────┘    └──────────────┘    └──────────────┘
    ↓
┌──────────────────────────────────────┐
│ Champion/Challenger Comparison       │
├──────────────────────────────────────┤
│ • Metrics Validation (Thresholds)    │
│ • Governance Gates                   │
│ • Explainability Validation          │
│ • Model Registration & Promotion     │
└──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE (Daily)                     │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────┐    ┌──────────────┐
│ Load Batch   │ → │ Inference    │ → predictions.csv
└──────────────┘    └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  MONITORING PIPELINE (6 hourly)                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Batch Data   │ → │ Data Drift   │ → │ Report       │
└──────────────┘    │ Detection    │    └──────────────┘
                    │ (Evidently)  │
                    └──────────────┘
```

## Project Structure

```
MLOps_Azure/
├── src/
│   ├── components/              # Reusable components
│   │   ├── logger.py           # Logging utilities
│   │   ├── data_loader.py      # Data loading from Azure/local
│   │   ├── model_trainer.py    # Model training and evaluation
│   │   ├── model_validator.py  # Metrics validation & governance
│   │   └── model_registry.py   # Azure ML registry operations
│   │
│   └── pipelines/              # Pipeline steps
│       ├── data_prep.py        # Data preparation
│       ├── train.py            # Model training
│       ├── evaluate.py         # Model evaluation
│       ├── champion_challenger.py  # Model comparison & promotion
│       ├── batch_inference.py  # Batch predictions
│       └── model_monitoring.py # Drift detection & monitoring
│
├── config/
│   ├── config.yaml            # Configuration (paths, params)
│   └── thresholds.yaml        # Model metrics thresholds & governance
│
├── .github/workflows/         # CI/CD pipelines
│   ├── model_training.yml     # Training workflow
│   ├── batch_inference.yml    # Inference workflow
│   └── model_monitoring.yml   # Monitoring workflow
│
├── scripts/
│   ├── setup.sh              # Setup environment
│   ├── run_pipeline.sh       # Run full pipeline
│   └── run_tests.sh          # Run tests
│
├── tests/
│   └── test_components.py    # Unit tests
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Features

### 🔐 Governance & Approval Gates

**Metrics Validation**
- Minimum accuracy: 0.60
- Minimum precision: 0.60
- Minimum recall: 0.60
- Minimum F1-score: 0.60

**Governance Checkpoints**
- ✅ Metrics validation required
- ✅ Explainability review required
- ✅ Feature importance tracking (top 10+ features)
- ⚠️ Manual approval flag available

**Data Drift Detection**
- Feature-level drift analysis
- Distribution shift detection
- Threshold-based alerting

### 📊 Modular Components

**Data Loading**
- Azure ML datastore integration
- Local filesystem support
- CSV format support
- Data validation

**Model Training**
- RandomForestClassifier (configurable)
- MLflow integration for tracking
- Hyperparameter logging
- Feature importance extraction

**Model Validation**
- Metrics-based thresholds
- Governance gate enforcement
- Explainability validation
- Promotion eligibility determination

**Model Registry**
- Azure ML registry integration
- Version tracking
- Role-based tags (champion/challenger)
- Model comparison

### 🔄 Batch Inference

- Download champion model from registry
- Process batch data (134,807+ observations)
- Generate fraud probability scores
- Save predictions and summary statistics

### 📈 Monitoring

- Data drift detection using Evidently AI
- Reference vs. current data comparison
- Feature-level drift metrics
- HTML report generation

## Getting Started

### Prerequisites

- Python 3.8+
- Azure ML workspace configured
- GitHub Actions enabled (for CI/CD)
- Azure CLI installed

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd MLOps_Azure

# 2. Make setup script executable
chmod +x scripts/setup.sh

# 3. Run setup
./scripts/setup.sh

# 4. Update .env with your Azure configuration
vi .env

# 5. Authenticate to Azure
az login
```

### Quick Start

#### Local Execution

```bash
# Run full pipeline locally
./scripts/run_pipeline.sh

# Or run individual steps
python -m src.pipelines.data_prep
python -m src.pipelines.train
python -m src.pipelines.evaluate
python -m src.pipelines.champion_challenger
python -m src.pipelines.batch_inference
python -m src.pipelines.model_monitoring
```

#### GitHub Actions (CI/CD)

Push to main/develop branch or trigger manually:

```bash
# Training pipeline
git push origin main

# Manual triggers (via GitHub UI)
- model_training.yml: Any time
- batch_inference.yml: Daily at 2 AM UTC
- model_monitoring.yml: Every 6 hours
```

### Configuration

**config/config.yaml** - Application settings
```yaml
model:
  name: "CreditCard_Fraud_Detection"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
```

**config/thresholds.yaml** - Governance policies
```yaml
model_validation:
  accuracy:
    min_threshold: 0.60
approval_gates:
  require_metrics_validation: true
  require_explainability_review: true
```

## Pipeline Details

### 1. Training Pipeline

**Step 1: Data Preparation**
- Load training data from Azure ML datastore
- Validate data quality
- Split into train/test sets (80/20)
- Save prepared data

**Step 2: Model Training**
- Train RandomForestClassifier
- Log parameters to MLflow
- Calculate evaluation metrics

**Step 3: Model Evaluation**
- Test model performance
- Extract feature importance
- Validate against thresholds
- Check governance gates

**Step 4: Champion/Challenger**
- Register model with metrics as tags
- Compare with current champion (if exists)
- Promote challenger if superior
- Archive previous champion

### 2. Batch Inference Pipeline

**Execution**: Daily at 2 AM UTC

- Download champion model from registry
- Process batch data (all observations)
- Generate predictions and probabilities
- Save results as CSV
- Generate summary statistics

### 3. Monitoring Pipeline

**Execution**: Every 6 hours

- Compare batch data to reference data
- Detect feature-level data drift
- Generate Evidently AI report
- Calculate monitoring statistics
- Alert on significant drift

## Usage Examples

### Train New Model

```python
from src.pipelines import train

result = train.run_training(
    prepared_data_dir="./outputs/prepared_data",
    output_dir="./outputs",
    model_params={
        'n_estimators': 100,
        'max_depth': 10
    }
)
print(result)  # Metrics and model path
```

### Run Batch Inference

```python
from src.pipelines import batch_inference

summary = batch_inference.run_batch_inference(
    batch_input_path="UI/2026-04-17_180302_UTC/creditcard_batch_input.csv",
    output_dir="./outputs/batch_results"
)
print(summary)  # Predictions summary
```

### Monitor Model

```python
from src.pipelines import model_monitoring

report = model_monitoring.run_model_monitoring(
    batch_results_path="./outputs/batch_results/batch_predictions.csv",
    reference_data_dir="./outputs",
    output_dir="./outputs/batch_results/model_monitoring"
)
print(report)  # Drift detection results
```

## Outputs

### Training Artifacts
```
outputs/
├── model.pkl                      # Trained model
├── x_test.pkl, y_test.pkl        # Test data
├── metrics.json                   # Performance metrics
├── feature_importance.json        # Top 10 features
├── training_result.json           # Training summary
└── evaluation_report.json         # Full evaluation with governance gates
```

### Batch Inference Results
```
outputs/batch_results/
├── batch_predictions.csv          # Predictions + probabilities
└── batch_summary.json             # Prediction statistics
```

### Monitoring Reports
```
outputs/batch_results/model_monitoring/
├── data_drift_report.html         # Evidently drift report
└── monitoring_summary.json        # Drift statistics
```

## Governance Controls

### Approval Gates

| Gate | Requirement | Override |
|------|-------------|----------|
| Metrics Validation | accuracy ≥ 0.60 | Manual review |
| Explainability | Top 10 features tracked | Manual review |
| Drift Detection | Monitored continuously | Alert threshold |

### Model Promotion Logic

1. **Metrics Validation** - New model must pass all thresholds
2. **Governance Gates** - All required gates must pass
3. **Champion Comparison** - New model compared to current champion
4. **Promotion Decision** - If superior on majority of metrics, promote
5. **Archival** - Previous champion archived with version preserved

## Testing

```bash
# Run unit tests
./scripts/run_tests.sh

# Run specific test
pytest tests/test_components.py::TestMetricsValidator -v

# Run with coverage
pytest tests/ --cov=src/
```

## Monitoring & Debugging

### Check Pipeline Status

```bash
# View recent logs
tail -f logs/mlops.log

# View training results
cat outputs/evaluation_report.json | jq .

# View batch results
head outputs/batch_results/batch_predictions.csv
```

### Common Issues

**Issue**: Model not found in registry
```bash
# Check registry
az ml model list --name "CreditCard_Fraud_Detection"
```

**Issue**: Data drift too high
```bash
# Check monitoring report
cat outputs/batch_results/model_monitoring/monitoring_summary.json
```

## Security & Best Practices

- ✅ Store credentials in GitHub Secrets, not in code
- ✅ Use Azure ML workspace authentication
- ✅ Log all model promotions and actions
- ✅ Version all model artifacts
- ✅ Maintain audit trail in registry tags
- ✅ Use approval gates for production deployments
- ✅ Monitor data drift continuously
- ✅ Test batch inference before production

## Performance Metrics

| Component | Typical Time | Notes |
|-----------|-------------|-------|
| Data Preparation | 1-2 min | Depends on data size |
| Model Training | 5-10 min | 100 estimators, full dataset |
| Model Evaluation | 1-2 min | Test set inference |
| Champion/Challenger | 2-3 min | Registry operations |
| Batch Inference | 10-15 min | 134K observations |
| Monitoring | 5-10 min | Drift analysis + reporting |

## Scaling Considerations

### For Larger Datasets
- Use Azure ML Compute for distributed training
- Implement feature engineering pipelines
- Use distributed XGBoost or LightGBM

### For Real-time Inference
- Deploy champion model as Azure ML endpoint
- Use REST API for online predictions
- Implement caching layer

### For Enhanced Monitoring
- Add custom metrics to Evidently reports
- Integrate with Azure Application Insights
- Set up automated alerting

## Future Enhancements

- [ ] AutoML integration for hyperparameter tuning
- [ ] Feature store integration
- [ ] Real-time inference deployment
- [ ] A/B testing framework
- [ ] Custom fairness metrics
- [ ] Model explainability dashboards
- [ ] Integration with Azure DevOps
- [ ] Kubernetes deployment configurations

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test: `./scripts/run_tests.sh`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Create Pull Request

## Support

For issues and questions:
1. Check logs in `logs/` directory
2. Review GitHub Actions workflow runs
3. Consult Azure ML documentation
4. Create GitHub issue with logs and error details

## License

[Your License Here]

## Contact

MLOps Team: mlops@company.com

---

**Last Updated**: April 2026
**Version**: 1.0.0
**Status**: Production Ready ✅
