#!/bin/bash
# Run complete MLOps pipeline end-to-end

set -e

echo "================================================"
echo "Running Complete MLOps Pipeline"
echo "================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pipeline_run_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "📝 Logging to: $LOG_DIR"

# Step 1: Data Preparation
echo ""
echo "▶ Step 1: Data Preparation"
python -m src.pipelines.data_prep \
    --input-datastore-path "UI/2026-04-04_090211_UTC/creditcard_train.csv" \
    --local-data-dir "./data" \
    --output-dir "./outputs/prepared_data" \
    2>&1 | tee $LOG_DIR/data_prep.log

# Step 2: Model Training
echo ""
echo "▶ Step 2: Model Training"
python -m src.pipelines.train \
    --prepared-data-dir "./outputs/prepared_data" \
    --output-dir "./outputs" \
    --mlflow-experiment "CreditCard_Fraud_Detection" \
    2>&1 | tee $LOG_DIR/train.log

# Step 3: Model Evaluation
echo ""
echo "▶ Step 3: Model Evaluation"
python -m src.pipelines.evaluate \
    --model-path "./outputs/model.pkl" \
    --test-data-dir "./outputs" \
    --output-dir "./outputs" \
    --thresholds-config "./config/thresholds.yaml" \
    2>&1 | tee $LOG_DIR/evaluate.log

# Step 4: Champion/Challenger Selection
echo ""
echo "▶ Step 4: Champion/Challenger Selection"
python -m src.pipelines.champion_challenger \
    --model-path "./outputs/model.pkl" \
    --metrics-path "./outputs/metrics.json" \
    --model-name "CreditCard_Fraud_Detection" \
    --output-dir "./outputs" \
    2>&1 | tee $LOG_DIR/champion_challenger.log

# Step 5: Batch Inference (optional - only if batch data exists)
if [ -f "./data/batch_input.csv" ] || [ -f "./outputs/batch_results/batch_predictions.csv" ]; then
    echo ""
    echo "▶ Step 5: Batch Inference"
    python -m src.pipelines.batch_inference \
        --batch-input-path "UI/2026-04-17_180302_UTC/creditcard_batch_input.csv" \
        --output-dir "./outputs/batch_results" \
        --model-name "CreditCard_Fraud_Detection" \
        2>&1 | tee $LOG_DIR/batch_inference.log || true
fi

# Step 6: Model Monitoring (optional - only if batch results exist)
if [ -f "./outputs/batch_results/batch_predictions.csv" ]; then
    echo ""
    echo "▶ Step 6: Model Monitoring"
    python -m src.pipelines.model_monitoring \
        --batch-results-path "./outputs/batch_results/batch_predictions.csv" \
        --reference-data-dir "./outputs" \
        --output-dir "./outputs/batch_results/model_monitoring" \
        2>&1 | tee $LOG_DIR/monitoring.log || true
fi

echo ""
echo "================================================"
echo "✅ Pipeline execution completed!"
echo "================================================"
echo ""
echo "📊 Results:"
echo "   - Training logs: $LOG_DIR/train.log"
echo "   - Evaluation report: outputs/evaluation_report.json"
echo "   - Model artifacts: outputs/"
echo "   - Batch results: outputs/batch_results/"
echo ""
