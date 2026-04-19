#!/bin/bash
# MLOps Pipeline Setup Script

set -e

echo "================================================"
echo "MLOps Pipeline Setup"
echo "================================================"

# Create directories
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p outputs/models
mkdir -p outputs/batch_results
mkdir -p outputs/batch_results/model_monitoring
mkdir -p outputs/prepared_data

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
echo "🔧 Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'
# Azure ML Configuration
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_WORKSPACE_NAME=your_workspace
AZURE_DATASTORE_NAME=workspaceblobstore

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=CreditCard_Fraud_Detection
MLFLOW_TRACKING_URI=http://localhost:5000

# Model Configuration
MODEL_NAME=CreditCard_Fraud_Detection
EOF
    echo "✅ .env template created. Please update with your configuration."
fi

# Download Azure ML config if in workspace
if [ -f "config.json" ]; then
    echo "✅ Found config.json (Azure ML workspace)"
else
    echo "⚠️  config.json not found. Make sure you're authenticated to Azure ML workspace."
fi

echo ""
echo "================================================"
echo "✅ Setup completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Update .env with your Azure configuration"
echo "2. Run: python -m src.pipelines.data_prep"
echo "3. Run: python -m src.pipelines.train"
echo "4. Run: python -m src.pipelines.evaluate"
echo "5. Run: python -m src.pipelines.champion_challenger"
echo ""
echo "For more information, see README.md"
