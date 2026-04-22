"""
Model training pipeline step.
"""
#Github Action - Pipeline Test11

import logging
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from azureml.core import Workspace
from src.components.logger import setup_logger, log_section, log_step
from src.components.model_trainer import ModelTrainer, ModelEvaluator, ModelSaver

logger = setup_logger(__name__, log_file="train.log")


def run_training(
    prepared_data_dir: str,
    output_dir: str,
    model_params: dict = None,
    mlflow_experiment: str = "CreditCard_Fraud_Detection",
    mlflow_enabled: bool = True
):
    """
    Train model.
    
    Args:
        prepared_data_dir: Directory with prepared data (x_train, y_train, etc.)
        output_dir: Directory to save model and artifacts
        model_params: Model hyperparameters
        mlflow_experiment: MLflow experiment name
        mlflow_enabled: Whether to use MLflow logging
    """
    log_section(logger, "MODEL TRAINING PIPELINE")
    
    try:
        # Default model parameters
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Load prepared data
        log_step(logger, "Loading prepared data")
        prepared_dir = Path(prepared_data_dir)
        
        # Try loading pickle files first, fallback to CSV
        try:
            logger.info("Attempting to load pickle files...")
            x_train = joblib.load(prepared_dir / "x_train.pkl")
            y_train = joblib.load(prepared_dir / "y_train.pkl")
            x_test = joblib.load(prepared_dir / "x_test.pkl")
            y_test = joblib.load(prepared_dir / "y_test.pkl")
            logger.info("✅ Loaded data from pickle files")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pickle files: {str(e)}")
            logger.info("Attempting CSV fallback...")
            
            try:
                # Try loading from CSV files in prepared_data directory
                import pandas as pd
                x_train = pd.read_csv(prepared_dir / "x_train.csv") if (prepared_dir / "x_train.csv").exists() else None
                y_train = pd.read_csv(prepared_dir / "y_train.csv") if (prepared_dir / "y_train.csv").exists() else None
                x_test = pd.read_csv(prepared_dir / "x_test.csv") if (prepared_dir / "x_test.csv").exists() else None
                y_test = pd.read_csv(prepared_dir / "y_test.csv") if (prepared_dir / "y_test.csv").exists() else None
                
                # Convert CSV back to numpy arrays for compatibility
                if x_train is not None and isinstance(x_train, pd.DataFrame):
                    x_train = x_train.values if x_train.shape[1] > 1 else x_train.iloc[:, 0].values
                if y_train is not None and isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[:, 0].values if y_train.shape[1] == 1 else y_train.values
                if x_test is not None and isinstance(x_test, pd.DataFrame):
                    x_test = x_test.values if x_test.shape[1] > 1 else x_test.iloc[:, 0].values
                if y_test is not None and isinstance(y_test, pd.DataFrame):
                    y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values
                    
                if all([x_train is not None, y_train is not None, x_test is not None, y_test is not None]):
                    logger.info("✅ Loaded data from CSV files")
                else:
                    raise ValueError("Not all data files found in CSV format")
            except Exception as csv_error:
                logger.error(f"❌ Both pickle and CSV loading failed: {str(csv_error)}")
                raise ValueError(f"Could not load training data: {str(e)} (pickle) and {str(csv_error)} (CSV)")
        
        logger.info(f"Loaded training data: X_train {x_train.shape}, X_test {x_test.shape}")
        
        # Setup MLflow
        if mlflow_enabled:
            try:
                workspace = Workspace.from_config()
                mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
            except:
                logger.warning("Could not connect to Azure ML workspace for MLflow")
            
            mlflow.set_experiment(mlflow_experiment)
        
        # Train model
        log_step(logger, "Training RandomForest classifier")
        trainer = ModelTrainer(model_params=model_params, mlflow_enabled=mlflow_enabled)
        model = trainer.train(x_train, y_train, run_name="model_training")
        
        # Evaluate
        log_step(logger, "Evaluating model on test set")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, x_test, y_test)
        
        # Feature importance
        log_step(logger, "Extracting feature importance")
        feature_importance = evaluator.get_feature_importance(
            model,
            feature_names=x_test.columns.tolist(),
            top_n=10
        )
        
        # Save model and artifacts
        log_step(logger, "Saving model and artifacts")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        ModelSaver.save_model(model, str(output_path), "model.pkl")
        ModelSaver.save_artifacts(x_test, y_test, metrics, str(output_path))
        
        # Save feature importance
        with open(output_path / "feature_importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)
        
        logger.info(f"✅ Model training completed successfully")
        logger.info(f"Output saved to: {output_dir}")
        
        result = {
            "status": "success",
            "model_path": str(output_path / "model.pkl"),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "output_dir": str(output_path)
        }
        
        # Save result
        with open(output_path / "training_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data-dir", type=str, default="./outputs/prepared_data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--mlflow-experiment", type=str, 
                       default="CreditCard_Fraud_Detection")
    
    args = parser.parse_args()
    
    result = run_training(
        prepared_data_dir=args.prepared_data_dir,
        output_dir=args.output_dir,
        mlflow_experiment=args.mlflow_experiment
    )
