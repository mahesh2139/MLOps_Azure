"""
Batch inference pipeline.
"""
import logging
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from src.components.logger import setup_logger, log_section, log_step
from src.components.data_loader import DataLoader
from src.components.model_registry import ModelRegistry

logger = setup_logger(__name__, log_file="batch_inference.log")


def run_batch_inference(
    batch_input_path: str,
    output_dir: str,
    model_name: str = "CreditCard_Fraud_Detection",
    use_workspace: bool = True
):
    """
    Run batch inference using champion model.
    
    Args:
        batch_input_path: Path to batch input data in datastore
        output_dir: Directory to save batch results
        model_name: Name of model in registry
        use_workspace: Whether to use Azure ML workspace
    """
    log_section(logger, "BATCH INFERENCE PIPELINE")
    
    try:
        # Connect to workspace
        workspace = None
        if use_workspace:
            try:
                from azureml.core import Workspace
                # Try to load from config first
                try:
                    workspace = Workspace.from_config()
                except:
                    # Fallback: create workspace from environment variables
                    import os
                    subscription_id = os.environ.get('AZUREML_SUBSCRIPTION_ID')
                    resource_group = os.environ.get('AZUREML_RESOURCE_GROUP')
                    workspace_name = os.environ.get('AZUREML_WORKSPACE_NAME')
                    
                    # Try service principal authentication
                    tenant_id = os.environ.get('AZURE_TENANT_ID')
                    client_id = os.environ.get('AZURE_CLIENT_ID')
                    client_secret = os.environ.get('AZURE_CLIENT_SECRET')
                    
                    if all([subscription_id, resource_group, workspace_name]):
                        if tenant_id and client_id and client_secret:
                            # Use service principal
                            from azure.identity import ClientSecretCredential
                            credential = ClientSecretCredential(
                                tenant_id=tenant_id,
                                client_id=client_id,
                                client_secret=client_secret
                            )
                            workspace = Workspace.get(
                                name=workspace_name,
                                subscription_id=subscription_id,
                                resource_group=resource_group,
                                credential=credential
                            )
                            logger.info(f"✅ Connected to workspace via service principal: {workspace.name}")
                        else:
                            # Try default credential
                            from azure.identity import DefaultAzureCredential
                            credential = DefaultAzureCredential()
                            workspace = Workspace.get(
                                name=workspace_name,
                                subscription_id=subscription_id,
                                resource_group=resource_group,
                                credential=credential
                            )
                            logger.info(f"✅ Connected to workspace from env: {workspace.name}")
            except Exception as e:
                logger.warning(f"Could not load workspace: {str(e)}")
        
        # Load batch data
        log_step(logger, "Loading batch data")
        loader = DataLoader(workspace=workspace)
        
        if use_workspace and workspace:
            batch_df = loader.load_from_datastore(
                datastore_path=batch_input_path,
                local_download_dir="./temp_data"
            )
        else:
            batch_df = loader.load_from_local(batch_input_path)
        
        logger.info(f"Batch data shape: {batch_df.shape}")
        
        # Get champion model from registry
        if workspace:
            log_step(logger, "Getting champion model from registry")
            registry = ModelRegistry(workspace, model_name)
            champion = registry.get_champion_model()
            
            if not champion:
                logger.warning("No champion model found in registry. Using local model...")
                local_model_path = "./outputs/model.pkl"
                if not Path(local_model_path).exists():
                    raise ValueError("No champion model found in registry or locally")
                model_path = local_model_path
            else:
                log_step(logger, "Downloading champion model")
                model_dir = registry.download_model(champion, "./temp_model")
                model_files = list(Path(model_dir).glob("*.pkl"))
                
                if not model_files:
                    raise FileNotFoundError(f"No model file found in {model_dir}")
                
                model_path = model_files[0]
            
            # Load model with error handling
            try:
                logger.info(f"Loading model from: {model_path}")
                model = joblib.load(model_path)
                logger.info("✅ Loaded model successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load model: {str(e)}")
                logger.info("Attempting to load model from local cache...")
                fallback_path = "./outputs/model.pkl"
                if Path(fallback_path).exists():
                    try:
                        model = joblib.load(fallback_path)
                        logger.info("✅ Loaded model from local cache")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load from cache: {str(e2)}")
                        raise ValueError(f"Could not load model: {str(e)} (primary) and {str(e2)} (fallback)")
                else:
                    raise ValueError(f"Could not load model and fallback not available: {str(e)}")
            
            model_version = champion.version if champion else "local"
            logger.info(f"Loaded champion model version {model_version}")
        else:
            # Use local model
            logger.info("Workspace not available, using local model")
            local_model_path = "./outputs/model.pkl"
            if Path(local_model_path).exists():
                try:
                    model = joblib.load(local_model_path)
                    logger.info("✅ Loaded local model successfully")
                    model_version = "local"
                except Exception as e:
                    raise ValueError(f"Could not load local model: {str(e)}")
            else:
                raise ValueError("Workspace required for batch inference and no local model found")
        
        # Prepare data
        log_step(logger, "Preparing data for inference")
        target_col = 'Class'
        
        # Drop non-feature columns (ID, Time) if present
        cols_to_drop = []
        if 'ID' in batch_df.columns:
            cols_to_drop.append('ID')
        if 'Time' in batch_df.columns:
            cols_to_drop.append('Time')
        if target_col in batch_df.columns:
            cols_to_drop.append(target_col)
        
        if cols_to_drop:
            X = batch_df.drop(columns=cols_to_drop)
            y_true = batch_df[target_col] if target_col in batch_df.columns else None
            has_target = target_col in batch_df.columns
        else:
            if target_col in batch_df.columns:
                X = batch_df.drop(columns=[target_col])
                y_true = batch_df[target_col]
                has_target = True
            else:
                X = batch_df
                y_true = None
                has_target = False
        
        logger.info(f"Features shape: {X.shape}")
        
        # Run inference
        log_step(logger, "Running batch predictions")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        fraud_count = sum(predictions)
        fraud_pct = 100 * fraud_count / len(predictions)
        logger.info(f"Predictions: {fraud_count} frauds detected ({fraud_pct:.2f}%)")
        
        # Create results
        log_step(logger, "Creating results dataframe")
        results_df = batch_df.copy()
        results_df['prediction'] = predictions
        results_df['fraud_probability'] = probabilities
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "batch_predictions.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved predictions to: {results_file}")
        
        # Create summary
        summary = {
            "batch_size": len(results_df),
            "fraud_predictions": int(fraud_count),
            "fraud_percentage": fraud_pct,
            "model_name": model_name,
            "model_version": str(model_version),
            "has_ground_truth": has_target
        }
        
        if has_target:
            report = classification_report(y_true, predictions, output_dict=True)
            summary["classification_report"] = report
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_true, predictions))
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to: {summary_file}")
        logger.info(f"\n✅ Batch inference completed successfully")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Batch inference failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-input-path", type=str,
                       default="UI/2026-04-17_180302_UTC/creditcard_batch_input.csv")
    parser.add_argument("--output-dir", type=str, 
                       default="./outputs/batch_results")
    parser.add_argument("--model-name", type=str,
                       default="CreditCard_Fraud_Detection")
    
    args = parser.parse_args()
    
    result = run_batch_inference(
        batch_input_path=args.batch_input_path,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
