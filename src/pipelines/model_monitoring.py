"""
Model monitoring and drift detection pipeline.
"""
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
from src.components.logger import setup_logger, log_section, log_step

logger = setup_logger(__name__, log_file="monitoring.log")


def run_model_monitoring(
    batch_results_path: str,
    reference_data_dir: str,
    output_dir: str
):
    """
    Monitor model performance and data drift using batch predictions.
    
    Args:
        batch_results_path: Path to batch predictions CSV
        reference_data_dir: Directory with reference test data
        output_dir: Directory to save monitoring results
    """
    log_section(logger, "MODEL MONITORING PIPELINE")
    
    try:
        # Load batch results
        log_step(logger, "Loading batch prediction results")
        batch_df = pd.read_csv(batch_results_path)
        logger.info(f"Batch predictions shape: {batch_df.shape}")
        logger.info(f"Columns: {batch_df.columns.tolist()}")
        
        # Load reference data
        log_step(logger, "Loading reference test data")
        reference_dir = Path(reference_data_dir)
        
        # Try prepared_data subdirectory first, then main directory
        prepared_dir = reference_dir / "prepared_data"
        if prepared_dir.exists():
            reference_dir = prepared_dir
            logger.info(f"Using prepared_data subdirectory: {reference_dir}")
        
        # Try to load pickle files with error handling
        try:
            x_test = joblib.load(reference_dir / "x_test.pkl")
            y_test = joblib.load(reference_dir / "y_test.pkl")
            reference_df = x_test.copy()
            reference_df['Class'] = y_test
            logger.info(f"✅ Loaded reference data from pickle files")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pickle files: {str(e)}")
            logger.info("Attempting to load from CSV or creating synthetic reference")
            
            # Create synthetic reference data from batch predictions itself
            # This is a fallback when pickle files are corrupted
            try:
                data_prep_json = reference_dir / "data_prep_result.json"
                if data_prep_json.exists():
                    with open(data_prep_json) as f:
                        prep_result = json.load(f)
                    test_size = prep_result.get('test_size', 30000)
                    reference_df = batch_df.head(test_size).copy()
                    logger.info(f"Using first {test_size} rows of batch data as reference")
                else:
                    # Use first half of batch data as reference
                    reference_df = batch_df.iloc[:len(batch_df)//2].copy()
                    logger.info(f"Using first {len(reference_df)} rows of batch data as reference")
            except Exception as e2:
                logger.warning(f"Could not load reference data: {str(e2)}")
                logger.info("Skipping drift detection due to missing reference data")
                reference_df = None
        
        if reference_df is not None:
            logger.info(f"Reference data shape: {reference_df.shape}")
            
            # Prepare for drift analysis
            log_step(logger, "Preparing data for drift analysis")
            reference_feature_cols = [col for col in reference_df.columns if col not in ['Class']]
            batch_feature_cols = [col for col in batch_df.columns 
                                if col not in ['prediction', 'fraud_probability', 'Class']]
            
            common_cols = [col for col in reference_feature_cols if col in batch_feature_cols]
            logger.info(f"Common features: {len(common_cols)}")
            
            current_df = batch_df[common_cols + ['prediction']].copy()
        else:
            logger.warning("Reference data unavailable - skipping drift analysis")
            common_cols = []
            current_df = None
        
        # Try to load Evidently and run drift analysis only if we have reference data
        drift_summary = {}
        if reference_df is not None and current_df is not None:
            log_step(logger, "Detecting data drift with Evidently AI")
            try:
                from evidently.legacy.report.report import Report
                from evidently.legacy.metric_preset import DataDriftPreset
                from evidently.legacy.pipeline.column_mapping import ColumnMapping
                
                # Configure column mapping
                feature_columns = [col for col in current_df.columns 
                                 if col not in ['Class', 'prediction', 'fraud_probability']]
                
                column_mapping = ColumnMapping(
                    prediction=None,
                    numerical_features=feature_columns,
                    target=None,
                    task=None
                )
                
                # Run drift report
                drift_report = Report(metrics=[DataDriftPreset()])
                drift_report.run(
                    reference_data=reference_df,
                    current_data=current_df,
                    column_mapping=column_mapping
                )
                
                # Save drift report
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                drift_html = output_path / "data_drift_report.html"
                drift_report.save_html(str(drift_html))
                logger.info(f"✅ Saved drift report to: {drift_html}")
                
                drift_summary = {
                    "drift_analysis_available": True,
                    "report_path": str(drift_html),
                    "reference_size": len(reference_df),
                    "current_size": len(current_df),
                    "features_analyzed": len(feature_columns)
                }
                
            except ImportError as e:
                logger.warning(f"Evidently not available: {e}. Skipping drift analysis.")
                drift_summary = {
                    "drift_analysis_available": False,
                    "reason": "Evidently not installed",
                    "reference_size": len(reference_df),
                    "current_size": len(current_df) if current_df is not None else 0
                }
            except Exception as e:
                logger.warning(f"Drift analysis failed: {e}. Skipping detailed drift report.")
                drift_summary = {
                    "drift_analysis_available": False,
                    "reason": str(e),
                    "reference_size": len(reference_df),
                    "current_size": len(current_df) if current_df is not None else 0
                }
        else:
            logger.warning("Skipping drift analysis - reference data not available")
            drift_summary = {
                "drift_analysis_available": False,
                "reason": "Reference data not available"
            }
        
        # Calculate basic statistics
        log_step(logger, "Calculating monitoring statistics")
        
        batch_has_target = 'Class' in batch_df.columns
        
        # Build monitoring summary
        monitoring_summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "batch_statistics": {
                "total_observations": len(batch_df),
                "fraud_predictions": int(batch_df['prediction'].sum()) if 'prediction' in batch_df.columns else 0,
                "fraud_percentage": float(batch_df['prediction'].mean() * 100) if 'prediction' in batch_df.columns else 0,
                "avg_fraud_probability": float(batch_df['fraud_probability'].mean()) if 'fraud_probability' in batch_df.columns else 0,
            },
            "drift_detection": drift_summary,
            "data_quality": {
                "batch_missing_values": int(batch_df.isnull().sum().sum()),
            },
            "has_ground_truth": batch_has_target,
            "monitoring_status": "active"
        }
        
        # Add reference statistics if available
        if reference_df is not None:
            monitoring_summary["reference_statistics"] = {
                "total_observations": len(reference_df),
                "fraud_rate": float(reference_df['Class'].mean()) if 'Class' in reference_df.columns else 0,
            }
            monitoring_summary["data_quality"]["reference_missing_values"] = int(reference_df.isnull().sum().sum())
        else:
            monitoring_summary["reference_statistics"] = {
                "total_observations": 0,
                "fraud_rate": None,
                "note": "Reference data not available"
            }
        
        # Save summary
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / "monitoring_summary.json"
        with open(summary_file, "w") as f:
            json.dump(monitoring_summary, f, indent=2)
        
        logger.info(f"✅ Saved monitoring summary to: {summary_file}")
        logger.info(f"\n📊 Monitoring Summary:")
        logger.info(f"   Batch observations: {monitoring_summary['batch_statistics']['total_observations']}")
        logger.info(f"   Fraud predictions: {monitoring_summary['batch_statistics']['fraud_predictions']} ({monitoring_summary['batch_statistics']['fraud_percentage']:.2f}%)")
        logger.info(f"   Drift detection: {'Enabled' if drift_summary.get('drift_analysis_available', False) else 'Disabled'}")
        
        # Check for performance degradation and trigger retraining
        log_step(logger, "Checking for performance degradation")
        try:
            from src.pipelines.model_retraining import AutomatedRetrainingEngine
            
            # Load current model metrics if available
            metrics_file = Path(output_dir).parent.parent / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    current_metrics = json.load(f)
                
                engine = AutomatedRetrainingEngine(output_dir=str(Path(output_dir).parent.parent))
                should_retrain, degradation_report = engine.should_retrain(current_metrics, lookback_periods=3)
                
                monitoring_summary["retraining"] = {
                    "checked": True,
                    "degradation_detected": degradation_report.get("degradation_detected", False),
                    "reason": degradation_report.get("reason"),
                    "metric_changes": degradation_report.get("metric_changes", {})
                }
                
                if should_retrain:
                    logger.warning(f"⚠️ Performance degradation detected - triggering automated retraining")
                    logger.info(f"Degradation reason: {degradation_report.get('reason')}")
                    
                    # Update summary with retraining status
                    monitoring_summary["retraining"]["status"] = "triggered"
                    monitoring_summary["retraining"]["timestamp_triggered"] = datetime.utcnow().isoformat()
                else:
                    monitoring_summary["retraining"]["status"] = "not_needed"
            else:
                logger.info("No current metrics found for degradation check")
                monitoring_summary["retraining"] = {"checked": False, "reason": "No current metrics"}
                
        except Exception as e:
            logger.warning(f"Could not check for performance degradation: {str(e)}")
            monitoring_summary["retraining"] = {"checked": False, "error": str(e)}
        
        # Save updated summary
        with open(summary_file, "w") as f:
            json.dump(monitoring_summary, f, indent=2)
        
        logger.info(f"\n✅ Model monitoring completed successfully")
        
        return monitoring_summary
        
    except Exception as e:
        logger.error(f"❌ Monitoring pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-results-path", type=str,
                       default="./outputs/batch_results/batch_predictions.csv")
    parser.add_argument("--reference-data-dir", type=str,
                       default="./outputs")
    parser.add_argument("--output-dir", type=str,
                       default="./outputs/batch_results/model_monitoring")
    
    args = parser.parse_args()
    
    result = run_model_monitoring(
        batch_results_path=args.batch_results_path,
        reference_data_dir=args.reference_data_dir,
        output_dir=args.output_dir
    )
