"""
Automated model retraining pipeline triggered by performance degradation.
Detects model performance drift and triggers automatic retraining.
"""
import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from src.components.logger import setup_logger, log_section, log_step
from src.components.model_trainer import ModelTrainer, ModelEvaluator, ModelSaver
from src.components.model_registry import ModelRegistry

logger = setup_logger(__name__, log_file="model_retraining.log")


class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(
        self,
        metrics_history_path: str = "./outputs/metrics_history.json",
        performance_threshold: float = 0.05
    ):
        """
        Initialize performance monitor.
        
        Args:
            metrics_history_path: Path to store metrics history
            performance_threshold: Acceptable performance drop percentage (default: 5%)
        """
        self.metrics_history_path = Path(metrics_history_path)
        self.performance_threshold = performance_threshold
        self.load_history()
        
    def load_history(self):
        """Load metrics history from file."""
        if self.metrics_history_path.exists():
            with open(self.metrics_history_path, 'r') as f:
                self.history = json.load(f)
            logger.info(f"Loaded metrics history with {len(self.history)} records")
        else:
            self.history = []
            logger.info("Starting new metrics history")
    
    def save_history(self):
        """Save metrics history to file."""
        self.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved metrics history with {len(self.history)} records")
    
    def record_metrics(
        self,
        metrics: Dict[str, float],
        model_version: str,
        model_name: str = "champion",
        environment: str = "production"
    ):
        """
        Record model metrics in history.
        
        Args:
            metrics: Dictionary of metrics (accuracy, precision, recall, f1, roc_auc)
            model_version: Version of the model
            model_name: Name of the model (champion, challenger, retrained)
            environment: Environment where model is deployed (production, staging)
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "model_name": model_name,
            "environment": environment,
            "metrics": metrics
        }
        
        self.history.append(record)
        self.save_history()
        logger.info(f"Recorded metrics for {model_name} v{model_version}")
    
    def detect_performance_degradation(
        self,
        current_metrics: Dict[str, float],
        lookback_periods: int = 5,
        primary_metric: str = "f1_score"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if model performance has degraded.
        
        Args:
            current_metrics: Current model metrics
            lookback_periods: Number of historical records to compare against
            primary_metric: Primary metric to monitor (f1_score, accuracy, roc_auc)
            
        Returns:
            Tuple of (degradation_detected, degradation_report)
        """
        if len(self.history) < lookback_periods:
            logger.info(f"Insufficient history ({len(self.history)} < {lookback_periods}) to detect degradation")
            return False, {"reason": "insufficient_history", "history_size": len(self.history)}
        
        # Get historical averages
        recent_records = self.history[-lookback_periods:]
        historical_metrics = {
            metric: np.mean([r['metrics'].get(metric, 0) for r in recent_records])
            for metric in current_metrics.keys()
        }
        
        # Calculate performance drops
        degradation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_periods": lookback_periods,
            "primary_metric": primary_metric,
            "metric_changes": {},
            "max_degradation": 0,
            "degradation_detected": False,
            "reason": None
        }
        
        current_primary = current_metrics.get(primary_metric, 0)
        historical_primary = historical_metrics.get(primary_metric, 0)
        
        # Calculate percentage changes
        for metric, historical_value in historical_metrics.items():
            current_value = current_metrics.get(metric, 0)
            
            if historical_value > 0:
                percentage_change = ((current_value - historical_value) / historical_value) * 100
            else:
                percentage_change = 0
            
            degradation_report["metric_changes"][metric] = {
                "historical_average": float(historical_value),
                "current_value": float(current_value),
                "percentage_change": float(percentage_change),
                "degraded": percentage_change < (-self.performance_threshold * 100)
            }
            
            # Track maximum degradation
            if percentage_change < 0:
                degradation_report["max_degradation"] = min(
                    degradation_report["max_degradation"],
                    percentage_change
                )
        
        # Detect degradation
        primary_degraded = degradation_report["metric_changes"][primary_metric]["degraded"]
        avg_degradation = np.mean([
            m["percentage_change"] for m in degradation_report["metric_changes"].values()
        ])
        
        if primary_degraded or avg_degradation < (-self.performance_threshold * 100):
            degradation_report["degradation_detected"] = True
            degradation_report["reason"] = f"{primary_metric} degraded by {abs(primary_degraded and degradation_report['metric_changes'][primary_metric]['percentage_change'] or 0):.2f}%"
            logger.warning(f"⚠️ Performance degradation detected: {degradation_report['reason']}")
            return True, degradation_report
        else:
            logger.info("✅ No performance degradation detected")
            return False, degradation_report


class AutomatedRetrainingEngine:
    """Automated model retraining triggered by performance degradation."""
    
    def __init__(
        self,
        model_name: str = "CreditCard_Fraud_Detection",
        output_dir: str = "./outputs",
        retrain_threshold: float = 0.05
    ):
        """
        Initialize retraining engine.
        
        Args:
            model_name: Name of the model
            output_dir: Output directory for artifacts
            retrain_threshold: Performance threshold triggering retraining (default: 5%)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.monitor = ModelPerformanceMonitor(
            metrics_history_path=str(self.output_dir / "metrics_history.json"),
            performance_threshold=retrain_threshold
        )
        
    def should_retrain(
        self,
        current_metrics: Dict[str, float],
        lookback_periods: int = 5
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if model should be retrained based on performance.
        
        Args:
            current_metrics: Current model metrics
            lookback_periods: Number of historical periods to check
            
        Returns:
            Tuple of (should_retrain, degradation_report)
        """
        degradation_detected, report = self.monitor.detect_performance_degradation(
            current_metrics,
            lookback_periods=lookback_periods
        )
        return degradation_detected, report
    
    def prepare_retraining_data(
        self,
        recent_data_path: str,
        prepared_data_dir: str = "./outputs/prepared_data"
    ) -> bool:
        """
        Prepare data for retraining using recent production data.
        
        Args:
            recent_data_path: Path to recent production data
            prepared_data_dir: Directory to save prepared data
            
        Returns:
            True if data preparation successful
        """
        logger.info("🔄 Preparing data for retraining...")
        
        try:
            # Load recent data
            if isinstance(recent_data_path, str):
                if recent_data_path.endswith('.csv'):
                    recent_data = pd.read_csv(recent_data_path)
                elif recent_data_path.endswith('.pkl'):
                    recent_data = joblib.load(recent_data_path)
                else:
                    raise ValueError(f"Unsupported file format: {recent_data_path}")
            else:
                recent_data = recent_data_path
            
            logger.info(f"Loaded recent data: {recent_data.shape}")
            
            # Data validation and preprocessing
            # Remove target column if present for splitting
            target_col = 'Class' if 'Class' in recent_data.columns else None
            
            if target_col:
                y = recent_data[target_col]
                X = recent_data.drop(columns=[target_col])
            else:
                logger.warning("No target column found, using last column as target")
                X = recent_data.iloc[:, :-1]
                y = recent_data.iloc[:, -1]
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(np.unique(y)) <= 10 else None
            )
            
            # Save prepared data
            prepared_dir = Path(prepared_data_dir)
            prepared_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as both pickle and CSV for compatibility
            joblib.dump(X_train, prepared_dir / "x_train.pkl")
            joblib.dump(y_train, prepared_dir / "y_train.pkl")
            joblib.dump(X_test, prepared_dir / "x_test.pkl")
            joblib.dump(y_test, prepared_dir / "y_test.pkl")
            
            X_train.to_csv(prepared_dir / "x_train.csv", index=False)
            X_test.to_csv(prepared_dir / "x_test.csv", index=False)
            pd.Series(y_train).to_csv(prepared_dir / "y_train.csv", index=False)
            pd.Series(y_test).to_csv(prepared_dir / "y_test.csv", index=False)
            
            logger.info(f"✅ Prepared retraining data: train {X_train.shape}, test {X_test.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare retraining data: {str(e)}", exc_info=True)
            return False
    
    def retrain_model(
        self,
        prepared_data_dir: str = "./outputs/prepared_data",
        model_params: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Retrain model with new data.
        
        Args:
            prepared_data_dir: Directory with prepared training data
            model_params: Model hyperparameters
            
        Returns:
            Tuple of (success, retraining_result)
        """
        logger.info("🤖 Starting automated model retraining...")
        
        try:
            # Default parameters
            if model_params is None:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            # Load data
            prepared_dir = Path(prepared_data_dir)
            try:
                X_train = joblib.load(prepared_dir / "x_train.pkl")
                y_train = joblib.load(prepared_dir / "y_train.pkl")
                X_test = joblib.load(prepared_dir / "x_test.pkl")
                y_test = joblib.load(prepared_dir / "y_test.pkl")
            except:
                logger.info("Pickle loading failed, using CSV fallback")
                X_train = pd.read_csv(prepared_dir / "x_train.csv")
                y_train = pd.read_csv(prepared_dir / "y_train.csv").iloc[:, 0]
                X_test = pd.read_csv(prepared_dir / "x_test.csv")
                y_test = pd.read_csv(prepared_dir / "y_test.csv").iloc[:, 0]
            
            # Train model
            trainer = ModelTrainer(model_params=model_params, mlflow_enabled=False)
            model = trainer.train(X_train, y_train, run_name="automated_retraining")
            
            # Evaluate
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            # Save model
            self.output_dir.mkdir(parents=True, exist_ok=True)
            retrain_dir = self.output_dir / "retraining"
            retrain_dir.mkdir(parents=True, exist_ok=True)
            
            ModelSaver.save_model(model, str(retrain_dir), "retrained_model.pkl")
            
            # Save metrics
            result = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "model_path": str(retrain_dir / "retrained_model.pkl"),
                "metrics": metrics,
                "data_samples": {
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                }
            }
            
            with open(retrain_dir / "retraining_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"✅ Model retraining completed successfully")
            logger.info(f"New metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            # Record in history
            self.monitor.record_metrics(
                metrics,
                model_version="retrained",
                model_name="retrained",
                environment="staging"
            )
            
            return True, result
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {str(e)}", exc_info=True)
            return False, None
    
    def promote_retrained_model(
        self,
        retrained_model_path: str,
        metrics: Dict[str, float],
        approval_required: bool = False
    ) -> Dict[str, Any]:
        """
        Promote retrained model to production.
        
        Args:
            retrained_model_path: Path to retrained model
            metrics: Model metrics
            approval_required: Require approval before promotion
            
        Returns:
            Promotion result
        """
        logger.info("📦 Promoting retrained model...")
        
        result = {
            "status": "pending_approval" if approval_required else "promoted",
            "timestamp": datetime.utcnow().isoformat(),
            "model_path": retrained_model_path,
            "metrics": metrics,
            "approval_required": approval_required,
            "reason": "Automated retraining due to performance degradation"
        }
        
        if approval_required:
            logger.info("⏳ Awaiting approval for model promotion...")
        else:
            logger.info("✅ Model promoted to production automatically")
            
            # Copy to main model location
            champion_model_path = self.output_dir / "model.pkl"
            if champion_model_path.exists():
                champion_model_path.unlink()  # Remove old champion
            
            import shutil
            shutil.copy(retrained_model_path, champion_model_path)
            logger.info(f"Copied retrained model to: {champion_model_path}")
        
        return result


def run_automated_retraining(
    current_metrics: Dict[str, float],
    recent_data_path: Optional[str] = None,
    model_name: str = "CreditCard_Fraud_Detection",
    output_dir: str = "./outputs",
    auto_promote: bool = False
):
    """
    End-to-end automated retraining pipeline.
    
    Args:
        current_metrics: Current model metrics
        recent_data_path: Path to recent production data for retraining
        model_name: Name of the model
        output_dir: Output directory
        auto_promote: Automatically promote retrained model
        
    Returns:
        Retraining result
    """
    log_section(logger, "AUTOMATED MODEL RETRAINING PIPELINE")
    
    try:
        # Initialize retraining engine
        engine = AutomatedRetrainingEngine(
            model_name=model_name,
            output_dir=output_dir,
            retrain_threshold=0.05
        )
        
        # Check if retraining needed
        log_step(logger, "Analyzing model performance for degradation")
        should_retrain, degradation_report = engine.should_retrain(current_metrics)
        
        if not should_retrain:
            logger.info("No retraining needed - model performing well")
            return {
                "status": "no_action",
                "reason": "Model performance is stable",
                "metrics": current_metrics
            }
        
        logger.warning(f"⚠️ Retraining triggered: {degradation_report.get('reason')}")
        
        # Prepare data if provided
        if recent_data_path:
            log_step(logger, "Preparing retraining data")
            data_prepared = engine.prepare_retraining_data(
                recent_data_path,
                prepared_data_dir=str(Path(output_dir) / "prepared_data_retrain")
            )
            
            if not data_prepared:
                logger.error("Failed to prepare retraining data")
                return {"status": "failed", "reason": "Data preparation failed"}
            
            # Retrain model
            log_step(logger, "Retraining model with recent data")
            success, retrain_result = engine.retrain_model(
                prepared_data_dir=str(Path(output_dir) / "prepared_data_retrain")
            )
            
            if not success:
                logger.error("Model retraining failed")
                return {"status": "failed", "reason": "Model training failed"}
            
            # Promote model if requested
            if auto_promote:
                promotion_result = engine.promote_retrained_model(
                    retrain_result["model_path"],
                    retrain_result["metrics"],
                    approval_required=False
                )
                retrain_result["promotion"] = promotion_result
        
        logger.info("✅ Automated retraining pipeline completed successfully")
        return retrain_result
        
    except Exception as e:
        logger.error(f"❌ Automated retraining pipeline failed: {str(e)}", exc_info=True)
        raise
