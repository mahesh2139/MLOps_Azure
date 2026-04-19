"""
Model training and evaluation components.
"""
import logging
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train machine learning models."""
    
    def __init__(self, model_params: Dict[str, Any], mlflow_enabled: bool = True):
        """
        Initialize trainer.
        
        Args:
            model_params: Model hyperparameters
            mlflow_enabled: Whether to log to MLflow
        """
        self.model_params = model_params
        self.mlflow_enabled = mlflow_enabled
        self.model = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        run_name: str = "model_training"
    ) -> Any:
        """
        Train model.
        
        Args:
            X_train: Training features
            y_train: Training target
            run_name: MLflow run name
            
        Returns:
            Trained model
        """
        logger.info("Training RandomForestClassifier...")
        
        if self.mlflow_enabled:
            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                for param_name, param_value in self.model_params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Train
                self.model = RandomForestClassifier(**self.model_params)
                self.model.fit(X_train, y_train)
                
                logger.info(f"✅ Model trained. MLflow Run ID: {run.info.run_id}")
        else:
            self.model = RandomForestClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
            logger.info("✅ Model trained")
        
        return self.model


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        
        logger.info("📊 Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"   {metric_name}: {metric_value:.4f}")
            if mlflow.active_run():
                mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: list, top_n: int = 10) -> Dict[str, float]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance scores
        """
        importances = model.feature_importances_
        feature_importance = {
            name: importance
            for name, importance in zip(feature_names, importances)
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:top_n])
        
        logger.info(f"Top {top_n} features:")
        for feature, importance in top_features.items():
            logger.info(f"   {feature}: {importance:.4f}")
        
        return top_features
    
    @staticmethod
    def get_detailed_report(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Generate detailed evaluation report with classification metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Detailed report dictionary
        """
        y_pred = model.predict(X_test)
        
        report = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        
        return report


class ModelSaver:
    """Save model artifacts."""
    
    @staticmethod
    def save_model(
        model: Any,
        output_dir: str,
        model_name: str = "model.pkl"
    ) -> Path:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            output_dir: Directory to save model
            model_name: Name of model file
            
        Returns:
            Path to saved model
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = Path(output_dir) / model_name
        joblib.dump(model, str(model_path))
        
        logger.info(f"✅ Model saved to: {model_path}")
        return model_path
    
    @staticmethod
    def save_artifacts(
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Dict[str, float],
        output_dir: str
    ):
        """
        Save all training artifacts.
        
        Args:
            X_test: Test features
            y_test: Test target
            metrics: Evaluation metrics
            output_dir: Directory to save artifacts
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save test data
        joblib.dump(X_test, Path(output_dir) / "x_test.pkl")
        joblib.dump(y_test, Path(output_dir) / "y_test.pkl")
        
        # Save metrics
        with open(Path(output_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"✅ Artifacts saved to: {output_dir}")
