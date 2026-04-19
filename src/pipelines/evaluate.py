"""
Model evaluation and validation pipeline step.
"""
import logging
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from src.components.logger import setup_logger, log_section, log_step
from src.components.model_trainer import ModelEvaluator
from src.components.model_validator import MetricsValidator, ExplainabilityValidator, GovernanceGate

logger = setup_logger(__name__, log_file="evaluate.log")


def run_evaluation(
    model_path: str,
    test_data_dir: str,
    output_dir: str,
    thresholds_config: str = "./config/thresholds.yaml"
):
    """
    Evaluate model and validate against thresholds.
    
    Args:
        model_path: Path to trained model
        test_data_dir: Directory with test data
        output_dir: Directory to save evaluation results
        thresholds_config: Path to thresholds YAML
    """
    log_section(logger, "MODEL EVALUATION PIPELINE")
    
    try:
        # Load model and data
        log_step(logger, "Loading model and test data")
        
        # Load model with error handling
        try:
            model = joblib.load(model_path)
            logger.info("✅ Loaded model successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise ValueError(f"Could not load model from {model_path}: {str(e)}")
        
        # Load test data with fallback
        test_data_path = Path(test_data_dir)
        try:
            logger.info("Attempting to load pickle files...")
            x_test = joblib.load(test_data_path / "x_test.pkl")
            y_test = joblib.load(test_data_path / "y_test.pkl")
            logger.info("✅ Loaded test data from pickle files")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pickle files: {str(e)}")
            logger.info("Attempting CSV fallback...")
            
            try:
                import pandas as pd
                x_test = pd.read_csv(test_data_path / "x_test.csv") if (test_data_path / "x_test.csv").exists() else None
                y_test = pd.read_csv(test_data_path / "y_test.csv") if (test_data_path / "y_test.csv").exists() else None
                
                if x_test is not None:
                    x_test = pd.DataFrame(x_test) if not isinstance(x_test, pd.DataFrame) else x_test
                if y_test is not None:
                    y_test = pd.Series(y_test.iloc[:, 0]) if isinstance(y_test, pd.DataFrame) else pd.Series(y_test)
                    
                if x_test is None or y_test is None:
                    raise FileNotFoundError("Could not find test data files")
                    
                logger.info("✅ Loaded test data from CSV files")
            except Exception as csv_error:
                logger.error(f"❌ Both pickle and CSV loading failed")
                raise ValueError(f"Could not load test data: pickle error: {str(e)}, CSV error: {str(csv_error)}")
        
        # Ensure x_test is a DataFrame for feature names
        if not isinstance(x_test, pd.DataFrame):
            import pandas as pd
            x_test = pd.DataFrame(x_test)
        
        # Evaluate
        log_step(logger, "Evaluating model performance")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, x_test, y_test)
        
        # Feature importance
        log_step(logger, "Extracting feature importance for explainability")
        feature_importance = evaluator.get_feature_importance(
            model,
            feature_names=x_test.columns.tolist(),
            top_n=10
        )
        
        # Validate metrics
        log_step(logger, "Validating metrics against thresholds")
        validator = MetricsValidator(thresholds_config=thresholds_config)
        metrics_passed, validation_results = validator.validate(metrics, fail_on_error=False)
        
        # Validate explainability
        log_step(logger, "Validating explainability")
        explainability_passed, explainability_report = ExplainabilityValidator.validate_feature_importance(
            feature_importance,
            min_features=10
        )
        
        # Check governance gates
        log_step(logger, "Checking governance gates")
        governance = GovernanceGate(config_path=thresholds_config)
        gates_passed, gate_status = governance.check_gates(
            validation_passed=metrics_passed,
            has_explainability=explainability_passed
        )
        
        # Prepare evaluation report
        evaluation_report = {
            "status": "passed" if (metrics_passed and explainability_passed) else "failed",
            "metrics": metrics,
            "validation_results": validation_results,
            "feature_importance": feature_importance,
            "explainability_report": explainability_report,
            "governance_gates": gate_status,
            "promotion_eligible": gates_passed
        }
        
        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        logger.info(f"✅ Evaluation completed")
        logger.info(f"Promotion eligible: {gates_passed}")
        
        return evaluation_report
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./outputs/model.pkl")
    parser.add_argument("--test-data-dir", type=str, default="./outputs")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--thresholds-config", type=str, default="./config/thresholds.yaml")
    
    args = parser.parse_args()
    
    result = run_evaluation(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        thresholds_config=args.thresholds_config
    )
