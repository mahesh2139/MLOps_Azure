"""
Model validation and governance components.
"""
import logging
from typing import Dict, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsValidator:
    """Validate model metrics against thresholds."""
    
    def __init__(self, thresholds_config: str = "./config/thresholds.yaml"):
        """
        Initialize validator with thresholds.
        
        Args:
            thresholds_config: Path to thresholds YAML file
        """
        self.thresholds_config = thresholds_config
        self.thresholds = self._load_thresholds()
    
    def _load_thresholds(self) -> Dict:
        """Load thresholds from YAML config."""
        if not Path(self.thresholds_config).exists():
            logger.warning(f"Thresholds file not found: {self.thresholds_config}")
            return self._get_default_thresholds()
        
        with open(self.thresholds_config, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('model_validation', self._get_default_thresholds())
    
    @staticmethod
    def _get_default_thresholds() -> Dict:
        """Get default thresholds if config not found."""
        return {
            'accuracy': {'min_threshold': 0.60},
            'precision': {'min_threshold': 0.60},
            'recall': {'min_threshold': 0.60},
            'f1_score': {'min_threshold': 0.60},
        }
    
    def validate(self, metrics: Dict[str, float], fail_on_error: bool = True) -> Tuple[bool, Dict]:
        """
        Validate metrics against thresholds.
        
        Args:
            metrics: Dictionary of metrics
            fail_on_error: Raise exception if validation fails
            
        Returns:
            Tuple of (passed: bool, validation_results: dict)
        """
        logger.info("📊 Validating model metrics...")
        
        validation_results = {}
        all_passed = True
        
        for metric_name, threshold_config in self.thresholds.items():
            if metric_name not in metrics:
                logger.warning(f"⚠️ Metric '{metric_name}' not found in results")
                validation_results[metric_name] = {"status": "missing", "value": None}
                all_passed = False
                continue
            
            metric_value = metrics[metric_name]
            min_threshold = threshold_config.get('min_threshold', 0.0)
            
            passed = metric_value >= min_threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            
            validation_results[metric_name] = {
                "status": passed,
                "value": metric_value,
                "threshold": min_threshold,
                "delta": metric_value - min_threshold
            }
            
            logger.info(f"{status}: {metric_name} = {metric_value:.4f} (threshold: {min_threshold})")
            
            if not passed:
                all_passed = False
        
        if not all_passed and fail_on_error:
            raise ValueError("❌ Model metrics validation failed. Model does not meet production thresholds.")
        
        return all_passed, validation_results


class GovernanceGate:
    """Governance checkpoints for model promotion."""
    
    def __init__(self, config_path: str = "./config/thresholds.yaml"):
        """
        Initialize governance gate.
        
        Args:
            config_path: Path to governance config
        """
        self.config_path = config_path
        self.approval_gates = self._load_approval_gates()
    
    def _load_approval_gates(self) -> Dict:
        """Load approval gate requirements."""
        if not Path(self.config_path).exists():
            return self._get_default_gates()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('approval_gates', self._get_default_gates())
    
    @staticmethod
    def _get_default_gates() -> Dict:
        """Get default approval gates."""
        return {
            'require_metrics_validation': True,
            'require_explainability_review': True,
            'require_manual_approval': False,
            'approval_timeout_hours': 24
        }
    
    def check_gates(self, validation_passed: bool, has_explainability: bool = True) -> Tuple[bool, Dict]:
        """
        Check all governance gates.
        
        Args:
            validation_passed: Whether metrics validation passed
            has_explainability: Whether explainability data is available
            
        Returns:
            Tuple of (all_gates_passed: bool, gate_status: dict)
        """
        logger.info("🔐 Checking governance gates...")
        
        gate_status = {
            "metrics_validation": validation_passed if self.approval_gates.get('require_metrics_validation') else True,
            "explainability_review": has_explainability if self.approval_gates.get('require_explainability_review') else True,
            "manual_approval_required": self.approval_gates.get('require_manual_approval', False),
        }
        
        for gate_name, gate_passed in gate_status.items():
            status = "✅" if gate_passed else "⚠️"
            logger.info(f"{status} {gate_name}: {gate_passed}")
        
        all_passed = all(gate_status.values()) and not gate_status.get("manual_approval_required", False)
        
        if not all_passed:
            logger.warning("⚠️ Some governance gates require attention before promotion")
        else:
            logger.info("✅ All governance gates passed")
        
        return all_passed, gate_status


class ExplainabilityValidator:
    """Validate model explainability metrics."""
    
    @staticmethod
    def validate_feature_importance(
        feature_importance: Dict[str, float],
        min_features: int = 10
    ) -> Tuple[bool, Dict]:
        """
        Validate that sufficient features have importance scores tracked.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            min_features: Minimum number of important features to track
            
        Returns:
            Tuple of (validation_passed: bool, report: dict)
        """
        logger.info("🔍 Validating explainability...")
        
        num_features = len(feature_importance)
        passed = num_features >= min_features
        
        report = {
            "features_tracked": num_features,
            "min_required": min_features,
            "passed": passed,
            "top_features": list(feature_importance.keys())[:5]
        }
        
        status = "✅" if passed else "⚠️"
        logger.info(f"{status} Features tracked: {num_features} (minimum: {min_features})")
        
        return passed, report
