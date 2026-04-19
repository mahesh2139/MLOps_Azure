"""
Azure ML Model Registry operations.
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from azureml.core import Workspace, Model

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage models in Azure ML registry."""
    
    def __init__(self, workspace: Workspace, model_name: str):
        """
        Initialize registry client.
        
        Args:
            workspace: Azure ML Workspace
            model_name: Name of model in registry
        """
        self.workspace = workspace
        self.model_name = model_name
    
    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        tags: Optional[Dict] = None,
        description: str = ""
    ) -> Model:
        """
        Register a new model version.
        
        Args:
            model_path: Local path to model file
            metrics: Dictionary of metrics to store as tags
            tags: Additional tags for the model
            description: Model description
            
        Returns:
            Registered Model object
        """
        logger.info(f"🚀 Registering model: {self.model_name}")
        
        # Prepare tags
        all_tags = {
            'framework': 'sklearn',
            'role': 'challenger',
            'status': 'staging',
        }
        
        # Add metrics as tags
        for metric_name, metric_value in metrics.items():
            all_tags[metric_name] = str(metric_value)
        
        # Add custom tags
        if tags:
            all_tags.update(tags)
        
        # Register
        registered_model = Model.register(
            workspace=self.workspace,
            model_path=model_path,
            model_name=self.model_name,
            tags=all_tags,
            description=description,
        )
        
        logger.info(f"✅ Model registered: {registered_model.name} v{registered_model.version}")
        logger.info(f"   Tags: {all_tags}")
        
        return registered_model
    
    def list_models(self, role: Optional[str] = None, status: Optional[str] = None) -> List[Model]:
        """
        List models from registry.
        
        Args:
            role: Filter by role ('champion', 'challenger', etc.)
            status: Filter by status ('production', 'staging', 'archived')
            
        Returns:
            List of Model objects
        """
        logger.info(f"Querying registry for {self.model_name}...")
        
        models = Model.list(self.workspace, name=self.model_name)
        
        filtered = []
        for model in models:
            tags = model.tags or {}
            
            if role and tags.get('role') != role:
                continue
            if status and tags.get('status') != status:
                continue
            
            filtered.append(model)
        
        logger.info(f"Found {len(filtered)} matching models")
        
        return filtered
    
    def get_champion_model(self) -> Optional[Model]:
        """
        Get current champion model.
        
        Returns:
            Champion Model or None if not found
        """
        champion_models = self.list_models(role='champion', status='production')
        
        if not champion_models:
            logger.warning("No champion model found")
            return None
        
        # Return latest version
        champion = max(champion_models, key=lambda m: int(m.version))
        logger.info(f"Found champion: {champion.name} v{champion.version}")
        
        return champion
    
    def promote_model(self, model: Model, new_role: str, new_status: str):
        """
        Update model role and status tags.
        
        Args:
            model: Model to promote
            new_role: New role (e.g., 'champion')
            new_status: New status (e.g., 'production')
        """
        logger.info(f"🔄 Promoting {model.name} v{model.version}")
        
        tags = model.tags or {}
        tags['role'] = new_role
        tags['status'] = new_status
        
        model.update(tags=tags)
        
        logger.info(f"✅ Updated tags: role={new_role}, status={new_status}")
    
    def download_model(self, model: Model, target_dir: str) -> Path:
        """
        Download model from registry to local directory.
        
        Args:
            model: Model to download
            target_dir: Directory to download to
            
        Returns:
            Path to downloaded model directory
        """
        logger.info(f"📥 Downloading {model.name} v{model.version}")
        
        model.download(target_dir=target_dir, exist_ok=True)
        
        logger.info(f"✅ Model downloaded to: {target_dir}")
        
        return Path(target_dir)
    
    def compare_models(
        self,
        model1: Model,
        model2: Model,
        metrics: List[str] = None
    ) -> Dict:
        """
        Compare two models on metrics.
        
        Args:
            model1: First model
            model2: Second model
            metrics: List of metric names to compare
            
        Returns:
            Comparison report
        """
        if not metrics:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        logger.info(f"📊 Comparing {model1.name} v{model1.version} vs v{model2.version}")
        
        tags1 = model1.tags or {}
        tags2 = model2.tags or {}
        
        comparison = {}
        for metric in metrics:
            val1 = float(tags1.get(metric, 0))
            val2 = float(tags2.get(metric, 0))
            
            delta = val1 - val2
            better = "MODEL1" if delta > 0 else "MODEL2" if delta < 0 else "TIE"
            
            comparison[metric] = {
                "model1": val1,
                "model2": val2,
                "delta": delta,
                "better": better
            }
            
            logger.info(f"   {metric}: {val1:.4f} vs {val2:.4f} ({delta:+.4f}) → {better}")
        
        return comparison
