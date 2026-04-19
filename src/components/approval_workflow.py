"""
Approval workflow management for regulated model promotions.
Handles creation and retrieval of approval records for model promotion decisions.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ApprovalWorkflow:
    """
    Manages approval workflow records for model promotion.
    Stores approval requests and decisions for audit compliance.
    """
    
    def __init__(self, approval_dir: str = "./outputs/approvals"):
        """
        Initialize approval workflow manager.
        
        Args:
            approval_dir: Directory to store approval records
        """
        self.approval_dir = Path(approval_dir)
        self.approval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Approval workflow initialized: {self.approval_dir}")
    
    def create_approval_record(
        self,
        approval_id: str,
        model_name: str,
        challenger_version: int,
        champion_version: int,
        status: str,
        comparison: Dict[str, Any],
        requester_email: str,
        comments: str = ""
    ) -> Dict[str, Any]:
        """
        Create an approval record for audit trail.
        
        Args:
            approval_id: Unique approval request ID
            model_name: Name of the model
            challenger_version: Version of challenger model
            champion_version: Version of current champion
            status: Status of approval (pending, approved, rejected)
            comparison: Comparison metrics between models
            requester_email: Email of person requesting approval
            comments: Additional comments
            
        Returns:
            Created approval record
        """
        record = {
            "approval_id": approval_id,
            "model_name": model_name,
            "challenger_version": challenger_version,
            "champion_version": champion_version,
            "status": status,
            "comparison": comparison,
            "requester_email": requester_email,
            "comments": comments,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "approved_by": None,
            "approval_timestamp": None
        }
        
        # Save record
        record_path = self.approval_dir / f"{approval_id}.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"Approval record created: {approval_id}")
        return record
    
    def get_approval_record(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an approval record by ID.
        
        Args:
            approval_id: Approval request ID
            
        Returns:
            Approval record or None if not found
        """
        record_path = self.approval_dir / f"{approval_id}.json"
        
        if not record_path.exists():
            logger.warning(f"Approval record not found: {approval_id}")
            return None
        
        with open(record_path, "r") as f:
            record = json.load(f)
        
        return record
    
    def update_approval_record(
        self,
        approval_id: str,
        status: str,
        approved_by: str = None,
        approval_comments: str = ""
    ) -> Dict[str, Any]:
        """
        Update approval record with decision.
        
        Args:
            approval_id: Approval request ID
            status: New status (approved, rejected)
            approved_by: Email of approver
            approval_comments: Comments on the approval decision
            
        Returns:
            Updated approval record
        """
        record_path = self.approval_dir / f"{approval_id}.json"
        
        if not record_path.exists():
            raise ValueError(f"Approval record not found: {approval_id}")
        
        with open(record_path, "r") as f:
            record = json.load(f)
        
        record["status"] = status
        record["approved_by"] = approved_by
        record["approval_timestamp"] = datetime.utcnow().isoformat()
        record["updated_at"] = datetime.utcnow().isoformat()
        if approval_comments:
            record["approval_comments"] = approval_comments
        
        # Save updated record
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"Approval record updated: {approval_id} -> {status}")
        return record
    
    def list_pending_approvals(self) -> list:
        """
        List all pending approval requests.
        
        Returns:
            List of pending approval records
        """
        pending_approvals = []
        
        for record_file in self.approval_dir.glob("*.json"):
            with open(record_file, "r") as f:
                record = json.load(f)
            
            if record.get("status") == "pending":
                pending_approvals.append(record)
        
        return pending_approvals
    
    def get_approval_history(self, model_name: str) -> list:
        """
        Get approval history for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of approval records for the model
        """
        history = []
        
        for record_file in self.approval_dir.glob("*.json"):
            with open(record_file, "r") as f:
                record = json.load(f)
            
            if record.get("model_name") == model_name:
                history.append(record)
        
        # Sort by created_at timestamp
        history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return history
