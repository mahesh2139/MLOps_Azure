"""
Champion/Challenger model comparison and promotion pipeline with human approval workflow.
Designed for regulated industries compliance.
"""
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
import uuid
import joblib
from azureml.core import Workspace
from src.components.logger import setup_logger, log_section, log_step
from src.components.model_registry import ModelRegistry
from src.components.approval_workflow import ApprovalWorkflow
from src.components.email_notifier import EmailNotifier

logger = setup_logger(__name__, log_file="champion_challenger.log")


def run_champion_challenger(
    model_path: str,
    metrics_path: str,
    model_name: str = "CreditCard_Fraud_Detection",
    output_dir: str = "./outputs",
    approval_email: str = "mahesh2139@gmail.com",
    require_approval: bool = True
):
    """
    Compare challenger model with current champion with optional human approval workflow.
    
    Args:
        model_path: Path to challenger model
        metrics_path: Path to metrics JSON file
        model_name: Name for model in registry
        output_dir: Directory to save results
        approval_email: Email for approval request
        require_approval: Require human approval before promotion (default: True for regulated industries)
    """
    log_section(logger, "CHAMPION/CHALLENGER PIPELINE WITH APPROVAL WORKFLOW")
    
    approval_id = str(uuid.uuid4())
    email_notifier = EmailNotifier()
    approval_workflow = ApprovalWorkflow()
    
    try:
        # Load challenger metrics
        log_step(logger, "Loading challenger metrics")
        with open(metrics_path, 'r') as f:
            challenger_metrics = json.load(f)
        
        logger.info(f"Challenger metrics: {challenger_metrics}")
        
        # Connect to workspace
        log_step(logger, "Connecting to Azure ML workspace")
        workspace = Workspace.from_config()
        registry = ModelRegistry(workspace, model_name)
        
        # Get current champion
        log_step(logger, "Retrieving current champion model")
        champion_model = registry.get_champion_model()
        
        if champion_model is None:
            logger.warning("No champion model found. Promoting challenger as initial champion.")
            
            log_step(logger, "Registering model as champion")
            registered = registry.register_model(
                model_path=model_path,
                metrics=challenger_metrics,
                tags={'role': 'champion', 'status': 'production', 'approval_id': approval_id},
                description="Initial champion model for credit card fraud detection"
            )
            
            result = {
                "status": "promoted",
                "action": "initial_champion",
                "model_version": registered.version,
                "reason": "No previous champion found",
                "approval_required": False,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("✅ Initial champion registered without approval requirement")
        else:
            # Register challenger
            log_step(logger, "Registering challenger model")
            challenger_model = registry.register_model(
                model_path=model_path,
                metrics=challenger_metrics,
                tags={'role': 'challenger', 'status': 'staging', 'approval_id': approval_id},
                description="Challenger model for credit card fraud detection"
            )
            
            # Compare models
            log_step(logger, "Comparing challenger vs champion")
            comparison = registry.compare_models(
                challenger_model,
                champion_model,
                metrics=['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            )
            
            logger.info(f"\nComparison Results:")
            logger.info(f"  Challenger Version: {challenger_model.version}")
            logger.info(f"  Champion Version: {champion_model.version}")
            logger.info(f"  Comparison: {json.dumps(comparison, indent=2)}")
            
            # Determine if challenger is better
            model1_wins = sum(1 for v in comparison.values() if v['better'] == 'MODEL1')
            model2_wins = sum(1 for v in comparison.values() if v['better'] == 'MODEL2')
            
            logger.info(f"  Score: Challenger {model1_wins} vs Champion {model2_wins}")
            
            promoted = model1_wins > model2_wins
            
            # Check if approval is required
            if require_approval and promoted:
                logger.info(f"\n🔍 Approval required for model promotion (regulated industry compliance)")
                
                # Create approval request
                log_step(logger, "Creating approval request for human review")
                approval_data = {
                    "approval_id": approval_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_name": model_name,
                    "challenger_version": challenger_model.version,
                    "champion_version": champion_model.version,
                    "challenger_metrics": challenger_metrics,
                    "champion_metrics": champion_model.properties.get('metrics', {}),
                    "comparison": comparison,
                    "auto_recommendation": "PROMOTE" if promoted else "REJECT",
                    "status": "pending_approval"
                }
                
                # Save approval request
                approval_path = Path(output_dir) / "approvals"
                approval_path.mkdir(parents=True, exist_ok=True)
                
                with open(approval_path / f"approval_{approval_id}.json", "w") as f:
                    json.dump(approval_data, f, indent=2)
                
                logger.info(f"Approval request created: {approval_id}")
                
                # Send email for approval
                log_step(logger, "Sending approval email")
                email_body = _generate_approval_email(
                    approval_data,
                    challenger_model.version,
                    champion_model.version
                )
                
                email_sent = email_notifier.send_approval_request(
                    to_email=approval_email,
                    subject=f"[URGENT] Model Promotion Approval Required - {model_name}",
                    body=email_body,
                    approval_id=approval_id,
                    comparison_data=comparison,
                    challenger_metrics=challenger_metrics,
                    champion_metrics=champion_model.properties.get('metrics', {})
                )
                
                if email_sent:
                    logger.info(f"✉️ Approval email sent to {approval_email}")
                else:
                    logger.error(f"❌ Failed to send approval email")
                    raise Exception("Email notification failed")
                
                # Store approval in workflow
                approval_workflow.create_approval_record(
                    approval_id=approval_id,
                    model_name=model_name,
                    challenger_version=challenger_model.version,
                    champion_version=champion_model.version,
                    status="pending",
                    comparison=comparison,
                    requester_email="mlops@company.com"
                )
                
                result = {
                    "status": "pending_approval",
                    "action": "awaiting_human_review",
                    "approval_id": approval_id,
                    "challenger_version": challenger_model.version,
                    "champion_version": champion_model.version,
                    "auto_recommendation": "PROMOTE" if promoted else "REJECT",
                    "comparison": comparison,
                    "approval_email": approval_email,
                    "approval_required": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            elif promoted and not require_approval:
                # Auto-promote without approval (non-regulated scenario)
                logger.info(f"\n🚀 Challenger outperforms champion! Promoting (approval not required)...")
                
                log_step(logger, "Archiving previous champion")
                registry.promote_model(champion_model, 'archived', 'archived')
                
                log_step(logger, "Promoting challenger to champion")
                registry.promote_model(challenger_model, 'champion', 'production')
                
                result = {
                    "status": "promoted",
                    "action": "auto_promotion",
                    "challenger_version": challenger_model.version,
                    "archived_version": champion_model.version,
                    "comparison": comparison,
                    "approval_required": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            else:
                logger.info(f"\n⚠️ Challenger did not outperform champion. No promotion.")
                
                result = {
                    "status": "rejected",
                    "action": "no_promotion",
                    "challenger_version": challenger_model.version,
                    "champion_version": champion_model.version,
                    "comparison": comparison,
                    "reason": "Challenger metrics not superior",
                    "approval_required": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Save result
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "champion_challenger_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"\n✅ Champion/Challenger comparison completed")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Champion/Challenger pipeline failed: {str(e)}", exc_info=True)
        raise


def process_approval_decision(approval_id: str, decision: str, approver_email: str, comments: str = ""):
    """
    Process approval decision and update model registry.
    Called when approver responds to approval email.
    
    Args:
        approval_id: Approval request ID
        decision: "approved" or "rejected"
        approver_email: Email of approver
        comments: Additional comments
    
    Returns:
        Decision result with status
    """
    log_section(logger, f"PROCESSING APPROVAL DECISION: {decision.upper()}")
    
    try:
        approval_workflow = ApprovalWorkflow()
        email_notifier = EmailNotifier()
        
        log_step(logger, f"Retrieving approval record: {approval_id}")
        approval_record = approval_workflow.get_approval_record(approval_id)
        
        if not approval_record:
            raise ValueError(f"Approval record not found: {approval_id}")
        
        workspace = Workspace.from_config()
        registry = ModelRegistry(workspace, approval_record['model_name'])
        
        if decision.lower() == "approved":
            logger.info("🚀 Approval granted. Promoting challenger to champion...")
            
            log_step(logger, "Retrieving models from registry")
            challenger_model = registry.get_model_by_version(
                approval_record['challenger_version']
            )
            champion_model = registry.get_model_by_version(
                approval_record['champion_version']
            )
            
            # Archive current champion
            log_step(logger, "Archiving previous champion")
            registry.promote_model(champion_model, 'archived', 'archived')
            
            # Promote challenger
            log_step(logger, "Promoting challenger to champion")
            registry.promote_model(
                challenger_model,
                'champion',
                'production',
                tags={'approval_id': approval_id, 'approved_by': approver_email}
            )
            
            approval_status = "approved"
            status_message = "✅ Model promoted to production"
            
        elif decision.lower() == "rejected":
            logger.info("⚠️ Approval rejected. Keeping current champion...")
            
            log_step(logger, "Archiving rejected challenger")
            challenger_model = registry.get_model_by_version(
                approval_record['challenger_version']
            )
            registry.promote_model(challenger_model, 'archived', 'archived')
            
            approval_status = "rejected"
            status_message = "⚠️ Model promotion rejected"
            
        else:
            raise ValueError(f"Invalid decision: {decision}")
        
        # Update approval record
        log_step(logger, "Updating approval record")
        approval_workflow.update_approval_record(
            approval_id=approval_id,
            status=approval_status,
            approved_by=approver_email,
            approval_timestamp=datetime.utcnow().isoformat(),
            comments=comments
        )
        
        # Send confirmation email
        log_step(logger, "Sending confirmation email")
        confirmation_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Model Promotion Decision: {decision.upper()}</h2>
            <p><strong>Approval ID:</strong> {approval_id}</p>
            <p><strong>Model:</strong> {approval_record['model_name']}</p>
            <p><strong>Challenger Version:</strong> {approval_record['challenger_version']}</p>
            <p><strong>Champion Version:</strong> {approval_record['champion_version']}</p>
            <p><strong>Decision:</strong> {status_message}</p>
            <p><strong>Approved By:</strong> {approver_email}</p>
            <p><strong>Timestamp:</strong> {datetime.utcnow().isoformat()}</p>
            {f"<p><strong>Comments:</strong> {comments}</p>" if comments else ""}
        </body>
        </html>
        """
        
        email_notifier.send_decision_notification(
            to_email=approval_record.get('requester_email', 'mlops@company.com'),
            subject=f"Model Promotion Decision - {approval_record['model_name']}",
            body=confirmation_body,
            decision=decision,
            approval_id=approval_id
        )
        
        logger.info(f"✉️ Confirmation email sent")
        logger.info(f"\n✅ Approval decision processed successfully")
        
        return {
            "approval_id": approval_id,
            "decision": decision,
            "status": approval_status,
            "approved_by": approver_email,
            "timestamp": datetime.utcnow().isoformat(),
            "message": status_message
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing approval: {str(e)}", exc_info=True)
        raise


def _generate_approval_email(approval_data, challenger_version, champion_version):
    """Generate formatted HTML email for approval."""
    
    comparison = approval_data.get('comparison', {})
    challenger_metrics = approval_data.get('challenger_metrics', {})
    champion_metrics = approval_data.get('champion_metrics', {})
    
    comparison_rows = "".join([
        f"""
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;">{metric}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{challenger_metrics.get(metric, 'N/A')}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{champion_metrics.get(metric, 'N/A')}</td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if comp.get('better') == 'MODEL1' else 'red'}; font-weight: bold;">
                {comp.get('better', 'N/A')}
            </td>
            <td style="padding: 8px; border: 1px solid #ddd;">{comp.get('difference', 'N/A')}</td>
        </tr>
        """
        for metric, comp in comparison.items()
    ])
    
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #0078d4; color: white; padding: 20px; border-radius: 5px; }}
            .section {{ margin-top: 20px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .metrics-table th {{ background-color: #f0f0f0; padding: 10px; text-align: left; border: 1px solid #ddd; }}
            .action-buttons {{ margin-top: 20px; }}
            .btn {{ display: inline-block; padding: 12px 20px; margin-right: 10px; text-decoration: none; border-radius: 5px; font-weight: bold; }}
            .btn-approve {{ background-color: #28a745; color: white; }}
            .btn-reject {{ background-color: #dc3545; color: white; }}
            .recommendation {{ padding: 10px; border-radius: 5px; margin-top: 10px; }}
            .recommendation.promote {{ background-color: #d4edda; border: 1px solid #28a745; color: #155724; }}
            .recommendation.reject {{ background-color: #fff3cd; border: 1px solid #ffc107; color: #856404; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Model Promotion Approval Required</h1>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p><strong>Model Name:</strong> {approval_data['model_name']}</p>
                <p><strong>Approval ID:</strong> <code>{approval_data['approval_id']}</code></p>
                <p><strong>Request Date:</strong> {approval_data['timestamp']}</p>
                <p><strong>Auto Recommendation:</strong>
                    <span class="recommendation {'promote' if approval_data['auto_recommendation'] == 'PROMOTE' else 'reject'}">
                        <strong>{approval_data['auto_recommendation']}</strong>
                    </span>
                </p>
            </div>
            
            <div class="section">
                <h2>Model Versions</h2>
                <p><strong>Challenger Version:</strong> <code>{challenger_version}</code> (New Model)</p>
                <p><strong>Current Champion Version:</strong> <code>{champion_version}</code> (Current Production)</p>
            </div>
            
            <div class="section">
                <h2>Metrics Comparison</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Challenger</th>
                        <th>Champion</th>
                        <th>Better</th>
                        <th>Difference</th>
                    </tr>
                    {comparison_rows}
                </table>
            </div>
            
            <div class="section action-buttons">
                <h2>Action Required</h2>
                <p>Please review the metrics comparison above and approve or reject the promotion:</p>
                <a href="https://your-approval-service.com/approve/{approval_data['approval_id']}" class="btn btn-approve">✅ APPROVE PROMOTION</a>
                <a href="https://your-approval-service.com/reject/{approval_data['approval_id']}" class="btn btn-reject">❌ REJECT PROMOTION</a>
            </div>
            
            <hr style="margin-top: 30px;">
            <p><small>This is an automated message from the MLOps pipeline. All model promotions require explicit human approval for regulated industries compliance (SOX, HIPAA, GDPR).</small></p>
        </div>
    </body>
    </html>
    """
    
    return html_body


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./outputs/model.pkl")
    parser.add_argument("--metrics-path", type=str, default="./outputs/metrics.json")
    parser.add_argument("--model-name", type=str, 
                       default="CreditCard_Fraud_Detection")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--approval-email", type=str, 
                       default="mahesh2139@gmail.com")
    parser.add_argument("--require-approval", type=bool, default=True,
                       help="Require human approval before promotion (default: True for regulated industries)")
    
    args = parser.parse_args()
    
    result = run_champion_challenger(
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        approval_email=args.approval_email,
        require_approval=args.require_approval
    )