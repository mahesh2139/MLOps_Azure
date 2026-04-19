"""
Email notification service for model approval requests and status updates.
Sends emails for approval notifications and decision updates.
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class EmailNotifier:
    """
    Handles email notifications for model promotion approvals.
    """
    
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address (from environment or config)
            sender_password: Sender email password (from environment or config)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or "mlops@company.com"
        self.sender_password = sender_password or "dummy_password"
        self.email_enabled = bool(sender_email)
        
        if not self.email_enabled:
            logger.warning(
                "Email notifications disabled - sender credentials not configured. "
                "Set MLOPS_EMAIL and MLOPS_EMAIL_PASSWORD environment variables."
            )
    
    def send_approval_request(
        self,
        to_email: str,
        subject: str,
        body: str,
        approval_id: str,
        comparison_data: Dict[str, Any],
        challenger_metrics: Dict[str, Any],
        champion_metrics: Dict[str, Any]
    ) -> bool:
        """
        Send approval request email with model comparison details.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body text
            approval_id: Approval request ID
            comparison_data: Model comparison metrics
            challenger_metrics: Challenger model metrics
            champion_metrics: Champion model metrics
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.email_enabled:
            logger.info(f"Email notification skipped (not configured): {to_email}")
            self._save_approval_notification_file(
                to_email, subject, body, approval_id, "skipped"
            )
            return True
        
        try:
            # Create HTML email with comparison table
            html_body = self._create_approval_email_html(
                body, comparison_data, challenger_metrics, champion_metrics, approval_id
            )
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = to_email
            
            # Attach plain text and HTML versions
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Approval request email sent to: {to_email}")
            self._save_approval_notification_file(
                to_email, subject, body, approval_id, "sent"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send approval email: {str(e)}")
            self._save_approval_notification_file(
                to_email, subject, body, approval_id, "failed"
            )
            return False
    
    def send_approval_decision(
        self,
        to_email: str,
        decision: str,
        approval_id: str,
        model_name: str,
        comments: str = ""
    ) -> bool:
        """
        Send approval decision notification.
        
        Args:
            to_email: Recipient email address
            decision: "approved" or "rejected"
            approval_id: Approval request ID
            model_name: Name of the model
            comments: Decision comments
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.email_enabled:
            logger.info(f"Email notification skipped (not configured): {to_email}")
            return True
        
        try:
            subject = f"[MLOps] Model Promotion {decision.upper()} - {model_name}"
            
            body = f"""
            Approval Decision: {decision.upper()}
            
            Model: {model_name}
            Approval ID: {approval_id}
            
            {f"Comments: {comments}" if comments else ""}
            
            Please log in to the MLOps dashboard for more details.
            """
            
            html_body = f"""
            <html>
                <body>
                    <h2>Model Promotion {decision.upper()}</h2>
                    <p><strong>Model:</strong> {model_name}</p>
                    <p><strong>Approval ID:</strong> {approval_id}</p>
                    <p><strong>Decision:</strong> <span style="color: {'green' if decision.lower() == 'approved' else 'red'}; font-weight: bold;">{decision.upper()}</span></p>
                    {f"<p><strong>Comments:</strong> {comments}</p>" if comments else ""}
                    <p>Please log in to the MLOps dashboard for more details.</p>
                </body>
            </html>
            """
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = to_email
            
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Decision notification email sent to: {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send decision email: {str(e)}")
            return False
    
    def _create_approval_email_html(
        self,
        body: str,
        comparison_data: Dict[str, Any],
        challenger_metrics: Dict[str, Any],
        champion_metrics: Dict[str, Any],
        approval_id: str
    ) -> str:
        """
        Create HTML version of approval email with metrics table.
        
        Args:
            body: Plain text body
            comparison_data: Model comparison data
            challenger_metrics: Challenger metrics
            champion_metrics: Champion metrics
            approval_id: Approval ID
            
        Returns:
            HTML formatted email body
        """
        # Build comparison table
        comparison_rows = ""
        if comparison_data:
            for metric, comparison in comparison_data.items():
                better = comparison.get("better", "N/A")
                challenger_val = comparison.get("model1", "N/A")
                champion_val = comparison.get("model2", "N/A")
                
                better_style = "color: green; font-weight: bold;" if better == "MODEL1" else ""
                
                comparison_rows += f"""
                <tr>
                    <td>{metric}</td>
                    <td style="{better_style}">{challenger_val}</td>
                    <td>{champion_val}</td>
                </tr>
                """
        
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2>Model Promotion Approval Required</h2>
                <p>{body}</p>
                
                <h3>Comparison Summary</h3>
                <table border="1" cellpadding="10" style="border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f2f2f2;">
                            <th>Metric</th>
                            <th>Challenger</th>
                            <th>Champion</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison_rows}
                    </tbody>
                </table>
                
                <h3>Approval ID</h3>
                <p><code>{approval_id}</code></p>
                
                <p style="margin-top: 30px; color: #666; font-size: 12px;">
                    This is an automated notification from the MLOps pipeline.
                    Do not reply to this email.
                </p>
            </body>
        </html>
        """
        return html
    
    def _save_approval_notification_file(
        self,
        to_email: str,
        subject: str,
        body: str,
        approval_id: str,
        status: str
    ) -> None:
        """
        Save approval notification to file for audit trail (fallback if email fails).
        
        Args:
            to_email: Recipient email
            subject: Email subject
            body: Email body
            approval_id: Approval ID
            status: Notification status (sent, failed, skipped)
        """
        try:
            notification_dir = Path("./outputs/notifications")
            notification_dir.mkdir(parents=True, exist_ok=True)
            
            notification = {
                "approval_id": approval_id,
                "to_email": to_email,
                "subject": subject,
                "status": status,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat()
            }
            
            notification_path = notification_dir / f"{approval_id}_notification.json"
            with open(notification_path, "w") as f:
                json.dump(notification, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save notification file: {str(e)}")
