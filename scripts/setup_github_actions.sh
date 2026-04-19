#!/bin/bash
# GitHub Actions Setup Script
# Configures GitHub repository with secrets and enables workflows

set -e

echo "================================================"
echo "GitHub Actions Setup"
echo "================================================"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI not found. Please install it:"
    echo "   https://cli.github.com/"
    exit 1
fi

# Get repository information
REPO=$(gh repo view --json nameWithOwner -q)
echo "📦 Repository: $REPO"

# Authenticate to GitHub if needed
echo "🔐 Authenticating to GitHub..."
gh auth status || gh auth login

# Collect Azure credentials
echo ""
echo "📝 Enter your Azure credentials:"
echo ""

read -p "Azure Subscription ID: " AZURE_SUBSCRIPTION_ID
read -p "Azure Resource Group: " AZURE_RESOURCE_GROUP
read -p "Azure Workspace Name: " AZURE_WORKSPACE_NAME
read -p "Azure Client ID (Service Principal): " AZURE_CLIENT_ID
read -p "Azure Client Secret: " -s AZURE_CLIENT_SECRET
echo ""
read -p "Azure Tenant ID: " AZURE_TENANT_ID

# Create AZURE_CREDENTIALS JSON
AZURE_CREDENTIALS=$(cat <<EOF
{
  "clientId": "$AZURE_CLIENT_ID",
  "clientSecret": "$AZURE_CLIENT_SECRET",
  "subscriptionId": "$AZURE_SUBSCRIPTION_ID",
  "tenantId": "$AZURE_TENANT_ID"
}
EOF
)

# Set GitHub secrets
echo ""
echo "🔒 Setting GitHub Actions secrets..."

gh secret set AZURE_CREDENTIALS --body "$AZURE_CREDENTIALS"
echo "✅ AZURE_CREDENTIALS set"

gh secret set AZURE_SUBSCRIPTION_ID --body "$AZURE_SUBSCRIPTION_ID"
echo "✅ AZURE_SUBSCRIPTION_ID set"

gh secret set AZURE_RESOURCE_GROUP --body "$AZURE_RESOURCE_GROUP"
echo "✅ AZURE_RESOURCE_GROUP set"

gh secret set AZURE_WORKSPACE_NAME --body "$AZURE_WORKSPACE_NAME"
echo "✅ AZURE_WORKSPACE_NAME set"

# Verify secrets are set
echo ""
echo "📋 Verifying secrets..."
gh secret list

echo ""
echo "================================================"
echo "✅ GitHub Actions setup completed!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Push your code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Add MLOps pipeline'"
echo "   git push origin main"
echo ""
echo "2. Check workflow status:"
echo "   gh run list"
echo ""
echo "3. View workflows:"
echo "   GitHub → Actions → Choose workflow"
echo ""
