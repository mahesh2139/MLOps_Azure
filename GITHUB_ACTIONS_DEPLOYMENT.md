# GitHub Actions Deployment Guide

Deploy your MLOps pipeline to GitHub Actions for automatic CI/CD.

## Prerequisites

- GitHub repository (public or private)
- GitHub CLI installed (`gh` command)
- Azure subscription with configured service principal
- Push access to repository

## Quick Setup (5 minutes)

### Step 1: Create Service Principal

```bash
# Create service principal
az ad sp create-for-rbac \
  --name mlops-github-action \
  --role Contributor \
  --scopes /subscriptions/{subscription-id}

# Output example:
# {
#   "appId": "...",
#   "displayName": "mlops-github-action",
#   "password": "...",
#   "tenant": "..."
# }
```

Save the output - you'll need it in Step 2.

### Step 2: Configure GitHub Secrets

#### Option A: Automated (Recommended)

```bash
# Make script executable
chmod +x scripts/setup_github_actions.sh

# Run setup
./scripts/setup_github_actions.sh

# Follow prompts to enter Azure credentials
```

#### Option B: Manual

Go to: `GitHub → Settings → Secrets and variables → Actions`

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `AZURE_CREDENTIALS` | Full JSON from service principal |
| `AZURE_SUBSCRIPTION_ID` | Your subscription ID |
| `AZURE_RESOURCE_GROUP` | Your resource group name |
| `AZURE_WORKSPACE_NAME` | Your ML workspace name |

**AZURE_CREDENTIALS** format:
```json
{
  "clientId": "your-app-id",
  "clientSecret": "your-password",
  "subscriptionId": "your-subscription",
  "tenantId": "your-tenant"
}
```

### Step 3: Push to Repository

```bash
# Commit all files
git add .
git commit -m "Add production MLOps pipeline with GitHub Actions"
git push origin main

# This triggers: model_training.yml
```

### Step 4: Monitor Workflow

```bash
# Check workflow status
gh run list

# View specific run
gh run view <run-id>

# View workflow output
gh run view <run-id> --log
```

Or visit: `GitHub → Actions → Workflows`

## Workflow Schedules

### model_training.yml
**Trigger**: Push to main/develop OR manual dispatch
**When**: Immediately on push
**Duration**: ~20-30 minutes
**Output**: Model artifacts, evaluation report

### batch_inference.yml
**Trigger**: Scheduled daily
**When**: 2:00 AM UTC (14:00 IST)
**Duration**: ~10-15 minutes
**Output**: batch_predictions.csv, batch_summary.json

### model_monitoring.yml
**Trigger**: Scheduled every 6 hours
**When**: 2 AM, 8 AM, 2 PM, 8 PM UTC
**Duration**: ~5-10 minutes
**Output**: data_drift_report.html, monitoring_summary.json

## Customizing Schedules

### Change Training Trigger

Edit `.github/workflows/model_training.yml`:

```yaml
on:
  push:
    branches:
      - main
      - develop
    paths:
      - 'src/pipelines/train.py'    # Only run if this changes
      - 'config/**'
```

### Change Batch Inference Schedule

Edit `.github/workflows/batch_inference.yml`:

```yaml
schedule:
  - cron: '0 2 * * *'    # 2 AM UTC daily
  # Common examples:
  # '0 2 * * *'        = 2 AM UTC daily
  # '0 14 * * *'       = 2 PM UTC daily
  # '0 */6 * * *'      = Every 6 hours
  # '30 1 * * 1'       = Monday 1:30 AM UTC
```

### Change Monitoring Schedule

Edit `.github/workflows/model_monitoring.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'   # Every 6 hours
  # '0 * * * *'          = Every hour
  # '0 */3 * * *'        = Every 3 hours
```

## Manual Workflow Triggers

### Trigger Training Manually

```bash
gh workflow run model_training.yml
```

Or via GitHub UI:
- Go to `Actions` tab
- Select `Model Training Pipeline`
- Click `Run workflow`

### Trigger Batch Inference Manually

```bash
gh workflow run batch_inference.yml \
  --ref main \
  -F batch_input_path="UI/custom/path/file.csv"
```

### Trigger Monitoring Manually

```bash
gh workflow run model_monitoring.yml \
  --ref main
```

## Viewing Results

### Check Workflow Runs

```bash
# List all runs
gh run list

# List training runs
gh run list --workflow model_training.yml

# View specific run details
gh run view <run-id> --json status,conclusion,createdAt
```

### Download Artifacts

```bash
# List artifacts
gh run list

# Download from run
gh run download <run-id> --dir ./artifacts

# View contents
ls -la artifacts/
```

### View Logs

```bash
# Stream live logs
gh run watch <run-id> --exit-status

# Download logs
gh run download <run-id> --name model-training-artifacts
```

## Troubleshooting

### Issue: "Authentication failed"

**Cause**: Invalid Azure credentials
**Solution**: 
```bash
# Verify secrets are set
gh secret list

# Verify service principal still has access
az role assignment list --assignee <app-id>
```

### Issue: "Workflow not triggering on push"

**Cause**: Push path doesn't match `paths` filter
**Solution**: 
- Remove `paths` filter to trigger on all pushes, OR
- Edit `.github/workflows/model_training.yml` and remove/modify the `paths` section

### Issue: "Scheduled workflow not running"

**Cause**: GitHub Actions may be disabled or schedule not set correctly
**Solution**:
```bash
# Verify CRON syntax
# Use: https://crontab.guru/

# Check if Actions are enabled
gh api repos/{owner}/{repo} --jq .has_discussions
```

### Issue: "Azure resources not found"

**Cause**: Workspace or datastore not in Azure ML
**Solution**:
```bash
# Verify workspace exists
az ml workspace list

# Verify datastore exists
az ml datastore list --workspace-name YOUR_WORKSPACE
```

## Security Best Practices

✅ **Secrets Management**
- Use GitHub Secrets, never hardcode credentials
- Rotate secrets periodically
- Use Service Principal with minimal permissions

✅ **Access Control**
- Limit branch protection to main/develop
- Require approval for production deployments
- Enable branch protection rules

✅ **Audit Trail**
```bash
# View workflow run history
gh run list --limit 50

# Export run history
gh run list --json url,conclusion,createdAt --limit 100 > runs.json
```

✅ **Cost Optimization**
- Use scheduled workflows efficiently
- Cancel old/redundant runs
- Monitor compute usage

## Advanced Configuration

### Enable PR Reviews

Add to `.github/workflows/model_training.yml`:

```yaml
on:
  pull_request:
    branches:
      - main
```

Then add this step for PR comments:

```yaml
- name: Comment on PR
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v6
  with:
    script: |
      const fs = require('fs');
      const report = JSON.parse(fs.readFileSync('./outputs/evaluation_report.json', 'utf8'));
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `## 📊 Model Evaluation Results\n- Accuracy: ${report.metrics.accuracy}\n- Promotion Eligible: ${report.promotion_eligible}`
      });
```

### Add Slack Notifications

```yaml
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'Model training: ${{ job.status }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Matrix Testing (Multiple Configurations)

```yaml
jobs:
  train:
    strategy:
      matrix:
        model-params: [
          {n_estimators: 50},
          {n_estimators: 100},
          {n_estimators: 200}
        ]
    steps:
      - name: Train with params
        run: |
          python -m src.pipelines.train \
            --n-estimators ${{ matrix.model-params.n_estimators }}
```

## Monitoring & Alerts

### GitHub Action Notifications

Enable in `GitHub → Settings → Notifications`:
- Workflow run failures
- Deployment reviews
- Security alerts

### Azure Monitor Integration

```yaml
- name: Send metrics to Azure Monitor
  run: |
    python << 'EOF'
    import json
    from azure.monitor.query import MetricsQueryClient
    # Send custom metrics to Azure Monitor
    EOF
```

## Deployment Strategies

### Strategy 1: Staged Rollout

```yaml
on:
  push:
    branches:
      - develop    # Development pipeline
on:
  push:
    branches:
      - main       # Production pipeline
```

### Strategy 2: Approval Gate

Add reviewer approval in Azure DevOps or external tool:

```yaml
- name: Wait for approval
  uses: trstringer/manual-approval@v1
  with:
    secret: ${{ secrets.GITHUB_TOKEN }}
    approvers: mlops-team
```

## Performance Optimization

### Parallel Job Execution

```yaml
jobs:
  data-prep:
    runs-on: ubuntu-latest
    steps:
      - name: Prepare data
        run: python -m src.pipelines.data_prep
  
  model-train:
    needs: data-prep
    runs-on: ubuntu-latest
    steps:
      - name: Train
        run: python -m src.pipelines.train
```

### Caching Dependencies

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

## Next Steps

1. ✅ Set up GitHub secrets: `./scripts/setup_github_actions.sh`
2. ✅ Push to repository: `git push origin main`
3. ✅ Monitor workflows: `GitHub → Actions`
4. ✅ Customize schedules as needed
5. ✅ Set up notifications

## Support

- GitHub Actions Docs: https://docs.github.com/en/actions
- Azure ML Docs: https://docs.microsoft.com/azure/machine-learning
- Workflow Syntax: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions

---

**Status**: Ready for Deployment ✅
