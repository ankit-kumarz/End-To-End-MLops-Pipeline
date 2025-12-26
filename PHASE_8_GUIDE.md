# 🚀 PHASE 8: CI/CD Automation with GitHub Actions - Complete Guide

## 📖 Table of Contents
1. [What is CI/CD for MLOps?](#what-is-cicd-for-mlops)
2. [Why Automation Matters](#why-automation-matters)
3. [GitHub Actions Overview](#github-actions-overview)
4. [Workflows Created](#workflows-created)
5. [Testing Infrastructure](#testing-infrastructure)
6. [Interview Focus Points](#interview-focus-points)

---

## What is CI/CD for MLOps?

### CI (Continuous Integration)
**Definition:** Automatically test every code change before merging

**For ML Systems:**
```
Developer pushes code →
  ├─ Run unit tests (code correctness)
  ├─ Validate data schemas
  ├─ Check model performance
  ├─ Lint code for quality
  └─ Report results to PR
```

### CD (Continuous Delivery/Deployment)
**Definition:** Automatically deploy validated changes

**For ML Systems:**
```
Tests pass →
  ├─ Trigger model retraining
  ├─ Deploy to staging environment
  ├─ Run integration tests
  ├─ Deploy to production (with approval)
  └─ Monitor deployment
```

---

## Why Automation Matters

### Without CI/CD (Manual MLOps)

```
❌ Problems:
- Manual testing → easy to skip
- No standard process → inconsistent
- Deploy takes 2 weeks → slow iteration
- Bugs found in production → costly
- Can't reproduce issues → debugging nightmare
- Fear of changes → stagnation
```

### With CI/CD (Automated MLOps)

```
✅ Benefits:
- Automatic testing → catch bugs early
- Standard pipeline → consistent quality
- Deploy in minutes → fast iteration
- Bugs caught in CI → prevent production issues
- Full audit trail → easy debugging
- Confidence to experiment → innovation
```

---

## GitHub Actions Overview

### What is GitHub Actions?

**Definition:** GitHub's CI/CD platform that runs workflows automatically

### Key Concepts:

1. **Workflows** = YAML files defining automation
   - Location: `.github/workflows/*.yml`
   - Trigger: on push, PR, schedule, manual

2. **Jobs** = Set of steps that run together
   - Run on: ubuntu-latest, windows-latest, macos-latest
   - Can run in parallel or sequentially

3. **Steps** = Individual commands/actions
   - Checkout code
   - Set up Python
   - Run tests
   - Deploy application

4. **Triggers** = Events that start workflows
   - `on: push` = Every code push
   - `on: pull_request` = When PR created
   - `on: schedule` = Cron schedule
   - `on: workflow_dispatch` = Manual trigger

---

## Workflows Created

### 1. CI Tests (`.github/workflows/ci-tests.yml`)

**Purpose:** Run tests on every push/PR

**Triggers:**
- Every push to main/master/develop
- Every pull request

**What it does:**
```yaml
steps:
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies from requirements.txt
  4. Run pytest (unit tests)
  5. Run flake8 (code linting)
  6. Validate config files
  7. Check imports
  8. Upload coverage report
```

**Why this matters:**
> "Catches bugs before they reach production. In interviews, emphasize that you validate code quality automatically on every change."

---

### 2. Data Validation (`.github/workflows/data-validation.yml`)

**Purpose:** Validate data quality when data changes

**Triggers:**
- Changes to `data/` directory
- Changes to validation scripts
- Manual trigger

**What it does:**
```yaml
steps:
  1. Checkout code
  2. Set up Python + DVC
  3. Generate/pull data
  4. Run data validation (schema, nulls, ranges, drift)
  5. Create validation report
  6. Upload report as artifact
  7. Fail if validation fails
```

**Why this matters:**
> "Data quality is the #1 cause of ML failures. This workflow prevents bad data from poisoning your models."

---

### 3. Model Training (`.github/workflows/model-training.yml`)

**Purpose:** Automatically retrain model when data is ready

**Triggers:**
- Manual dispatch (with algorithm choice)
- After successful data validation
- Weekly schedule (Sunday 2 AM)

**What it does:**
```yaml
steps:
  1. Checkout code
  2. Set up Python + DVC
  3. Pull/generate data
  4. Run DVC pipeline (dvc repro)
  5. Extract model metrics
  6. Check performance threshold (>= 90%)
  7. Save model artifacts
  8. Create training report
  9. Comment metrics on PR
```

**Why this matters:**
> "Ensures models are always up-to-date. The threshold check prevents deploying underperforming models."

---

### 4. Deploy API (`.github/workflows/deploy-api.yml`)

**Purpose:** Deploy API after successful training

**Triggers:**
- Manual dispatch
- After successful model training

**Stages:**
```
Staging Deployment:
  ├─ Download trained model
  ├─ Start API server
  ├─ Test endpoints (health, predict)
  ├─ Create deployment package
  └─ Upload artifacts

Production Deployment:
  ├─ Requires manual approval (environment protection)
  ├─ Download deployment package
  ├─ Deploy to cloud (Azure/AWS/GCP)
  ├─ Verify deployment
  └─ Monitor performance
```

**Why this matters:**
> "Staging catches deployment issues before production. Manual approval for production prevents accidental releases."

---

### 5. Scheduled Retraining (`.github/workflows/schedule-retrain.yml`)

**Purpose:** Automatically check for data drift and retrain

**Triggers:**
- Every Monday at 3 AM UTC
- Manual dispatch

**What it does:**
```yaml
steps:
  1. Check for data drift
  2. If drift detected → trigger retraining
  3. If no drift → skip retraining
  4. Notify team of decision
```

**Why this matters:**
> "Handles model decay. Real-world data changes over time, this ensures your model stays accurate."

---

## Testing Infrastructure

### Test Files Created:

#### 1. `tests/test_data_pipeline.py`
```python
Tests:
  ✓ test_clean_data_removes_nulls
  ✓ test_clean_data_removes_duplicates  
  ✓ test_engineer_features_creates_expected_features
  ✓ test_engineer_features_polynomial
  ✓ test_engineer_features_interactions
  ✓ test_engineer_features_handles_edge_cases

Purpose: Ensure data preprocessing is correct
```

#### 2. `tests/test_validation.py`
```python
Tests:
  ✓ test_validate_schema_success/failure
  ✓ test_validate_nulls_success/failure
  ✓ test_validate_ranges_success/failure
  ✓ test_validate_target_distribution_success/failure

Purpose: Ensure data validation catches issues
```

#### 3. `tests/test_api.py`
```python
Tests:
  ✓ test_root_endpoint
  ✓ test_health_endpoint
  ✓ test_model_info_endpoint
  ✓ test_predict_endpoint_valid_input
  ✓ test_predict_endpoint_invalid_input
  ✓ test_predict_endpoint_out_of_range
  ✓ test_engineer_features_output_shape
  ✓ test_engineer_features_bin_creation

Purpose: Ensure API works correctly
```

### Running Tests Locally:

```bash
# Install test dependencies
pip install pytest pytest-cov flake8 httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with detailed output
pytest tests/ -v -s
```

---

## Interview Focus Points

### Question 1: "How do you implement CI/CD for ML?"

**Answer:**
```
"I use GitHub Actions for CI/CD automation:

1. CI (Continuous Integration):
   - Run unit tests on every push
   - Validate data quality automatically
   - Check model performance thresholds
   - Lint code for quality standards

2. CD (Continuous Deployment):
   - Automatically retrain on data changes
   - Deploy to staging first for testing
   - Require manual approval for production
   - Monitor deployments with health checks

Key difference from traditional software: ML systems need data validation
and model performance checks, not just code tests."
```

### Question 2: "How do you handle model retraining?"

**Answer:**
```
"I use a combination of triggers:

1. Scheduled Retraining:
   - Weekly cron job checks for data drift
   - Retrains if drift exceeds threshold
   
2. Event-Based Retraining:
   - Trigger on new data uploads
   - Trigger on performance degradation
   
3. Manual Retraining:
   - workflow_dispatch for experiments
   - Choose algorithm as input parameter

Each retraining run:
- Validates data quality first
- Checks model performance threshold
- Only deploys if accuracy >= 90%
- Saves full artifact trail (MLflow + GitHub Actions artifacts)"
```

### Question 3: "How do you prevent bad models from reaching production?"

**Answer:**
```
"Multiple safety gates:

1. Data Validation:
   - Schema checks (correct types/columns)
   - Range validation (no outliers)
   - Distribution checks (no data drift)
   - Minimum samples per class

2. Model Validation:
   - Performance threshold (test_accuracy >= 0.90)
   - Compare against current production model
   - Cross-validation scores
   
3. Deployment Gates:
   - Staging deployment first
   - Smoke tests on staging
   - Manual approval for production
   - Canary deployments (1% traffic)
   
4. Rollback Plan:
   - Model artifacts versioned in MLflow
   - Can rollback to previous version instantly
   - Monitor error rates post-deployment"
```

### Question 4: "How do you test ML code differently from regular code?"

**Answer:**
```
"ML testing has unique challenges:

1. Traditional Unit Tests:
   - Test data cleaning functions
   - Test feature engineering logic
   - Test API endpoints
   
2. ML-Specific Tests:
   - Data validation tests (schemas, distributions)
   - Model performance tests (accuracy thresholds)
   - Training reproducibility tests
   - Prediction consistency tests
   
3. Integration Tests:
   - End-to-end pipeline tests
   - API + model integration
   - Data versioning (DVC) integration
   
4. Non-Determinism Handling:
   - Set random seeds for reproducibility
   - Test ranges instead of exact values
   - Use fixtures for consistent test data"
```

### Question 5: "How would you scale this to multiple models?"

**Answer:**
```
"Several strategies:

1. Matrix Strategy:
   - GitHub Actions supports matrix builds
   - Run same workflow with different algorithms:
     strategy:
       matrix:
         algorithm: [random_forest, xgboost, neural_net]
   
2. Workflow Templates:
   - Create reusable workflow files
   - Call with different parameters
   
3. Model Registry:
   - MLflow Model Registry tracks all models
   - Each workflow registers its model
   - Promote best performer to production
   
4. A/B Testing:
   - Deploy multiple models simultaneously
   - Route traffic based on experiment
   - Champion/challenger pattern
   
5. Monitoring:
   - Track performance per model
   - Detect drift per model
   - Retrain independently"
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│              GITHUB ACTIONS CI/CD PIPELINE               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Developer pushes code                                   │
│         │                                                │
│         ▼                                                │
│    [CI Tests]                                            │
│    ├─ Run pytest                                         │
│    ├─ Lint code                                          │
│    ├─ Validate configs                                   │
│    └─ Upload coverage                                    │
│         │                                                │
│         ├─ ✅ Tests pass → Continue                      │
│         └─ ❌ Tests fail → Block merge                   │
│                                                          │
│  Data changes detected                                   │
│         │                                                │
│         ▼                                                │
│    [Data Validation]                                     │
│    ├─ Check schema                                       │
│    ├─ Check for drift                                    │
│    ├─ Validate ranges                                    │
│    └─ Create report                                      │
│         │                                                │
│         ├─ ✅ Valid → Trigger training                   │
│         └─ ❌ Invalid → Alert team                       │
│                                                          │
│  Training triggered                                      │
│         │                                                │
│         ▼                                                │
│    [Model Training]                                      │
│    ├─ Run DVC pipeline                                   │
│    ├─ Track with MLflow                                  │
│    ├─ Check threshold                                    │
│    └─ Save artifacts                                     │
│         │                                                │
│         ├─ ✅ Accuracy >= 90% → Deploy                   │
│         └─ ❌ Accuracy < 90% → Alert                     │
│                                                          │
│  Deployment triggered                                    │
│         │                                                │
│         ▼                                                │
│    [Deploy API]                                          │
│    ├─ Deploy to staging                                  │
│    ├─ Run smoke tests                                    │
│    ├─ Manual approval                                    │
│    └─ Deploy to production                               │
│         │                                                │
│         └─ ✅ Monitor & alert                            │
│                                                          │
│  Weekly schedule (Monday 3 AM)                           │
│         │                                                │
│         ▼                                                │
│    [Scheduled Retrain]                                   │
│    ├─ Check data drift                                   │
│    ├─ Decide if retraining needed                        │
│    └─ Trigger training if needed                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Real-World Production Patterns

### 1. Environment Protection Rules

In GitHub Settings → Environments → Create "production":
```
Protection Rules:
  ✓ Required reviewers: 2 approvals
  ✓ Wait timer: 10 minutes (cooldown)
  ✓ Restrict to branch: main only
  ✓ Secrets: Production API keys
```

### 2. Slack/Teams Notifications

Add to workflows:
```yaml
- name: Notify team
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Model training complete! Accuracy: ${{ steps.metrics.outputs.test_accuracy }}"
      }
```

### 3. Model Performance Degradation Alerts

```yaml
- name: Compare with production model
  run: |
    PROD_ACCURACY=0.95  # Load from MLflow registry
    NEW_ACCURACY=${{ steps.metrics.outputs.test_accuracy }}
    
    if (( $(echo "$NEW_ACCURACY < $PROD_ACCURACY - 0.02" | bc -l) )); then
      echo "⚠️  New model is 2% worse than production!"
      exit 1
    fi
```

### 4. Canary Deployments

```yaml
- name: Canary deployment
  run: |
    # Deploy new model to 5% of traffic
    kubectl set image deployment/model-api \
      app=model-api:${{ github.sha }} --record
    
    # Monitor error rates for 10 minutes
    # If errors increase, rollback
```

---

## Next Steps

### Immediate Actions:
1. ✅ Push to GitHub to trigger workflows
2. ✅ Check Actions tab to see runs
3. ✅ Fix any failing tests
4. ✅ Set up environment protection

### Phase 9 Preview: Containerization & Deployment
- Docker containerization
- Kubernetes deployment
- Cloud platform integration (Azure/AWS/GCP)
- Infrastructure as Code (Terraform)

---

## 🎯 Key Takeaways

1. **CI/CD for ML ≠ Traditional CI/CD**
   - Must handle data + code + models
   - Data validation is critical
   - Model performance thresholds required

2. **GitHub Actions = Powerful Automation**
   - 5 workflows cover full ML lifecycle
   - Triggers: push, schedule, manual, workflow_run
   - Artifacts preserve training history

3. **Testing is Multi-Layered**
   - Unit tests (code correctness)
   - Data tests (quality validation)
   - Model tests (performance thresholds)
   - Integration tests (end-to-end)

4. **Safety Gates Prevent Disasters**
   - Staging before production
   - Manual approval for production
   - Performance thresholds
   - Rollback capabilities

5. **Automation Enables Innovation**
   - Deploy multiple times per day
   - Experiment with confidence
   - Full audit trail
   - Reproducible workflows

**You now have enterprise-grade CI/CD for MLOps!** 🚀

