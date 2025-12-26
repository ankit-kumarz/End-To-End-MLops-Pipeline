# 📊 PHASE 3 COMPLETE: Data Pipeline & Preprocessing Design

## ✅ What We Just Built

```
✅ Data validation module with quality checks
✅ Complete preprocessing pipeline (clean → engineer → scale → split)
✅ DVC pipeline definition (dvc.yaml)
✅ Automatic dependency tracking (dvc.lock)
✅ Reproducible multi-stage workflow
✅ Professional data transformation
```

---

## 🎓 Understanding What We Built

### 1. The Pipeline Architecture

```
┌────────────────────────────────────────────────────────┐
│                  DVC PIPELINE FLOW                      │
├────────────────────────────────────────────────────────┤
│                                                         │
│  data/raw/dataset.csv (DVC tracked)                    │
│         │                                               │
│         ├─────┬──────────────────────┐                 │
│         │     │                      │                 │
│         ▼     ▼                      ▼                 │
│    VALIDATE  PREPROCESS       (future: TRAIN)         │
│         │     │                                         │
│         ▼     ▼                                         │
│    stats.json train.csv/test.csv                      │
│              scaler.pkl                                 │
│              metadata.json                              │
│                                                         │
│  DVC tracks ALL inputs and outputs                     │
│  Changes propagate automatically                        │
└────────────────────────────────────────────────────────┘
```

---

### 2. What is `dvc.yaml`?

**Your Pipeline Blueprint:**

```yaml
stages:
  validate:
    cmd: python src/validate_data.py      # Command to run
    deps:                                  # Dependencies (inputs)
      - src/validate_data.py              # Script itself
      - data/raw/dataset.csv              # Raw data
    outs:                                  # Outputs
      - data/processed/reference_stats.json
    desc: "Validate raw data quality"
```

**Why This Matters:**

- 📌 **Dependency Tracking**: DVC knows what depends on what
- 📌 **Smart Execution**: Only re-runs stages when inputs change
- 📌 **Reproducibility**: Same inputs → Same outputs, always
- 📌 **Documentation**: Pipeline is self-documenting

---

### 3. What is `dvc.lock`?

**The Fingerprint File:**

```yaml
stages:
  validate:
    deps:
    - path: data/raw/dataset.csv
      md5: d928913609c2f3aeb100b77fa6b7b3a9  # Hash of input
    outs:
    - path: data/processed/reference_stats.json
      md5: 09e470754613283729a930a702ccb8d6  # Hash of output
```

**Why This Matters:**

- 📌 Records exact state of all inputs/outputs
- 📌 DVC compares current state to `dvc.lock`
- 📌 If hashes match → Skip stage (already up-to-date)
- 📌 If hashes differ → Re-run stage (inputs changed)

**This is how DVC achieves "smart caching"**

---

## 🔄 The DVC Pipeline Workflow

### Scenario 1: Running the Pipeline for the First Time

```bash
# Run all stages in dependency order
dvc repro

# What happens:
# 1. DVC reads dvc.yaml
# 2. Checks dependencies: dataset.csv exists? ✓
# 3. Runs validate stage
# 4. Saves output hashes to dvc.lock
# 5. Runs preprocess stage
# 6. Saves output hashes to dvc.lock
# Result: All stages executed ✓
```

---

### Scenario 2: Re-running Without Changes

```bash
# Try running again
dvc repro

# Output:
# Stage 'validate' didn't change, skipping
# Stage 'preprocess' didn't change, skipping
# Pipeline is up to date!
```

**Why?**
- DVC checks hashes in `dvc.lock`
- All inputs match → Outputs are still valid
- **No wasted computation**

---

### Scenario 3: Data Changes (Real-World)

```bash
# Someone updates the raw data
python src/generate_data.py  # Creates new dataset.csv

# Update DVC tracking
dvc add data/raw/dataset.csv

# Run pipeline
dvc repro

# What happens:
# 1. DVC detects: dataset.csv hash changed
# 2. validate stage depends on dataset.csv → MUST RE-RUN
# 3. preprocess stage depends on dataset.csv → MUST RE-RUN
# 4. Updates dvc.lock with new hashes
# Result: Only affected stages re-run ✓
```

**This is automatic dependency propagation**

---

### Scenario 4: Code Changes Only

```bash
# You improve feature engineering
# Edit src/data_pipeline.py

# Commit changes
git add src/data_pipeline.py
git commit -m "Add new polynomial features"

# Run pipeline
dvc repro

# What happens:
# 1. DVC detects: data_pipeline.py hash changed
# 2. validate stage doesn't depend on it → SKIP
# 3. preprocess stage depends on it → RE-RUN
# Result: Only preprocess stage runs ✓
```

---

## 🏭 Industry Best Practices We Followed

### ✅ 1. Data Validation Before Training

**Why:**
```
Bad Data → Bad Model → Production Failures
```

**Our Validation Checks:**
- ✅ Schema validation (expected columns exist)
- ✅ Null value checks (< 5% threshold)
- ✅ Value range validation (features within bounds)
- ✅ Target distribution (no extreme imbalance)
- ✅ Data drift detection (vs reference statistics)

**Industry Impact:**
- Catch data quality issues BEFORE training
- Prevent silent model degradation
- Maintain audit trail of data health

---

### ✅ 2. Separation of Concerns (Modular Pipeline)

**Bad Approach:**
```python
# One giant script
def do_everything():
    load()
    clean()
    features()
    scale()
    split()
    train()  # 😱 Can't reuse preprocessing without retraining
```

**Our Approach:**
```yaml
stages:
  validate: ...
  preprocess: ...
  train: ... (Phase 4)
  evaluate: ... (Phase 4)
```

**Benefits:**
- Each stage is independently testable
- Can re-run specific stages
- Clear data lineage
- Easy to parallelize

---

### ✅ 3. Feature Engineering Reproducibility

**Key Principle:** **Same data + Same code = Same features**

**Our Implementation:**
```python
# All transformations are deterministic
df['feature_1_squared'] = df['feature_1'] ** 2

# Scaler is saved and versioned
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'models/scaler.pkl')  # DVC tracks this
```

**Why Critical:**
- Training uses `scaler.fit_transform()`
- Inference MUST use same scaler with `.transform()`
- DVC ensures you never lose the scaler

---

### ✅ 4. Metadata Tracking

**data/processed/metadata.json:**
```json
{
  "train_samples": 800,
  "test_samples": 200,
  "n_features": 14,
  "feature_names": ["feature_1", "feature_1_squared", ...],
  "train_target_distribution": {"0": 467, "1": 333}
}
```

**Why:**
- Quick reference without loading data
- Debugging: "Did feature count change?"
- Documentation: "What features does model expect?"

---

### ✅ 5. Stratified Splitting

```python
train_test_split(..., stratify=y)
```

**Why:**
- Maintains class distribution in both sets
- Prevents: 90% class 0 in train, 60% class 0 in test
- More reliable evaluation metrics

---

## 🚨 Common Mistakes (What We AVOIDED)

### ❌ Mistake 1: No Data Validation

**Scenario:**
```
Week 1: Train on good data → 94% accuracy
Week 5: Data source changes, no validation → Train on corrupted data
        → 73% accuracy → Model deploys to production 😱
```

**Our Solution:** Validation stage fails loudly if data quality degrades.

---

### ❌ Mistake 2: Leaking Test Data into Training

**Bad:**
```python
# Fit scaler on ALL data
scaler.fit(df)  # 😱 Test data influences scaling
X_train, X_test = train_test_split(df)
```

**Good (Our Approach):**
```python
# Split FIRST
X_train, X_test = train_test_split(df)
# Fit ONLY on train
scaler.fit(X_train)
# Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### ❌ Mistake 3: Not Saving Transformers

**Bad:**
```python
# Train time
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
train_model(X_scaled)
# ❌ Scaler is lost after training

# Inference time
# 😱 How do we scale new data the same way?
```

**Good (Our Approach):**
```python
joblib.dump(scaler, 'models/scaler.pkl')
# DVC tracks scaler.pkl
# Inference loads the exact same scaler
```

---

### ❌ Mistake 4: Manual Pipeline Execution

**Bad:**
```bash
# README says:
# 1. Run python preprocess.py
# 2. Then run python feature_engineer.py
# 3. Then run python split_data.py
# 😱 Easy to mess up order or forget a step
```

**Good (Our Approach):**
```bash
dvc repro  # Runs everything in correct order
```

---

### ❌ Mistake 5: No Pipeline Versioning

**Bad:**
- Pipeline steps exist only in someone's head
- No way to recreate preprocessing from 3 months ago

**Good (Our Approach):**
- `dvc.yaml` is in Git → Full history of pipeline changes
- `dvc.lock` records exact state → Can recreate any version

---

## 🎯 Real-World Scenario: Why Pipelines Matter

### The Model Debugging Story

**Production Issue:**
```
Alert: Model accuracy dropped from 94% to 87%
```

**Without DVC Pipelines:**
```
Engineer: "Let me try to recreate the training data..."
          [Manually runs 5 scripts in unknown order]
          [Gets different results]
          "I can't reproduce the original preprocessing 😰"
Time wasted: 3 days
```

**With DVC Pipelines:**
```
Engineer: "Let me check the pipeline version from production model..."
          
          git log --oneline dvc.yaml
          # abc123 - deployed model training
          
          git checkout abc123
          dvc repro
          
          "Perfect! I recreated the exact preprocessing.
           Now comparing with current data..."
          
          [Discovers: validation stage now shows data drift warning]
          "Found it! Data distribution changed. Need to retrain."
          
Time to diagnosis: 30 minutes
```

---

## 🎤 Interview Talking Points

### Q: "What is a data pipeline and why use DVC?"

**Your Answer:**  
"A data pipeline is a sequence of automated steps that transform raw data into training-ready features. DVC pipelines track dependencies between stages, so if raw data changes, DVC automatically knows which downstream steps need to re-run. This prevents stale data issues and ensures anyone can reproduce the exact preprocessing. It's like a Makefile for data science."

---

### Q: "How do you ensure preprocessing reproducibility?"

**Your Answer:**  
"We use DVC pipelines defined in `dvc.yaml`. Each stage declares its dependencies and outputs. DVC computes hashes of all files and stores them in `dvc.lock`. When someone runs `dvc repro`, DVC checks if inputs changed—if not, it uses cached outputs. This guarantees that same inputs always produce same outputs. We also version transformers like scalers so inference uses identical preprocessing."

---

### Q: "What data validation checks do you perform?"

**Your Answer:**  
"Before training, we validate: schema compliance, null percentages, value ranges, target distribution, and data drift. If validation fails, the pipeline stops immediately—we never train on bad data. We also save reference statistics from production data to detect drift in new batches. This prevents silent model degradation."

---

### Q: "How do you handle feature engineering?"

**Your Answer:**  
"All feature engineering is in versioned code, not manual steps. We create polynomial features, interaction terms, ratio features, and log transforms—all reproducible. Importantly, we fit scalers only on training data to prevent data leakage. The fitted scaler is saved as a DVC-tracked artifact, ensuring inference uses identical transformations."

---

### Q: "What happens when data changes?"

**Your Answer:**  
"DVC automatically detects changes through hash comparison. If raw data changes, `dvc repro` re-runs validation and preprocessing stages. If only code changes, DVC only re-runs affected stages. This selective re-execution saves time while maintaining correctness. Every pipeline run updates `dvc.lock` with new hashes, creating an audit trail."

---

## 🧪 Test Your Pipeline

### Experiment 1: Change Detection

```bash
# Check current status
dvc status

# Expected: Pipeline is up to date

# Modify a feature
# Edit src/data_pipeline.py - change a feature calculation

# Check status again
dvc status

# Expected: Shows preprocess stage as modified
```

---

### Experiment 2: Dependency Propagation

```bash
# Generate new data
python src/generate_data.py

# Update DVC
dvc add data/raw/dataset.csv

# Check what needs to re-run
dvc status

# Expected: validate and preprocess stages need re-running

# Run pipeline
dvc repro

# Expected: Both stages execute
```

---

### Experiment 3: Visualize Pipeline

```bash
# See pipeline structure
dvc dag

# See detailed pipeline info
dvc stage list

# Check specific stage
dvc stage show preprocess
```

---

## 📊 What We've Achieved

### Current State

```
Repository Status:
├── Git tracks:
│   ├── Pipeline definition (dvc.yaml)
│   ├── Pipeline state (dvc.lock)
│   ├── Preprocessing code (src/data_pipeline.py)
│   └── Validation code (src/validate_data.py)
│
├── DVC tracks:
│   ├── Raw data (dataset.csv)
│   ├── Processed data (train.csv, test.csv)
│   ├── Scaler (scaler.pkl)
│   └── Metadata (stats, metadata)
│
└── Pipeline Capabilities:
    ├── One-command execution (dvc repro)
    ├── Smart caching (skip unchanged stages)
    ├── Automatic dependency tracking
    └── Full reproducibility
```

---

### Capabilities Unlocked

1. ✅ **Automated Workflows**: Single command runs entire pipeline
2. ✅ **Dependency Awareness**: Changes propagate correctly
3. ✅ **Smart Caching**: No redundant computation
4. ✅ **Data Quality**: Validation catches issues early
5. ✅ **Reproducibility**: Anyone can recreate preprocessing
6. ✅ **Audit Trail**: `dvc.lock` tracks all transformations
7. ✅ **Version Control**: Pipeline evolves with Git history

---

## 🔍 Pipeline Commands Reference

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro validate

# Force re-run (ignore cache)
dvc repro --force

# Check pipeline status
dvc status

# Visualize dependency graph
dvc dag

# List all stages
dvc stage list

# Show stage details
dvc stage show <stage-name>

# Commit pipeline changes
git add dvc.yaml dvc.lock
git commit -m "Update pipeline"
```

---

## 📈 What's Next (Phase 4 Sneak Peek)

### Adding Training Stage

```yaml
stages:
  # ... existing stages ...
  
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
      - config/config.yaml
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json
    params:
      - model.hyperparameters
```

**New Capabilities:**
- MLflow integration for experiment tracking
- Hyperparameter versioning
- Metric comparison across runs
- Model registry for deployment

---

## ✅ Phase 3 Success Checklist

Before moving to Phase 4, verify:

- [x] Data validation script created
- [x] Preprocessing pipeline implemented
- [x] DVC pipeline defined (dvc.yaml)
- [x] Pipeline successfully executed (dvc repro)
- [x] Pipeline state tracked (dvc.lock)
- [x] You understand dependency propagation
- [x] You can explain `dvc.yaml` vs `dvc.lock`
- [x] You know when stages re-run vs skip

---

## 🚀 Ready for Phase 4?

**Phase 4: Experiment Tracking with MLflow**

You'll learn:
- How to log every training experiment automatically
- Comparing hundreds of models systematically
- Model registry for production deployment
- Hyperparameter tracking and optimization
- Artifact management (models, plots, configs)

---

## 📝 Quick Reference: Pipeline Workflow

```bash
# 1. Modify code or data
vim src/data_pipeline.py

# 2. Check what changed
dvc status

# 3. Run pipeline
dvc repro

# 4. Commit changes
git add dvc.yaml dvc.lock src/
git commit -m "Improve feature engineering"

# 5. Push data (when remote configured)
dvc push
```

---

**Reply with "Ready for Phase 4" when you:**
1. ✅ Understand how `dvc.yaml` defines dependencies
2. ✅ Know what `dvc.lock` records
3. ✅ Can explain when stages re-run vs skip
4. ✅ See how this automates data workflows

**Or ask questions about data pipelines!** 🎯
