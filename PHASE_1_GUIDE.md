# 📚 PHASE 1 GUIDE: Understanding Your Project Structure

## ✅ What We Just Built

```
End-To-End-MLOPs/
│
├── .github/workflows/        # 🤖 CI/CD automation lives here
├── src/                      # 💻 Production code only
│   ├── __init__.py          
│   ├── config.py            # Configuration loader
│   └── utils.py             # Shared utilities
├── data/                     # 📊 Data storage (DVC manages this)
│   ├── raw/                 
│   └── processed/           
├── models/                   # 🎯 Model artifacts (MLflow manages this)
├── notebooks/                # 📓 Exploration only (NOT production)
├── tests/                    # ✅ Testing suite
├── config/                   # ⚙️ Configuration files
│   └── config.yaml          
├── .gitignore               # 🚫 What Git ignores
├── .dvcignore               # 🚫 What DVC ignores
├── requirements.txt         # 📦 Dependencies
├── LICENSE                  # 📄 MIT License
└── README.md                # 📖 Documentation
```

## 🎓 Key Concepts Explained

### 1. Why Separate `src/` from `notebooks/`?

**Notebooks:** Quick experiments, visualizations, EDA
- ❌ Don't use notebooks in production
- ❌ Hard to test
- ❌ Hard to version control (contain outputs)
- ✅ Great for exploration

**src/:** Production-ready Python modules
- ✅ Testable functions
- ✅ Importable by other scripts
- ✅ CI/CD can execute them
- ✅ Version control friendly

**Industry Rule:** "If it goes to production, it's a `.py` file in `src/`, not a notebook."

---

### 2. Why `.gitkeep` Files?

Git doesn't track empty directories. But we need these folders to exist.

**Solution:** Put a `.gitkeep` file in each empty directory.
- Git tracks the file → directory exists
- Actual data files are gitignored
- Structure is preserved for collaborators

---

### 3. Why Separate `config/config.yaml`?

**Bad Practice:**
```python
# Hardcoded in code
n_estimators = 100
learning_rate = 0.01
```

**Good Practice:**
```yaml
# config/config.yaml
model:
  n_estimators: 100
  learning_rate: 0.01
```

**Why?**
- ✅ Change hyperparameters without editing code
- ✅ Different configs for dev/staging/production
- ✅ Easy to track what changed between experiments

---

### 4. What's in `.gitignore`?

**Three Categories:**

**1. Large/Binary Files (tracked by DVC instead):**
```
data/raw/*
models/*.pkl
```

**2. Generated Files (recreatable):**
```
__pycache__/
mlruns/
```

**3. Secrets (NEVER commit):**
```
.env
*.env
```

**Interview Question:** "Why not commit data to Git?"  
**Answer:** Git tracks every version. A 1GB dataset × 10 versions = 10GB repo. DVC stores data efficiently with deduplication.

---

### 5. Why `requirements.txt` Now?

**Reproducibility Principle:**
```
Your Code + Different Libraries = Different Results
```

**Example:**
- You train with `scikit-learn==1.3.0` → 95% accuracy
- Colleague uses `scikit-learn==1.4.0` → 93% accuracy (API changes)

**Solution:** Pin exact versions in `requirements.txt`

---

## 🏭 Industry Best Practices We Followed

### ✅ 1. Separation of Concerns
Each directory has ONE responsibility:
- `src/` = Logic
- `data/` = Storage
- `tests/` = Validation
- `config/` = Settings

### ✅ 2. Automation-Friendly Structure
CI/CD needs to know:
- Where's the training script? → `src/train.py`
- Where are tests? → `tests/`
- What are dependencies? → `requirements.txt`

Clear structure = easy automation.

### ✅ 3. Collaboration-Ready
New team member onboards in 5 minutes:
1. Clone repo
2. Read README
3. Run setup commands
4. Start working

### ✅ 4. Tool Integration
Each tool knows its place:
- Git → Code
- DVC → Data (data/)
- MLflow → Models (models/)
- GitHub Actions → CI/CD (.github/workflows/)

---

## 🚨 Common Mistakes (What We AVOIDED)

### ❌ Mistake 1: Flat Structure
```
project/
├── train.py
├── preprocess.py
├── serve.py
├── test1.py
└── utils_final_v2.py  # 😱
```
**Problem:** Scales terribly. Hard to navigate.

### ❌ Mistake 2: Mixing Notebooks and Scripts
```
src/
├── train.py
└── experiment.ipynb  # 😱 Doesn't belong here
```
**Problem:** Notebooks aren't production code.

### ❌ Mistake 3: No Configuration File
All settings hardcoded in scripts.
**Problem:** Can't experiment easily. Must edit code each time.

### ❌ Mistake 4: Committing Data to Git
```
git add data/large_dataset.csv  # 😱 GBs in Git
```
**Problem:** Bloats repo, slow clones.

### ❌ Mistake 5: Poor README
"Here's my ML project" + no setup instructions
**Problem:** No one can reproduce your work.

---

## 🎯 What This Structure Enables (Future Phases)

### Phase 2 (DVC):
```
data/raw/
  └── dataset.csv  ← We'll track this with DVC
```

### Phase 4 (MLflow):
```
src/train.py  ← Will log to MLflow
models/      ← MLflow stores artifacts here
```

### Phase 7 (FastAPI):
```
src/serve.py  ← API code goes here
```

### Phase 8 (CI/CD):
```
.github/workflows/train-deploy.yml  ← Automation pipeline
```

**Every future component has a clear home.**

---

## 🎤 Interview Talking Points

**Q: Why did you structure your project this way?**  
**A:** "I separated concerns: `src/` for production code, `notebooks/` for exploration, `tests/` for validation. This makes the project automation-friendly—CI/CD knows exactly where to find training scripts and tests. It also follows industry standards, making collaboration easier."

**Q: How do you manage dependencies?**  
**A:** "I use `requirements.txt` to pin exact versions of all libraries. This ensures reproducibility—anyone can recreate the exact environment that produced a specific model. I also use virtual environments to isolate dependencies."

**Q: How do you prevent data from bloating your Git repo?**  
**A:** "I use `.gitignore` to exclude data files from Git. Instead, I track data with DVC, which is designed for large files and provides deduplication and versioning without bloating the repo."

---

## ✅ Phase 1 Success Checklist

Before moving to Phase 2, verify:

- [x] Repository initialized with Git
- [x] Proper directory structure created
- [x] `.gitignore` excludes data, models, and secrets
- [x] `requirements.txt` has all necessary dependencies
- [x] Configuration file created
- [x] README documents the project
- [x] Initial commit made
- [x] You understand WHY each directory exists

---

## 🚀 What's Next?

**Phase 2: Data Versioning with DVC**

You'll learn:
- How to track datasets like code
- How to share data without bloating Git
- How to version data pipelines
- How to ensure anyone can get the exact data you used

**When to proceed:** Once you understand:
1. Why we separate `src/` from `notebooks/`
2. What `.gitignore` prevents
3. Why configuration files matter
4. How this structure enables automation

---

**Reply with "Ready for Phase 2" or ask any questions about the structure.**
