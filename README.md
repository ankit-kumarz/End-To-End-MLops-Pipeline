# 🚀 End-to-End MLOps Pipeline

> **Industry-grade ML system with data versioning, experiment tracking, automated training, and CI/CD deployment**

[![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9+-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [MLOps Workflow](#mlops-workflow)

---

## 🎯 Project Overview

This project demonstrates a complete MLOps pipeline that solves real production problems:

- ✅ **Data Versioning**: Track datasets like code with DVC
- ✅ **Experiment Tracking**: Log every hyperparameter, metric, and artifact with MLflow
- ✅ **Reproducibility**: Anyone can recreate exact model results
- ✅ **Automated Training**: CI/CD pipelines handle retraining
- ✅ **Model Serving**: FastAPI endpoints for real-time predictions
- ✅ **Monitoring**: Track model performance over time

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Data  │────>│ DVC Pipeline │────>│  Processed  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Training   │
                    │   (MLflow)   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Model Registry│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  FastAPI API │
                    └──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Version Control** | Git + GitHub | Code versioning & collaboration |
| **Data Versioning** | DVC | Track datasets and data pipelines |
| **Experiment Tracking** | MLflow | Log experiments, models, metrics |
| **Model Serving** | FastAPI | REST API for predictions |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Containerization** | Docker | Consistent environments |
| **Language** | Python 3.9+ | Core development |

---

## 📂 Project Structure

```
End-To-End-MLOPs/
│
├── .github/
│   └── workflows/           # CI/CD pipeline definitions
│       └── train-deploy.yml
│
├── src/                     # Source code (production-ready)
│   ├── __init__.py
│   ├── data_pipeline.py     # Data preprocessing logic
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation
│   └── serve.py             # FastAPI serving logic
│
├── data/                    # Data storage (managed by DVC)
│   ├── raw/                 # Original, immutable data
│   └── processed/           # Cleaned, feature-engineered data
│
├── models/                  # Model artifacts (managed by MLflow)
│   └── .gitkeep
│
├── notebooks/               # Jupyter notebooks (EDA only, not production)
│   └── 01_exploratory_data_analysis.ipynb
│
├── tests/                   # Data validation & model tests
│   ├── test_data_pipeline.py
│   └── test_model.py
│
├── config/                  # Configuration files
│   └── config.yaml
│
├── .dvcignore              # DVC ignore patterns
├── .gitignore              # Git ignore patterns
├── dvc.yaml                # DVC pipeline definition (created later)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
└── README.md               # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git installed
- (Optional) Docker for containerization

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd End-To-End-MLOPs
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Initialize DVC
```bash
dvc init
```

### 5️⃣ Start MLflow Tracking Server (Optional)
```bash
mlflow ui --port 5000
```
Access at: http://localhost:5000

---

## 📖 Usage

### Training Pipeline
```bash
python src/train.py
```

### Model Serving
```bash
uvicorn src.serve:app --reload
```
Access API docs at: http://localhost:8000/docs

### Run Tests
```bash
pytest tests/
```

---

## 🔄 MLOps Workflow

### Phase 1: Data Preparation
1. Add raw data to `data/raw/`
2. Version data with DVC: `dvc add data/raw/dataset.csv`
3. Commit DVC metadata: `git commit -am "Track dataset"`

### Phase 2: Experimentation
1. Run training script: `python src/train.py`
2. MLflow logs metrics, params, and models automatically
3. Compare experiments in MLflow UI

### Phase 3: Model Selection
1. Choose best model from MLflow registry
2. Promote to "Production" stage
3. Model artifacts are versioned automatically

### Phase 4: Deployment
1. Push code to GitHub
2. CI/CD pipeline triggers automatically
3. Tests run → Model retrains → API deploys

### Phase 5: Monitoring
1. Track prediction latency
2. Monitor model performance metrics
3. Trigger retraining when drift detected

---

## 🎓 Learning Outcomes

By studying this project, you'll understand:

- ✅ How to version datasets efficiently
- ✅ How to track experiments systematically
- ✅ How to build reproducible ML pipelines
- ✅ How to serve models as production APIs
- ✅ How to automate ML workflows with CI/CD
- ✅ How to structure ML projects for collaboration

---

## 📝 Interview Talking Points

**Q: How do you ensure model reproducibility?**  
A: We use the "versioning trinity": Git for code, DVC for data, and MLflow for models. Every training run logs the exact code version, data version, and hyperparameters used.

**Q: How do you handle model deployment?**  
A: We use FastAPI to serve models as REST APIs. GitHub Actions automates testing and deployment. Every model version is containerized with Docker for consistency.

**Q: What happens when data drifts?**  
A: We monitor prediction distributions and model performance. When drift is detected, we trigger an automated retraining pipeline through our CI/CD system.

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new features
- Improve documentation
- Report issues
- Suggest better practices

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for production ML systems**
