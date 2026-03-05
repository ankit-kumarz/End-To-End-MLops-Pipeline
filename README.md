# End-to-End MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/DVC-3.65.0-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.8.0-blue.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.1-green.svg)](https://fastapi.tiangolo.com/) 

A production-ready MLOps pipeline implementing industry best practices for machine learning workflows, from data versioning to model deployment with automated CI/CD.
 
## 🎯 Overview 
     
This project demonstrates a complete MLOps system featuring:     
    
- **Data Versioning** with DVC (Data Version Control)   
- **Automated ML Pipelines** with reproducible workflows  
- **Experiment Tracking** using MLflow
- **Model Serving** via FastAPI REST API   
- **CI/CD Automation** with GitHub Actions
- **Comprehensive Testing** with pytest  
- **Production-Ready Architecture** with multi-stage deployment  

## 🚀 Features  
 
### ✅ Data Management
- DVC for data versioning and tracking
- Automated data validation (schema, nulls, ranges, drift detection) 
- Data preprocessing pipeline with feature engineering 
 
### ✅ Model Training  
- MLflow experiment tracking with full metrics logging
- Cross-validation and performance evaluation
- Random Forest classifier achieving **99.5% test accuracy** 
- Hyperparameter configuration via YAML
 
### ✅ Model Serving
- FastAPI REST API with automatic documentation  
- Input validation using Pydantic  
- Health checks and model info endpoints
- Batch prediction support
- Swagger UI at `/docs`
  
### ✅ CI/CD Pipeline  
- Automated testing on every push
- Data validation workflows 
- Model training with performance thresholds   
- Multi-stage deployment (Staging → Production) 
- Scheduled retraining (weekly) 

### ✅ Testing
- Unit tests for data pipeline
- Data validation tests 
- API endpoint tests  
- Code coverage tracking  
 
## 📋 Prerequisites 
  
- Python 3.12    
- Git 
- Virtual environment (venv)  
  
## 🔧 Installation 

### 1. Clone the Repository 

```bash 
git clone https://github.com/ankit-kumarz/End-To-End-MLops-Pipeline.git  
cd End-To-End-MLops-Pipeline  
``` 

### 2. Create Virtual Environment 
    
```bash
# Windows
python -m venv venv  
.\venv\Scripts\activate 

# Linux/Mac   
python3 -m venv venv 
source venv/bin/activate 
```
 
### 3. Install Dependencies 

```bash 
pip install -r requirements.txt
```  

### 4. Initialize DVC
  
```bash 
dvc init
dvc repro  # Run the entire pipeline 
```  

## 🏃 Quick Start
  
### Generate Sample Data
  
```bash
python src/generate_data.py  
```
 
### Run Data Pipeline
 
```bash
# Validate data 
python src/validate_data.py

# Preprocess data
python src/data_pipeline.py 

# Or run the entire DVC pipeline
dvc repro   
```

### Train Model
 
```bash 
python src/train.py
```

### Start MLflow UI 
  
```bash
mlflow ui --port 5000
# Visit: http://localhost:5000
```
 
### Start API Server 
 
```bash 
# Method 1: Direct
python src/serve.py

# Method 2: With uvicorn (recommended) 
uvicorn src.serve:app --reload --port 8000 

# Visit: http://localhost:8000/docs
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \ 
  -H "Content-Type: application/json" \
  -d '{
    "feature_1": 0.5,
    "feature_2": 1.2,
    "feature_3": -0.3,
    "feature_4": 75.5,
    "feature_5": 5
  }'
``` 

## 🧪 Running Tests 

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term

# Run specific test file
pytest tests/test_data_pipeline.py -v
```

## 📂 Project Structure

```
End-To-End-MLops-Pipeline/
├── .github/ 
│   └── workflows/          # GitHub Actions CI/CD workflows
│       ├── ci-tests.yml            # Automated testing
│       ├── data-validation.yml     # Data quality checks
│       ├── model-training.yml      # Training pipeline
│       ├── deploy-api.yml          # Deployment pipeline
│       └── schedule-retrain.yml    # Scheduled retraining
│
├── config/
│   └── config.yaml         # Centralized configuration
│
├── data/
│   ├── raw/                # Raw data (DVC tracked)
│   └── processed/          # Processed data 
│ 
├── models/                 # Trained models and artifacts
│   ├── model_random_forest.pkl
│   ├── scaler.pkl
│   └── metadata_random_forest.json
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration loader
│   ├── utils.py            # Utility functions
│   ├── generate_data.py    # Sample data generation
│   ├── validate_data.py    # Data validation
│   ├── data_pipeline.py    # Data preprocessing
│   ├── train.py            # Model training with MLflow
│   └── serve.py            # FastAPI model serving
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_validation.py
│   └── test_api.py
│
├── notebooks/              # Jupyter notebooks for exploration
├── mlruns/                 # MLflow experiment tracking
├── .dvc/                   # DVC configuration
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC pipeline state
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔄 DVC Pipeline

The automated pipeline consists of three stages:

```yaml
stages:
  validate:     # Data quality validation
  preprocess:   # Feature engineering & scaling
  train:        # Model training with MLflow
```

Run the entire pipeline:

```bash
dvc repro
```

View pipeline DAG:

```bash
dvc dag
```

## 📊 MLflow Tracking

All experiments are tracked with:

- **Parameters**: algorithm, hyperparameters, data info
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC
- **Artifacts**: confusion matrix, feature importance, models
- **Model Registry**: versioned model storage

Access MLflow UI:

```bash
mlflow ui
# Visit: http://localhost:5000
```

## 🌐 API Endpoints

### Health Check
```
GET /health
```

### Model Information
```
GET /model/info
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "feature_1": 0.5,
  "feature_2": 1.2,
  "feature_3": -0.3,
  "feature_4": 75.5,
  "feature_5": 5
}
```

### Batch Prediction
```
POST /predict/batch 
Content-Type: application/json

{
  "instances": [
    {"feature_1": 0.5, ...},
    {"feature_1": -0.3, ...}
  ]
}
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤖 GitHub Actions Workflows

### CI Tests
- Triggers: Every push and pull request
- Actions: Run pytest, lint code, validate configs 

### Data Validation
- Triggers: Data file changes
- Actions: Schema validation, drift detection

### Model Training
- Triggers: Manual, after validation, weekly schedule
- Actions: Train model, check performance threshold, save artifacts

### API Deployment
- Triggers: After successful training
- Actions: Deploy to staging, run tests, deploy to production (with approval)

### Scheduled Retraining
- Triggers: Every Monday at 3 AM
- Actions: Check drift, trigger retraining if needed

## 🎯 Model Performance

Current model: **Random Forest Classifier**

| Metric | Score |
|--------|-------|
| Test Accuracy | 99.5% |
| Test Precision | 98.8% |
| Test Recall | 100.0% |
| Test F1-Score | 99.4% |
| Test ROC-AUC | 99.99% |
| CV Accuracy | 99.63% (±0.31%) |

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  algorithm: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

serving:
  host: "0.0.0.0"
  port: 8000

mlflow:
  experiment_name: "model-training"
```

## 🛠️ Development

### Adding New Features

1. Update code in `src/`
2. Add tests in `tests/`
3. Run tests: `pytest tests/ -v`
4. Commit and push (CI/CD will run automatically)

### Retraining Models

```bash
# Manual retraining
dvc repro

# Or trigger via GitHub Actions
# Go to Actions → Model Training Pipeline → Run workflow
```

### Deploying Updates

1. Push to `main` branch
2. GitHub Actions runs tests
3. If tests pass, model is trained
4. Model is deployed to staging
5. Manual approval required for production

## 📝 Dependencies

### Core
- `scikit-learn==1.8.0` - Machine learning
- `pandas==2.3.3` - Data manipulation
- `numpy==2.4.0` - Numerical computing

### MLOps Tools
- `dvc==3.65.0` - Data versioning
- `mlflow==3.8.0` - Experiment tracking
- `fastapi==0.127.1` - API framework
- `uvicorn==0.40.0` - ASGI server
- `pydantic==2.12.5` - Data validation

### Development
- `pytest==9.0.2` - Testing framework
- `pytest-cov==7.0.0` - Coverage reporting
- `flake8` - Code linting

See `requirements.txt` for full list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ankit Kumar**
- GitHub: [@ankit-kumarz](https://github.com/ankit-kumarz)
- Email: ankitrajj1068@gmail.com 

## 🙏 Acknowledgments

- Built with industry-standard MLOps practices.
- Inspired by real-world production systems.
- Designed for learning and interview preparation. 

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
