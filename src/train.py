"""
Model Training with MLflow Integration

Industry-grade training script that logs everything to MLflow:
- Hyperparameters
- Metrics (train, validation, test)
- Model artifacts
- Plots and visualizations
- Data version information
"""

import pandas as pd
import numpy as np
import yaml
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(train_path: str, test_path: str):
    """
    Load preprocessed train and test data.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"✅ Loaded training data: {X_train.shape}")
    print(f"✅ Loaded test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def create_model(algorithm: str, hyperparameters: dict):
    """
    Create model based on algorithm and hyperparameters.
    
    Args:
        algorithm: Model type ('random_forest', 'logistic_regression')
        hyperparameters: Dictionary of model hyperparameters
        
    Returns:
        Scikit-learn model instance
    """
    if algorithm == "random_forest":
        model = RandomForestClassifier(**hyperparameters)
    elif algorithm == "logistic_regression":
        model = LogisticRegression(**hyperparameters)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def evaluate_model(model, X, y, dataset_name: str = "test"):
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        dataset_name: Name of dataset for logging
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
        f'{dataset_name}_precision': precision_score(y, y_pred, average='binary'),
        f'{dataset_name}_recall': recall_score(y, y_pred, average='binary'),
        f'{dataset_name}_f1': f1_score(y, y_pred, average='binary'),
        f'{dataset_name}_roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ Saved confusion matrix to {save_path}")


def plot_feature_importance(model, feature_names, save_path: str, top_n: int = 10):
    """
    Create and save feature importance plot.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        save_path: Path to save plot
        top_n: Number of top features to show
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"✅ Saved feature importance to {save_path}")
    else:
        print("⚠️  Model does not have feature_importances_ attribute")


def train_model(config: dict, run_name: str = None):
    """
    Train model with MLflow tracking.
    
    Args:
        config: Configuration dictionary
        run_name: Optional name for the MLflow run
    """
    print("\n" + "="*60)
    print("🚀 STARTING MODEL TRAINING WITH MLFLOW")
    print("="*60 + "\n")
    
    # Set MLflow experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"📊 MLflow Run ID: {run_id}")
        print(f"📊 Experiment: {config['mlflow']['experiment_name']}")
        
        # Load data
        print("\n📥 STEP 1: Loading data...")
        X_train, X_test, y_train, y_test = load_data(
            train_path=f"{config['data']['processed_data_path']}/train.csv",
            test_path=f"{config['data']['processed_data_path']}/test.csv"
        )
        
        # Log data info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", str(list(X_train.columns)))
        
        # Create model
        print("\n🔧 STEP 2: Creating model...")
        algorithm = config['model']['algorithm']
        hyperparameters = config['model']['hyperparameters']
        
        print(f"Algorithm: {algorithm}")
        print(f"Hyperparameters: {hyperparameters}")
        
        model = create_model(algorithm, hyperparameters)
        
        # Log hyperparameters
        mlflow.log_param("algorithm", algorithm)
        for param, value in hyperparameters.items():
            mlflow.log_param(param, value)
        
        # Train model
        print("\n🎯 STEP 3: Training model...")
        model.fit(X_train, y_train)
        print("✅ Model trained successfully")
        
        # Cross-validation on training set
        print("\n📊 STEP 4: Cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std", cv_std)
        
        # Evaluate on training set
        print("\n📈 STEP 5: Evaluating on training set...")
        train_metrics, _, _ = evaluate_model(model, X_train, y_train, "train")
        for metric_name, metric_value in train_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
            mlflow.log_metric(metric_name, metric_value)
        
        # Evaluate on test set
        print("\n📈 STEP 6: Evaluating on test set...")
        test_metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, "test")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
            mlflow.log_metric(metric_name, metric_value)
        
        # Generate classification report
        print("\n📋 Classification Report:")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Save and log artifacts
        print("\n💾 STEP 7: Saving artifacts...")
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        cm_path = artifacts_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, str(cm_path))
        mlflow.log_artifact(str(cm_path))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            fi_path = artifacts_dir / "feature_importance.png"
            plot_feature_importance(model, list(X_train.columns), str(fi_path))
            mlflow.log_artifact(str(fi_path))
        
        # Save classification report
        report_path = artifacts_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))
        
        # Log model
        print("\n🎯 STEP 8: Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"{config['mlflow']['experiment_name']}-{algorithm}"
        )
        print("✅ Model logged to MLflow")
        
        # Save model locally
        model_dir = Path(config['paths']['models_dir'])
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"model_{algorithm}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved locally to {model_path}")
        
        # Save metadata
        metadata = {
            "run_id": run_id,
            "experiment_name": config['mlflow']['experiment_name'],
            "algorithm": algorithm,
            "hyperparameters": hyperparameters,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "metrics": {**train_metrics, **test_metrics},
            "cv_accuracy": {"mean": cv_mean, "std": cv_std},
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = model_dir / f"metadata_{algorithm}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to {metadata_path}")
        
        # Log tags
        mlflow.set_tag("model_type", algorithm)
        mlflow.set_tag("data_version", "v1.0")
        mlflow.set_tag("stage", "development")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"\n🔗 View run in MLflow UI:")
        print(f"   mlflow ui --port 5000")
        print(f"   http://localhost:5000/#/experiments/...")
        print(f"\n📊 Key Metrics:")
        print(f"   Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"   Test F1-Score: {test_metrics['test_f1']:.4f}")
        print(f"   Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")


if __name__ == "__main__":
    config = load_config()
    
    # Generate run name with timestamp
    run_name = f"{config['model']['algorithm']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    train_model(config, run_name=run_name)
