"""
Master Index - Data Science Master System

Interactive navigation and progress tracking for the notebook curriculum.
"""

import os
import json
from pathlib import Path
from datetime import datetime

NOTEBOOK_STRUCTURE = {
    "00_getting_started": {
        "title": "Getting Started",
        "notebooks": [
            ("00", "installation_setup", "Environment Setup", 15),
            ("01", "quick_start_tutorial", "Quick Start", 20),
            ("02", "basic_data_analysis", "Data Analysis", 30),
            ("03", "eda_visualization", "EDA & Viz", 30),
            ("04", "first_ml_model", "First ML Model", 45),
        ]
    },
    "01_core_ml": {
        "title": "Core Machine Learning",
        "notebooks": [
            ("05", "classification_models", "Classification", 60),
            ("06", "regression_models", "Regression", 60),
            ("07", "clustering_models", "Clustering", 45),
            ("08", "feature_engineering", "Feature Engineering", 60),
            ("09", "hyperparameter_tuning", "HPO", 45),
            ("10", "model_evaluation", "Evaluation", 45),
        ]
    },
    "02_computer_vision": {
        "title": "Computer Vision",
        "notebooks": [
            ("11", "cv_image_classification", "Image Classification", 60),
            ("12", "cv_object_detection", "Object Detection", 75),
            ("13", "cv_image_segmentation", "Segmentation", 90),
            ("14", "cv_generative_models", "GANs", 90),
        ]
    },
    "03_nlp_transformers": {
        "title": "NLP & Transformers",
        "notebooks": [
            ("15", "nlp_text_classification", "Text Classification", 45),
            ("16", "nlp_sentiment_analysis", "Sentiment", 45),
            ("17", "nlp_transformers_advanced", "Transformers", 60),
            ("18", "nlp_text_generation", "Text Generation", 75),
            ("19", "nlp_question_answering", "QA", 60),
            ("20", "nlp_multilingual", "Multilingual", 60),
        ]
    },
    "04_time_series": {
        "title": "Time Series",
        "notebooks": [
            ("21", "time_series_forecasting", "Forecasting", 75),
            ("22", "anomaly_detection", "Anomaly Detection", 45),
        ]
    },
    "05_advanced_topics": {
        "title": "Advanced Topics",
        "notebooks": [
            ("23", "recommender_systems", "Recommenders", 60),
            ("24", "graph_neural_networks", "GNN", 75),
            ("25", "reinforcement_learning", "RL", 90),
        ]
    },
    "06_production": {
        "title": "Production & Deployment",
        "notebooks": [
            ("26", "model_deployment_api", "API Deployment", 60),
            ("27", "mlops_pipelines", "MLOps", 75),
            ("28", "cloud_deployment", "Cloud", 60),
            ("29", "model_monitoring", "Monitoring", 45),
        ]
    },
}


def print_curriculum():
    """Print the complete curriculum."""
    print("=" * 60)
    print("📚 DATA SCIENCE MASTER SYSTEM - CURRICULUM")
    print("=" * 60)
    
    total_time = 0
    for folder, info in NOTEBOOK_STRUCTURE.items():
        print(f"\n📁 {info['title'].upper()}")
        print("-" * 40)
        for num, name, title, time_min in info['notebooks']:
            print(f"  {num}. {title:<30} ({time_min} min)")
            total_time += time_min
    
    print(f"\n{'=' * 60}")
    print(f"📊 TOTAL: 30 Notebooks | {total_time} minutes (~{total_time//60} hours)")
    print("=" * 60)


def get_learning_path(track="full"):
    """Get learning path by track."""
    paths = {
        "beginner": ["00_getting_started", "01_core_ml"],
        "cv": ["00_getting_started", "01_core_ml", "02_computer_vision"],
        "nlp": ["00_getting_started", "01_core_ml", "03_nlp_transformers"],
        "full": list(NOTEBOOK_STRUCTURE.keys())
    }
    return paths.get(track, paths["full"])


def check_prerequisites():
    """Check if required packages are installed."""
    required = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']
    optional = ['torch', 'transformers', 'optuna', 'mlflow']
    
    print("📦 Package Check:")
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} (required)")
    
    print("\n📦 Optional Packages:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚪ {pkg} (optional)")


if __name__ == "__main__":
    print_curriculum()
    print("\n")
    check_prerequisites()
