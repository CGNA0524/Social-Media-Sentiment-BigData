"""
Model Training Pipeline for Social Media Sentiment Analysis.

Builds and evaluates:
1. Multinomial Naive Bayes (Baseline, computationally lightweight, aligned with BDC curriculum)
2. Logistic Regression (High-accuracy linear baseline comparison)

Features:
- Stratified Train/Test split (no data leakage)
- Scikit-learn Pipeline incorporating TF-IDF Vectorization
- Artifact persistence in models/
- Machine-readable evaluation export in reports/experiment_results.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_processing.load_dataset import get_project_root, load_sentiment140
from src.data_processing.preprocess import clean_text
from src.models.evaluate import evaluate_predictions

PROJECT_ROOT = get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_pipeline(classifier_type: str = "naive_bayes", max_features: int = 25000, min_df: int = 2) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with TF-IDF Vectorizer and chosen classifier.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=min_df,
    )

    if classifier_type == "naive_bayes":
        classifier = MultinomialNB(alpha=1.0)
    elif classifier_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def train_and_evaluate(
    sample_size: int = 50000,
    compare_logistic: bool = True,
    max_features: int = 25000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Run the end-to-end training and evaluation pipeline.

    Args:
        sample_size: Number of samples to train on (e.g. 50,000 for fast laptop training, None for full).
        compare_logistic: Whether to train and compare Logistic Regression.
        max_features: Maximum TF-IDF vocabulary features.
        random_state: Random seed for split and reproducibility.
        test_size: Proportion of dataset held out for testing.

    Returns:
        Dictionary containing experimental metrics for all trained models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Ingesting Sentiment140 dataset (sample_size={sample_size})...")
    df = load_sentiment140(sample_size=sample_size, random_state=random_state)
    print(f"[*] Ingested {len(df)} records. Cleaning text...")

    # Preprocess text
    t0_prep = time.time()
    df["clean_text"] = df["text"].apply(lambda x: clean_text(x, remove_stopwords=True))
    # Drop rows that ended up empty
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    prep_duration = round(time.time() - t0_prep, 2)
    print(f"[+] Text cleaning completed in {prep_duration}s. Usable rows: {len(df)}")

    X = df["clean_text"]
    y = df["sentiment"]  # 0=negative, 1=positive

    print(f"[*] Performing stratified train/test split (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[+] Train set size: {len(X_train)} | Test set size: {len(X_test)}")

    experiment_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": random_state,
        "sample_size": sample_size,
        "actual_dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size_ratio": test_size,
        "max_features": max_features,
        "models": {},
    }

    # 1. Train Multinomial Naive Bayes (Baseline)
    print("\n[*] Training Multinomial Naive Bayes Baseline...")
    nb_pipe = build_pipeline(classifier_type="naive_bayes", max_features=max_features)
    t0_nb = time.time()
    nb_pipe.fit(X_train, y_train)
    nb_train_time = round(time.time() - t0_nb, 2)

    nb_pred = nb_pipe.predict(X_test)
    nb_proba = nb_pipe.predict_proba(X_test)
    nb_metrics = evaluate_predictions(y_test, nb_pred, nb_proba)
    nb_metrics["training_time_seconds"] = nb_train_time
    nb_metrics["algorithm"] = "Multinomial Naive Bayes"
    print(f"[+] Naive Bayes Test Accuracy: {nb_metrics['accuracy'] * 100:.2f}% | F1: {nb_metrics['f1_macro']:.4f} (trained in {nb_train_time}s)")

    # Save baseline model
    nb_model_path = MODELS_DIR / "sentiment_model.joblib"
    joblib.dump(nb_pipe, nb_model_path)
    print(f"[+] Saved baseline model to {nb_model_path}")

    experiment_results["models"]["multinomial_nb"] = nb_metrics

    # 2. Train Logistic Regression Comparison if requested
    if compare_logistic:
        print("\n[*] Training Logistic Regression Comparison...")
        lr_pipe = build_pipeline(classifier_type="logistic_regression", max_features=max_features)
        t0_lr = time.time()
        lr_pipe.fit(X_train, y_train)
        lr_train_time = round(time.time() - t0_lr, 2)

        lr_pred = lr_pipe.predict(X_test)
        lr_proba = lr_pipe.predict_proba(X_test)
        lr_metrics = evaluate_predictions(y_test, lr_pred, lr_proba)
        lr_metrics["training_time_seconds"] = lr_train_time
        lr_metrics["algorithm"] = "Logistic Regression"
        print(f"[+] Logistic Regression Test Accuracy: {lr_metrics['accuracy'] * 100:.2f}% | F1: {lr_metrics['f1_macro']:.4f} (trained in {lr_train_time}s)")

        # Save logistic regression model
        lr_model_path = MODELS_DIR / "logistic_regression_model.joblib"
        joblib.dump(lr_pipe, lr_model_path)
        print(f"[+] Saved Logistic Regression model to {lr_model_path}")

        experiment_results["models"]["logistic_regression"] = lr_metrics

    # Save experiment metrics to JSON
    json_path = REPORTS_DIR / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n[+] Experiment results successfully saved to {json_path}")

    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of records to sample for training (default: 50,000 for fast laptop training)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip Logistic Regression comparison",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the full 1.6M dataset",
    )
    args = parser.parse_args()

    sample_val = None if args.full else args.sample_size
    train_and_evaluate(sample_size=sample_val, compare_logistic=not args.no_compare)
