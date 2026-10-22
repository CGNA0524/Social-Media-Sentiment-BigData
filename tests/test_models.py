"""
Tests for Model Training, Evaluation, and Inference Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.models.evaluate import evaluate_predictions
from src.models.train import build_pipeline
from src.models.predict import predict_text, predict_dataframe, load_model


def test_evaluate_predictions_metrics():
    """Verify evaluation function calculates proper precision, recall, f1, and confusion matrix."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.75
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None


def test_build_pipeline_and_fixture_training():
    """Verify building and fitting a pipeline on a small synthetic fixture."""
    pipe = build_pipeline(classifier_type="naive_bayes", max_features=100, min_df=1)
    assert isinstance(pipe, Pipeline)

    X_train = ["terrible bad flight delay", "horrible service angry", "wonderful great flight love", "fantastic crew loved"]
    y_train = [0, 0, 1, 1]

    pipe.fit(X_train, y_train)
    preds = pipe.predict(["great flight"])
    assert preds[0] == 1


def test_predict_text_format():
    """Verify predict_text returns structured dictionary with valid confidence."""
    res = predict_text("I loved the flight experience!")
    assert isinstance(res, dict)
    assert "sentiment" in res
    assert res["sentiment"] in ["positive", "negative"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert "positive" in res["probabilities"]
    assert "negative" in res["probabilities"]


def test_batch_dataframe_prediction():
    """Verify predict_dataframe augments dataframe with predicted columns."""
    df = pd.DataFrame({
        "text": ["Best service ever!", "Worst delay and lost luggage"]
    })
    out = predict_dataframe(df, text_column="text")
    assert "predicted_sentiment" in out.columns
    assert "confidence" in out.columns
    assert len(out) == 2
    assert set(out["predicted_sentiment"].unique()).issubset({"positive", "negative"})


def test_predict_dataframe_missing_column():
    """Ensure error is raised if text column does not exist."""
    df = pd.DataFrame({"random_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="Specified text column"):
        predict_dataframe(df, text_column="nonexistent")
