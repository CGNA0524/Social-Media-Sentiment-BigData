"""
Unit tests for Transformer-based Sentiment Analysis (BERT & RoBERTa).
"""

import pytest
from src.models.transformer_models import (
    TransformerSentimentModel,
    predict_transformer,
    compare_with_baseline,
    TRANSFORMER_REGISTRY,
)


def test_transformer_registry():
    assert "bert" in TRANSFORMER_REGISTRY
    assert "roberta" in TRANSFORMER_REGISTRY
    assert "distilbert" in TRANSFORMER_REGISTRY["bert"]["model_id"]
    assert "roberta" in TRANSFORMER_REGISTRY["roberta"]["model_id"]


def test_bert_single_prediction():
    text = "The delivery was fast, and the product quality is excellent."
    result = predict_transformer(text, model_type="bert")
    assert result["status"] == "success"
    assert result["predicted_sentiment"] == "positive"
    assert result["confidence"] > 0.80
    assert "probabilities" in result
    assert result["display_name"] == "DistilBERT (SST-2 Finetuned)"


def test_empty_input_fallback():
    result = predict_transformer("", model_type="bert")
    assert result["status"] == "empty_input_fallback"
    assert result["confidence"] == 0.5


def test_compare_with_baseline():
    text = "The product arrived quickly and exceeded my expectations!"
    comp = compare_with_baseline(text)
    assert comp["input_text"] == text
    models = comp["models"]
    assert "multinomial_naive_bayes" in models
    assert "logistic_regression" in models
    assert "bert" in models
    # Check that baseline models returned valid predictions
    assert models["multinomial_naive_bayes"]["sentiment"] in ["positive", "negative"]
    assert models["logistic_regression"]["sentiment"] in ["positive", "negative"]
    assert models["bert"]["sentiment"] == "positive"
