"""
Transformer-Based NLP Module for Contextual Social Media Sentiment Analysis.

Integrates:
- Pretrained BERT: 'distilbert-base-uncased-finetuned-sst-2-english' (Hugging Face / Stanford SST-2)
- Pretrained RoBERTa: 'cardiffnlp/twitter-roberta-base-sentiment-latest' (Cardiff NLP Twitter Sentiment)
- CPU-compatible inference with batching support
- Proper label mappings (negative, neutral, positive)
- Evaluation metric computation (Accuracy, Precision, Recall, Macro F1, Confusion Matrix)
- Memory management (lazy-loading and explicit cache unloading)
- Resilient error handling and empty input protection
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Configuration defaults
DEFAULT_BERT_MODEL = os.getenv("BERT_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
DEFAULT_ROBERTA_MODEL = os.getenv("ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Model Registry Metadata
TRANSFORMER_REGISTRY = {
    "bert": {
        "model_id": DEFAULT_BERT_MODEL,
        "name": "DistilBERT (SST-2 Finetuned)",
        "source": "Hugging Face (distilbert/distilbert-base-uncased-finetuned-sst-2-english)",
        "type": "BERT architecture (DistilBERT distilled version for fast CPU inference)",
        "classes": ["negative", "positive"],
        "label_map": {
            "NEGATIVE": "negative",
            "POSITIVE": "positive",
            "LABEL_0": "negative",
            "LABEL_1": "positive",
        },
    },
    "roberta": {
        "model_id": DEFAULT_ROBERTA_MODEL,
        "name": "Twitter RoBERTa (CardiffNLP)",
        "source": "Hugging Face (cardiffnlp/twitter-roberta-base-sentiment-latest)",
        "type": "RoBERTa architecture specifically trained on 124M Twitter social media posts",
        "classes": ["negative", "neutral", "positive"],
        "label_map": {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        },
    },
}


class TransformerSentimentModel:
    """
    Production-ready wrapper for Pretrained Transformer sentiment models.
    Supports on-demand loading, GPU/CPU execution, and batch inference.
    """

    def __init__(self, model_key: str = "bert", model_name_or_path: Optional[str] = None, device: int = -1):
        """
        Initialize the transformer model container without immediately allocating GPU/RAM.

        Args:
            model_key: Identifier key ('bert' or 'roberta').
            model_name_or_path: Optional custom Hugging Face model checkpoint or path.
            device: -1 for CPU (default, safe for local environments), or >= 0 for CUDA GPU.
        """
        self.model_key = model_key.lower()
        if self.model_key not in TRANSFORMER_REGISTRY:
            self.model_key = "bert"

        meta = TRANSFORMER_REGISTRY[self.model_key]
        self.model_name = model_name_or_path or meta["model_id"]
        self.display_name = meta["name"]
        self.source = meta["source"]
        self.model_type_desc = meta["type"]
        self.label_map = meta["label_map"]
        self.classes = meta["classes"]
        self.device = device
        self._pipeline = None
        self._is_loaded = False

    def load(self) -> None:
        """Explicitly load the transformer pipeline into memory."""
        if self._is_loaded and self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=128,
                top_k=None,  # Return probabilities for all classes
            )
            self._is_loaded = True
        except Exception as e:
            self._is_loaded = False
            self._pipeline = None
            raise RuntimeError(
                f"Failed to load transformer model '{self.model_name}'. "
                f"Error details: {e}. Check network connection or local HuggingFace cache."
            ) from e

    def unload(self) -> None:
        """Unload model from memory to free RAM/VRAM when switching models."""
        self._pipeline = None
        self._is_loaded = False
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _normalize_label(self, raw_label: str) -> str:
        """Map raw model output labels to canonical 'negative', 'neutral', or 'positive'."""
        raw_upper = raw_label.strip()
        if raw_upper in self.label_map:
            return self.label_map[raw_upper]
        raw_lower = raw_upper.lower()
        if raw_lower in self.label_map:
            return self.label_map[raw_lower]
        if "neg" in raw_lower:
            return "negative"
        if "pos" in raw_lower:
            return "positive"
        return "neutral"

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on an individual social media text string.

        Args:
            text: Raw or preprocessed social media post.

        Returns:
            Structured dict containing input text, predicted sentiment, confidence,
            all class probabilities, latency, and model metadata.
        """
        start_time = time.perf_counter()

        # Handle empty or whitespace-only text
        if not text or not str(text).strip():
            return {
                "input_text": text,
                "predicted_sentiment": "neutral" if "neutral" in self.classes else "negative",
                "confidence": 0.5,
                "probabilities": {cls: 1.0 / len(self.classes) for cls in self.classes},
                "model_name": self.model_name,
                "display_name": self.display_name,
                "source": self.source,
                "latency_ms": 0.1,
                "status": "empty_input_fallback",
            }

        if not self._is_loaded:
            self.load()

        clean_input = str(text).strip()
        # Truncate overly long text defensively
        if len(clean_input) > 1000:
            clean_input = clean_input[:1000]

        outputs = self._pipeline(clean_input)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Output format when top_k=None is a list of dicts: [{'label': ..., 'score': ...}, ...]
        # or nested [[{'label': ..., 'score': ...}]]
        raw_scores = outputs[0] if isinstance(outputs, list) and isinstance(outputs[0], list) else outputs

        prob_dict = {}
        best_label = None
        best_score = -1.0

        for item in raw_scores:
            mapped_lbl = self._normalize_label(item["label"])
            score = float(round(item["score"], 4))
            prob_dict[mapped_lbl] = score
            if score > best_score:
                best_score = score
                best_label = mapped_lbl

        return {
            "input_text": text,
            "predicted_sentiment": best_label or "positive",
            "confidence": round(best_score, 4),
            "probabilities": prob_dict,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "source": self.source,
            "architecture": self.model_type_desc,
            "latency_ms": latency_ms,
            "status": "success",
        }

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Run batched inference over a list of texts for high throughput.
        """
        if not texts:
            return []

        if not self._is_loaded:
            self.load()

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for text in chunk:
                results.append(self.predict(text))
        return results

    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "sentiment_label",
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate empirical evaluation metrics on labeled test data.

        Returns:
            Dictionary with accuracy, precision, recall, f1_score, and confusion matrix.
        """
        df = test_df.copy()
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)

        y_true = df[label_column].astype(str).str.lower().tolist()
        texts = df[text_column].fillna("").astype(str).tolist()

        preds = self.predict_batch(texts)
        y_pred = [p["predicted_sentiment"] for p in preds]

        # Filter to matching evaluation labels
        labels_present = sorted(list(set(y_true).union(set(y_pred))))

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels_present).tolist()

        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "sample_size": len(df),
            "labels": labels_present,
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "confusion_matrix": cm,
        }


# Singleton model cache to avoid re-instantiation across UI runs
_MODEL_CACHE: Dict[str, TransformerSentimentModel] = {}


def get_transformer_model(model_key: str = "bert") -> TransformerSentimentModel:
    """Retrieve or create a cached instance of the requested transformer model."""
    key = model_key.lower()
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = TransformerSentimentModel(model_key=key)
    return _MODEL_CACHE[key]


def predict_transformer(text: str, model_type: str = "bert") -> Dict[str, Any]:
    """Convenience helper for single-tweet prediction."""
    model = get_transformer_model(model_type)
    return model.predict(text)


def compare_with_baseline(text: str) -> Dict[str, Any]:
    """
    Execute inference across all models:
    1. Baseline Multinomial Naive Bayes
    2. Baseline Logistic Regression
    3. Pretrained BERT (DistilBERT SST-2)
    4. Pretrained RoBERTa (Twitter RoBERTa)
    """
    from src.models.predict import predict_text, load_model
    from src.data_processing.load_dataset import get_project_root

    root = get_project_root()
    nb_path = root / "models" / "sentiment_model.joblib"
    lr_path = root / "models" / "logistic_regression_model.joblib"

    comparison = {}

    # 1. Baseline Naive Bayes
    try:
        nb_res = predict_text(text, model_path=nb_path)
        comparison["multinomial_naive_bayes"] = {
            "name": "Multinomial Naive Bayes (Baseline)",
            "sentiment": nb_res["sentiment"],
            "confidence": nb_res["confidence"],
            "type": "TF-IDF + Naive Bayes (Laplace alpha=1.0)",
        }
    except Exception as e:
        comparison["multinomial_naive_bayes"] = {"error": str(e)}

    # 2. Logistic Regression
    try:
        lr_res = predict_text(text, model_path=lr_path)
        comparison["logistic_regression"] = {
            "name": "Logistic Regression (Comparison)",
            "sentiment": lr_res["sentiment"],
            "confidence": lr_res["confidence"],
            "type": "TF-IDF + L-BFGS Logistic Regression",
        }
    except Exception as e:
        comparison["logistic_regression"] = {"error": str(e)}

    # 3. Pretrained BERT
    try:
        bert_model = get_transformer_model("bert")
        bert_res = bert_model.predict(text)
        comparison["bert"] = {
            "name": bert_res["display_name"],
            "sentiment": bert_res["predicted_sentiment"],
            "confidence": bert_res["confidence"],
            "type": bert_res["architecture"],
            "latency_ms": bert_res["latency_ms"],
        }
    except Exception as e:
        comparison["bert"] = {"error": str(e)}

    # 4. Pretrained RoBERTa
    try:
        roberta_model = get_transformer_model("roberta")
        roberta_res = roberta_model.predict(text)
        comparison["roberta"] = {
            "name": roberta_res["display_name"],
            "sentiment": roberta_res["predicted_sentiment"],
            "confidence": roberta_res["confidence"],
            "type": roberta_res["architecture"],
            "latency_ms": roberta_res["latency_ms"],
        }
    except Exception as e:
        comparison["roberta"] = {"error": str(e)}

    return {
        "input_text": text,
        "models": comparison,
    }
