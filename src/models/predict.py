"""
Inference and Prediction Pipeline for Social Media Sentiment Analysis.

Supports:
- Single-tweet real-time prediction with class confidence scores
- Batch DataFrame and CSV prediction
- Safe model lazy loading with caching
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.data_processing.load_dataset import get_project_root
from src.data_processing.preprocess import clean_text

PROJECT_ROOT = get_project_root()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"

_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def load_model(model_path: Optional[Union[str, Path]] = None):
    """
    Load and cache the trained sentiment Pipeline model.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if model_path is None:
        target_path = DEFAULT_MODEL_PATH
    else:
        target_path = Path(model_path)

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == target_path:
        return _CACHED_MODEL

    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {target_path}. "
            "Please run 'python -m src.models.train' first to train and save the model."
        )

    _CACHED_MODEL = joblib.load(target_path)
    _CACHED_MODEL_PATH = target_path
    return _CACHED_MODEL


def predict_text(text: str, model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Predict sentiment for an individual tweet string.

    Args:
        text: Raw social media tweet text.
        model_path: Optional custom model file path.

    Returns:
        Dictionary with cleaned text, sentiment label ('negative' or 'positive'),
        confidence score, and probability distribution.
    """
    model = load_model(model_path)
    cleaned = clean_text(text, remove_stopwords=True)

    # If text is empty after cleaning, handle edge case gracefully
    if not cleaned.strip():
        cleaned = text.strip() or "empty"

    pred_class = int(model.predict([cleaned])[0])
    label = "positive" if pred_class == 1 else "negative"

    confidence = 0.5
    probabilities = {"negative": 0.5, "positive": 0.5}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        prob_neg = float(round(proba[0], 4))
        prob_pos = float(round(proba[1], 4))
        probabilities = {"negative": prob_neg, "positive": prob_pos}
        confidence = prob_pos if pred_class == 1 else prob_neg

    return {
        "original_text": text,
        "clean_text": cleaned,
        "sentiment": label,
        "sentiment_code": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


def predict_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    model_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Batch predict sentiments for all rows in a DataFrame.

    Args:
        df: Input DataFrame.
        text_column: Name of the column containing tweet texts.
        model_path: Optional custom model path.

    Returns:
        DataFrame augmented with 'clean_text', 'predicted_sentiment', and 'confidence' columns.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Specified text column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    model = load_model(model_path)
    df_out = df.copy()

    # Preprocess text
    df_out["clean_text"] = df_out[text_column].astype(str).apply(
        lambda x: clean_text(x, remove_stopwords=True)
    )
    # Fill any empty cleaned rows with original text fallback
    mask_empty = df_out["clean_text"].str.strip() == ""
    df_out.loc[mask_empty, "clean_text"] = df_out.loc[mask_empty, text_column].astype(str)

    # Batch prediction
    preds = model.predict(df_out["clean_text"])
    label_map = {0: "negative", 1: "positive"}
    df_out["predicted_sentiment"] = [label_map[p] for p in preds]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df_out["clean_text"])
        # Confidence is the max probability assigned to the chosen class
        confidences = np.max(probas, axis=1)
        df_out["confidence"] = np.round(confidences, 4)
    else:
        df_out["confidence"] = 1.0

    return df_out


def predict_csv(
    input_csv_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    text_column: str = "text",
    model_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Perform batch prediction on an uploaded or local CSV file.
    """
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")

    # Try common encodings
    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            df = pd.read_csv(input_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV using standard encodings (utf-8, latin-1).")

    # Auto-detect text column if specified column is not present
    if text_column not in df.columns:
        possible_text_cols = [c for c in df.columns if any(k in c.lower() for k in ["text", "tweet", "comment", "message", "content"])]
        if possible_text_cols:
            text_column = possible_text_cols[0]
        else:
            raise ValueError(
                f"Could not identify a text column. Available columns: {list(df.columns)}"
            )

    result_df = predict_dataframe(df, text_column=text_column, model_path=model_path)

    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)

    return result_df
