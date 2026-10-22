"""
Visualization Module for Social Media Sentiment Big Data Project.

Generates sleek, dark-themed charts aligned with modern dashboard aesthetics:
- Sentiment Donut / Bar distributions
- Top n-gram horizontal bar charts
- Confusion Matrix heatmaps
- Model benchmark comparison charts
- Temporal trend charts
- Airline sentiment breakdown charts
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Color palette definition for dark theme
COLOR_BG = "#0e1117"
COLOR_CARD = "#161b22"
COLOR_TEXT = "#e6edf3"
COLOR_MUTED = "#8b949e"
COLOR_GRID = "#21262d"

COLOR_POSITIVE = "#2ea043"   # GitHub Emerald
COLOR_NEGATIVE = "#da3633"   # Crimson Red
COLOR_NEUTRAL = "#58a6ff"    # Light Sky Blue
COLOR_ACCENT = "#1f6feb"     # Deep Blue

SENTIMENT_COLORS = {
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral": COLOR_NEUTRAL,
}


def apply_dark_theme(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent dark-mode styling to matplotlib figure and axis."""
    fig.patch.set_facecolor(COLOR_CARD)
    ax.set_facecolor(COLOR_CARD)
    ax.tick_params(colors=COLOR_TEXT, which="both", labelsize=10)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
    ax.grid(color=COLOR_GRID, linestyle="--", linewidth=0.6, alpha=0.7)


def plot_sentiment_donut(distribution_df: pd.DataFrame) -> plt.Figure:
    """
    Generate a modern donut chart for sentiment class distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=COLOR_CARD)
    labels = distribution_df["sentiment"].tolist()
    counts = distribution_df["count"].tolist()
    colors = [SENTIMENT_COLORS.get(str(l).lower(), COLOR_ACCENT) for l in labels]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor=COLOR_CARD, linewidth=2),
    )

    for text in texts:
        text.set_color(COLOR_TEXT)
        text.set_fontsize(11)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title("Sentiment Class Distribution", color=COLOR_TEXT, fontsize=13, pad=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_top_terms(df_terms: pd.DataFrame, title: str = "Top Terms", bar_color: str = COLOR_ACCENT) -> plt.Figure:
    """
    Generate a clean horizontal bar chart for top keywords or n-grams.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=COLOR_CARD)
    apply_dark_theme(ax, fig)

    if df_terms.empty:
        ax.text(0.5, 0.5, "No terms available", ha="center", va="center", color=COLOR_MUTED)
        return fig

    # Invert so highest count is at the top
    sorted_df = df_terms.sort_values(by="count", ascending=True)
    bars = ax.barh(sorted_df["term"], sorted_df["count"], color=bar_color, height=0.65, edgecolor="none")

    # Add count labels beside bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (w * 0.015), bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                ha="left", va="center", color=COLOR_MUTED, fontsize=8.5)

    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Frequency Count", fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot an annotated heatmap of the evaluation confusion matrix.
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=COLOR_CARD)
    apply_dark_theme(ax, fig)

    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold", "color": "#ffffff"},
    )

    ax.set_title("Test Set Confusion Matrix", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground Truth Class", fontsize=10, labelpad=8)
    fig.tight_layout()
    return fig


def plot_model_comparison_bar(results_dict: Dict[str, Any]) -> plt.Figure:
    """
    Compare baseline Multinomial Naive Bayes against Logistic Regression.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=COLOR_CARD)
    apply_dark_theme(ax, fig)

    models_info = results_dict.get("models", {})
    if not models_info:
        ax.text(0.5, 0.5, "No model comparison data", ha="center", va="center", color=COLOR_MUTED)
        return fig

    model_names = []
    accuracies = []
    f1_scores = []

    for key, data in models_info.items():
        name = data.get("algorithm", key.replace("_", " ").title())
        model_names.append(name)
        accuracies.append(data.get("accuracy", 0) * 100)
        f1_scores.append(data.get("f1_macro", 0) * 100)

    x = np.arange(len(model_names))
    width = 0.35

    rects1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#1f6feb")
    rects2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score (%)", color="#2ea043")

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom",
                color=COLOR_TEXT, fontsize=9, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom",
                color=COLOR_TEXT, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Model Benchmark Comparison", fontsize=12, pad=12, fontweight="bold")
    ax.legend(facecolor=COLOR_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    fig.tight_layout()
    return fig


def plot_temporal_trends(trend_df: pd.DataFrame, date_col: str = "period") -> plt.Figure:
    """
    Line chart showing sentiment evolution over time periods.
    """
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLOR_CARD)
    apply_dark_theme(ax, fig)

    if trend_df.empty or date_col not in trend_df.columns:
        ax.text(0.5, 0.5, "No temporal data available", ha="center", va="center", color=COLOR_MUTED)
        return fig

    periods = trend_df[date_col].astype(str)
    for col in trend_df.columns:
        if col != date_col:
            color = SENTIMENT_COLORS.get(str(col).lower(), COLOR_ACCENT)
            ax.plot(periods, trend_df[col], marker="o", linewidth=2, label=str(col).capitalize(), color=color)

    ax.set_title("Sentiment Dynamics Over Time", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Time Period", fontsize=10)
    ax.set_ylabel("Tweet Count", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    ax.legend(facecolor=COLOR_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    fig.tight_layout()
    return fig
