"""
Aspect-Based Sentiment Analysis (ABSA) Engine for Social Media & Reviews.

Identifies:
1. Target aspects / entities / features (including multi-word compounds like 'battery life', 'customer service').
2. Clause boundaries and contrasting conjunctions ('but', 'however', 'although', 'yet', 'while').
3. Aspect-specific sentiment polarity ('positive', 'negative', 'neutral') and confidence scores.
4. Handles mixed sentiments, single sentiments, and inputs with no detectable aspects.
5. Produces schema compatible with Elasticsearch nested indexing and downstream analytics.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# Common aspect terms across social media, e-commerce, tech, and airlines
COMMON_ASPECT_VOCAB = {
    # Product / Tech
    "camera", "battery", "battery life", "screen", "display", "performance", "speed",
    "design", "build quality", "sound", "speaker", "audio", "software", "storage",
    "processor", "price", "cost", "value", "packaging", "delivery", "shipping",
    # Service / Social / Airlines
    "customer service", "customer support", "staff", "crew", "flight", "pilot",
    "seat", "legroom", "food", "meal", "drink", "wifi", "internet", "entertainment",
    "booking", "app", "website", "refund", "ticket", "baggage", "luggage", "delay",
    "cleanliness", "comfort", "atmosphere", "quality"
}

# Contrastive discourse markers that flip or contrast sentiment between clauses
CONTRASTIVE_MARKERS = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\beven though\b",
    r"\byet\b", r"\bwhile\b", r"\bnevertheless\b", r"\bnonetheless\b",
    r"\bon the other hand\b", r"\bwhereas\b"
]
CONTRASTIVE_REGEX = re.compile("|".join(CONTRASTIVE_MARKERS), re.IGNORECASE)

# Sentiment lexicons with negation awareness
POSITIVE_LEXICON = {
    "excellent", "amazing", "great", "good", "fast", "awesome", "fantastic", "wonderful",
    "superb", "brilliant", "smooth", "love", "loved", "perfect", "reliable", "impressed",
    "impressive", "outstanding", "flawless", "friendly", "helpful", "clean", "spacious",
    "comfortable", "prompt", "top", "best", "satisfied", "enjoyable", "worth", "stunning"
}

NEGATIVE_LEXICON = {
    "disappointing", "disappointed", "poor", "terrible", "awful", "bad", "horrible",
    "slow", "laggy", "broken", "useless", "worst", "unacceptable", "rude", "delayed",
    "uncomfortable", "cramped", "expensive", "waste", "hate", "hated", "flawed", "defective",
    "unresponsive", "frustrating", "frustrated", "annoying", "annoyed", "pathetic", "dirty"
}

NEGATION_TERMS = {"not", "never", "no", "cannot", "hardly", "barely", "neither", "without"}
INTENSIFIERS = {"very", "extremely", "really", "super", "highly", "absolutely", "exceptionally"}


class AspectBasedSentimentAnalyzer:
    """
    Aspect-Based Sentiment Analysis module.
    Extracts explicit opinion targets and calculates aspect-targeted polarity.
    """

    def __init__(self, custom_aspects: Optional[List[str]] = None):
        self.aspect_vocab = set(COMMON_ASPECT_VOCAB)
        if custom_aspects:
            for asp in custom_aspects:
                self.aspect_vocab.add(asp.lower().strip())

        # Compile regex for fast compound aspect matching (sorted longest first to match compounds before unigrams)
        sorted_vocab = sorted(list(self.aspect_vocab), key=lambda x: len(x.split()), reverse=True)
        escaped_vocab = [r"\b" + re.escape(term) + r"\b" for term in sorted_vocab]
        self.aspect_pattern = re.compile("|".join(escaped_vocab), re.IGNORECASE)

    def _split_into_clauses(self, text: str) -> List[Tuple[str, str]]:
        """
        Segment text into logical clauses separated by contrastive markers or punctuation.
        Returns list of (clause_text, marker_type).
        """
        # Split text on punctuation semicolons, periods, or contrastive markers
        clauses = []
        # Split on sentence boundaries first
        raw_sentences = re.split(r"[.!?]+", text)
        for sent in raw_sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # Check for contrastive markers within the sentence
            parts = CONTRASTIVE_REGEX.split(sent_str)
            if len(parts) > 1:
                # Matched contrastive connector
                matches = list(CONTRASTIVE_REGEX.finditer(sent_str))
                last_idx = 0
                for match in matches:
                    before_clause = sent_str[last_idx:match.start()].strip()
                    if before_clause:
                        clauses.append((before_clause, "normal"))
                    last_idx = match.end()
                after_clause = sent_str[last_idx:].strip()
                if after_clause:
                    clauses.append((after_clause, "contrasting"))
            else:
                clauses.append((sent_str, "normal"))

        return clauses if clauses else [(text, "normal")]

    def _score_clause_sentiment(self, clause: str, aspect_term: str) -> Tuple[str, float, float]:
        """
        Calculate sentiment polarity for a specific aspect in its surrounding clause context.
        Returns: (sentiment_label, confidence, polarity_score)
        """
        words = re.findall(r"\b[a-zA-Z']+\b", clause.lower())
        pos_score = 0.0
        neg_score = 0.0

        for i, word in enumerate(words):
            # Check for negation in preceding 3 tokens
            window_start = max(0, i - 3)
            preceding_tokens = words[window_start:i]
            is_negated = any(neg in preceding_tokens for neg in NEGATION_TERMS)

            # Check for intensifier
            is_intensified = any(inten in preceding_tokens for inten in INTENSIFIERS)
            weight = 1.5 if is_intensified else 1.0

            if word in POSITIVE_LEXICON:
                if is_negated:
                    neg_score += 1.2 * weight
                else:
                    pos_score += 1.0 * weight
            elif word in NEGATIVE_LEXICON:
                if is_negated:
                    pos_score += 1.0 * weight
                else:
                    neg_score += 1.2 * weight

        net_polarity = pos_score - neg_score
        total_signals = pos_score + neg_score

        if total_signals == 0:
            # Fallback to neutral
            return "neutral", 0.60, 0.0
        elif net_polarity > 0.2:
            confidence = min(0.99, round(0.70 + (pos_score / (total_signals + 1)) * 0.28, 4))
            return "positive", confidence, round(net_polarity, 2)
        elif net_polarity < -0.2:
            confidence = min(0.99, round(0.70 + (neg_score / (total_signals + 1)) * 0.28, 4))
            return "negative", confidence, round(net_polarity, 2)
        else:
            return "neutral", 0.65, 0.0

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform Aspect-Based Sentiment Analysis on input text.

        Args:
            text: Input social media post or product review.

        Returns:
            Structured dictionary with detected aspects, sentiments, confidence, and metadata.
        """
        if not text or not str(text).strip():
            return {
                "original_text": text,
                "aspects": [],
                "aspect_count": 0,
                "overall_sentiment": "neutral",
                "method": "Target-Entity Extraction & Clause Sentiment Decomposition",
                "status": "empty_input",
                "limitations": "Lexicon and dependency heuristic baseline; unlisted slang aspects require retraining.",
            }

        text_str = str(text).strip()
        clauses = self._split_into_clauses(text_str)

        detected_aspects = []
        seen_aspects = set()

        # Iterate through clauses to link aspects with their local context
        for clause_text, clause_type in clauses:
            # Find aspect matches in this clause
            matches = list(self.aspect_pattern.finditer(clause_text))
            for m in matches:
                aspect_term = m.group(0).lower().strip()
                if aspect_term in seen_aspects:
                    continue
                seen_aspects.add(aspect_term)

                sentiment, confidence, polarity_score = self._score_clause_sentiment(clause_text, aspect_term)

                detected_aspects.append({
                    "aspect": aspect_term,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "polarity_score": polarity_score,
                    "context_clause": clause_text,
                    "is_contrasting_clause": (clause_type == "contrasting"),
                })

        # Determine overall synthesized sentiment across detected aspects
        if not detected_aspects:
            overall_sentiment = "no_detectable_aspects"
        else:
            sentiments = [a["sentiment"] for a in detected_aspects]
            has_pos = "positive" in sentiments
            has_neg = "negative" in sentiments
            if has_pos and has_neg:
                overall_sentiment = "mixed"
            elif has_pos:
                overall_sentiment = "positive"
            elif has_neg:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

        return {
            "original_text": text_str,
            "aspects": detected_aspects,
            "aspect_count": len(detected_aspects),
            "overall_sentiment": overall_sentiment,
            "method": "Target-Entity Extraction & Clause Sentiment Decomposition",
            "model_version": "v1.0-clause-targeted",
            "status": "success",
            "limitations": (
                "Rule-and-lexicon guided aspect extraction handles common domain nouns and multi-word terms; "
                "novel out-of-vocabulary entities without clear sentiment polarity fall back to neutral."
            ),
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch analyze a collection of social media texts."""
        return [self.analyze(t) for t in texts]


# Module-level singleton
_ABSA_ANALYZER = None


def get_absa_analyzer() -> AspectBasedSentimentAnalyzer:
    """Retrieve singleton instance of AspectBasedSentimentAnalyzer."""
    global _ABSA_ANALYZER
    if _ABSA_ANALYZER is None:
        _ABSA_ANALYZER = AspectBasedSentimentAnalyzer()
    return _ABSA_ANALYZER


def analyze_aspect_sentiment(text: str) -> Dict[str, Any]:
    """Convenience helper to analyze aspect sentiment for a text string."""
    return get_absa_analyzer().analyze(text)
