"""
Elasticsearch Index Schema and Lifecycle Management Module.

Defines:
- Standard schema mapping for social media sentiment records
- Idempotent index creation (preserves existing data upon rerun)
- Nested mapping for Aspect-Based Sentiment Analysis (ABSA) results
"""

import os
from typing import Any, Dict, Optional
from elasticsearch import Elasticsearch

DEFAULT_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "social_media_sentiment")

# Strict schema mapping for social media records
SENTIMENT_INDEX_MAPPING: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "tweet_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "tweet_id": {"type": "keyword"},
            "user": {"type": "keyword"},
            "original_text": {
                "type": "text",
                "analyzer": "tweet_text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "clean_text": {
                "type": "text",
                "analyzer": "tweet_text_analyzer",
            },
            "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss"},
            "predicted_sentiment": {"type": "keyword"},
            "model_name": {"type": "keyword"},
            "confidence": {"type": "float"},
            "source_dataset": {"type": "keyword"},
            # Nested ABSA entities
            "aspects": {
                "type": "nested",
                "properties": {
                    "aspect": {"type": "keyword"},
                    "sentiment": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "polarity_score": {"type": "float"},
                    "context_clause": {"type": "text"},
                },
            },
            "indexed_at": {"type": "date"},
        }
    },
}


def get_index_mapping() -> Dict[str, Any]:
    """Retrieve the standard Elasticsearch index definition mapping."""
    return SENTIMENT_INDEX_MAPPING


def create_sentiment_index(
    client: Elasticsearch,
    index_name: str = DEFAULT_INDEX_NAME,
    force_recreate: bool = False,
) -> Dict[str, Any]:
    """
    Idempotently initialize the social media sentiment index.
    If the index already exists, it is preserved unless force_recreate is explicitly True.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the target index.
        force_recreate: If True, drops and rebuilds index (WARNING: destructive).

    Returns:
        Status dictionary with outcome metadata.
    """
    try:
        exists = client.indices.exists(index=index_name)
        if exists:
            if force_recreate:
                client.indices.delete(index=index_name)
                client.indices.create(index=index_name, body=SENTIMENT_INDEX_MAPPING)
                return {"status": "recreated", "index": index_name, "message": "Existing index deleted and recreated."}
            return {"status": "exists", "index": index_name, "message": "Index already exists. Preserved existing data."}

        client.indices.create(index=index_name, body=SENTIMENT_INDEX_MAPPING)
        return {"status": "created", "index": index_name, "message": f"Successfully created index '{index_name}'."}
    except Exception as e:
        return {"status": "error", "index": index_name, "error": str(e)}
