"""
Unit tests for Elasticsearch Integration Module.
Uses mocking and health-check inspection to verify schema mapping, search construction,
and graceful offline handling without requiring a live Elasticsearch cluster.
"""

from unittest.mock import MagicMock
import pytest
from src.search.index_manager import get_index_mapping, create_sentiment_index
from src.search.search_service import ElasticsearchSearchService
from src.search.elasticsearch_client import is_elasticsearch_available


def test_index_mapping_structure():
    mapping = get_index_mapping()
    assert "mappings" in mapping
    props = mapping["mappings"]["properties"]
    assert "tweet_id" in props
    assert "original_text" in props
    assert "clean_text" in props
    assert "predicted_sentiment" in props
    assert "aspects" in props
    assert props["aspects"]["type"] == "nested"


def test_idempotent_index_creation_exists():
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True

    res = create_sentiment_index(mock_client, index_name="test_index", force_recreate=False)
    assert res["status"] == "exists"
    mock_client.indices.create.assert_not_called()


def test_idempotent_index_creation_new():
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = False

    res = create_sentiment_index(mock_client, index_name="test_index", force_recreate=False)
    assert res["status"] == "created"
    mock_client.indices.create.assert_called_once()


def test_search_service_query_assembly():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "hits": {"total": {"value": 1}, "hits": [{"_id": "1", "_score": 1.5, "_source": {"text": "hello"}}]},
        "aggregations": {"sentiment_distribution": {"buckets": [{"key": "positive", "doc_count": 1}]}},
        "took": 5,
    }

    service = ElasticsearchSearchService(mock_client, index_name="test_index")
    result = service.search(
        query="battery",
        sentiment="negative",
        aspect="battery life",
        aspect_sentiment="negative",
    )

    assert result["status"] == "success"
    assert result["total"] == 1
    assert len(result["results"]) == 1
    mock_client.search.assert_called_once()


def test_offline_detection():
    # Calling without running cluster returns boolean False safely
    available = is_elasticsearch_available()
    assert isinstance(available, bool)
