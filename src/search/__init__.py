"""
Elasticsearch Integration Package for Social Media Sentiment Analysis.
"""

from src.search.elasticsearch_client import get_elasticsearch_client, is_elasticsearch_available
from src.search.index_manager import create_sentiment_index, get_index_mapping, DEFAULT_INDEX_NAME
from src.search.search_service import ElasticsearchSearchService

__all__ = [
    "get_elasticsearch_client",
    "is_elasticsearch_available",
    "create_sentiment_index",
    "get_index_mapping",
    "DEFAULT_INDEX_NAME",
    "ElasticsearchSearchService",
]
