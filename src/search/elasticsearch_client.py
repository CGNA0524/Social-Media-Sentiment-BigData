"""
Elasticsearch Client Factory & Health Check Module.

Safely connects to a running Elasticsearch cluster using environment variables.
Provides connection pooling, timeout handling, authentication, and offline detection.
"""

import os
from typing import Any, Dict, Optional, Tuple
from elasticsearch import Elasticsearch

DEFAULT_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
DEFAULT_ES_USER = os.getenv("ELASTICSEARCH_USER", "")
DEFAULT_ES_PASS = os.getenv("ELASTICSEARCH_PASSWORD", "")
DEFAULT_ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")


def get_elasticsearch_client(
    url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout: float = 3.0,
) -> Elasticsearch:
    """
    Construct and return a configured Elasticsearch client instance.
    No credentials are hardcoded; all configuration is read from parameters or env variables.
    """
    es_url = url or os.getenv("ELASTICSEARCH_URL", DEFAULT_ES_URL)
    es_user = username if username is not None else os.getenv("ELASTICSEARCH_USER", DEFAULT_ES_USER)
    es_pass = password if password is not None else os.getenv("ELASTICSEARCH_PASSWORD", DEFAULT_ES_PASS)
    es_api_key = api_key if api_key is not None else os.getenv("ELASTICSEARCH_API_KEY", DEFAULT_ES_API_KEY)

    client_kwargs: Dict[str, Any] = {
        "hosts": [es_url],
        "request_timeout": request_timeout,
        "max_retries": 1,
    }

    if es_api_key:
        client_kwargs["api_key"] = es_api_key
    elif es_user and es_pass:
        client_kwargs["basic_auth"] = (es_user, es_pass)

    return Elasticsearch(**client_kwargs)


def check_elasticsearch_connection(client: Optional[Elasticsearch] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Ping the Elasticsearch cluster and retrieve cluster health metadata.

    Returns:
        (is_connected: bool, info_dict: dict)
    """
    es = client or get_elasticsearch_client()
    try:
        if es.ping():
            info = es.info()
            return True, {
                "status": "connected",
                "cluster_name": info.get("cluster_name", "unknown"),
                "version": info.get("version", {}).get("number", "unknown"),
                "lucene_version": info.get("version", {}).get("lucene_version", "unknown"),
                "url": DEFAULT_ES_URL,
            }
    except Exception as e:
        return False, {
            "status": "disconnected",
            "error": str(e),
            "url": DEFAULT_ES_URL,
            "hint": "Start Elasticsearch via Docker: 'docker run -d -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false elasticsearch:8.13.0'",
        }

    return False, {"status": "unreachable", "url": DEFAULT_ES_URL}


def is_elasticsearch_available() -> bool:
    """Return True only if a real Elasticsearch cluster responds to ping."""
    connected, _ = check_elasticsearch_connection()
    return connected
