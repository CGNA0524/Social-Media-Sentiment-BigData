"""
Elasticsearch Search and Ingestion Service.

Provides:
- Individual record indexing
- Bulk batched indexing using elasticsearch.helpers.bulk
- Full-text keyword search across original and cleaned text
- Multi-dimensional filtering:
  - Sentiment filter ('positive', 'negative', 'neutral')
  - Temporal range filter ('start_time', 'end_time')
  - Nested aspect-level filter ('aspect', 'aspect_sentiment')
- Offset pagination and aggregation statistics
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
from elasticsearch import Elasticsearch, helpers

from src.search.elasticsearch_client import get_elasticsearch_client
from src.search.index_manager import DEFAULT_INDEX_NAME, create_sentiment_index


class ElasticsearchSearchService:
    """
    Search and Indexing Service interacting with Elasticsearch cluster.
    """

    def __init__(self, client: Optional[Elasticsearch] = None, index_name: str = DEFAULT_INDEX_NAME):
        self.client = client or get_elasticsearch_client()
        self.index_name = index_name

    def ensure_index(self) -> Dict[str, Any]:
        """Ensure target index exists before indexing."""
        return create_sentiment_index(self.client, index_name=self.index_name)

    def index_tweet(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index an individual social media record into Elasticsearch.
        """
        doc = dict(record)
        doc.setdefault("indexed_at", datetime.now(timezone.utc).isoformat())

        tweet_id = str(doc.get("tweet_id", doc.get("id", "")))
        try:
            res = self.client.index(
                index=self.index_name,
                id=tweet_id if tweet_id else None,
                document=doc,
                refresh=True,
            )
            return {"status": "success", "id": res.get("_id"), "result": res.get("result")}
        except Exception as e:
            return {"status": "error", "error": str(e), "record": tweet_id}

    def index_batch(self, records: List[Dict[str, Any]], chunk_size: int = 500) -> Dict[str, Any]:
        """
        Bulk index a collection of social media records.
        """
        if not records:
            return {"status": "success", "indexed_count": 0, "errors": []}

        actions = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for rec in records:
            doc = dict(rec)
            doc.setdefault("indexed_at", now_iso)
            t_id = str(doc.get("tweet_id", doc.get("id", "")))

            action = {
                "_index": self.index_name,
                "_source": doc,
            }
            if t_id:
                action["_id"] = t_id
            actions.append(action)

        try:
            success_count, errors = helpers.bulk(
                self.client,
                actions,
                chunk_size=chunk_size,
                raise_on_error=False,
                refresh=True,
            )
            return {
                "status": "success",
                "indexed_count": success_count,
                "error_count": len(errors) if isinstance(errors, list) else 0,
                "errors": errors if isinstance(errors, list) else [],
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "indexed_count": 0}

    def search(
        self,
        query: Optional[str] = None,
        sentiment: Optional[str] = None,
        aspect: Optional[str] = None,
        aspect_sentiment: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Search social media records with optional full-text query and filters.

        Args:
            query: Free-text search term.
            sentiment: Optional filter by predicted sentiment ('positive', 'negative', 'neutral').
            aspect: Optional filter by mentioned aspect term (e.g., 'camera', 'flight', 'battery').
            aspect_sentiment: Optional sentiment towards the filtered aspect.
            start_time: Lower bound ISO timestamp.
            end_time: Upper bound ISO timestamp.
            page: 1-indexed page number.
            page_size: Number of items per page.

        Returns:
            Dictionary with matching hits, total count, page info, and latency.
        """
        must_clauses: List[Dict[str, Any]] = []
        filter_clauses: List[Dict[str, Any]] = []

        # 1. Text Query
        if query and query.strip():
            must_clauses.append({
                "multi_match": {
                    "query": query.strip(),
                    "fields": ["original_text^2", "clean_text"],
                    "fuzziness": "AUTO",
                }
            })
        else:
            must_clauses.append({"match_all": {}})

        # 2. Overall Sentiment Filter
        if sentiment and sentiment.strip():
            filter_clauses.append({"term": {"predicted_sentiment": sentiment.lower().strip()}})

        # 3. Temporal Range Filter
        if start_time or end_time:
            range_filter: Dict[str, Any] = {}
            if start_time:
                range_filter["gte"] = start_time
            if end_time:
                range_filter["lte"] = end_time
            filter_clauses.append({"range": {"timestamp": range_filter}})

        # 4. Nested Aspect-Based Filter
        if aspect and aspect.strip():
            nested_must: List[Dict[str, Any]] = [
                {"term": {"aspects.aspect": aspect.lower().strip()}}
            ]
            if aspect_sentiment and aspect_sentiment.strip():
                nested_must.append({"term": {"aspects.sentiment": aspect_sentiment.lower().strip()}})

            filter_clauses.append({
                "nested": {
                    "path": "aspects",
                    "query": {"bool": {"must": nested_must}},
                }
            })

        # Assemble full search payload
        body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            },
            "from": max(0, (page - 1) * page_size),
            "size": max(1, min(page_size, 100)),
            "sort": [{"_score": {"order": "desc"}}, {"indexed_at": {"order": "desc", "unmapped_type": "date"}}],
            "aggs": {
                "sentiment_distribution": {
                    "terms": {"field": "predicted_sentiment"}
                }
            }
        }

        try:
            res = self.client.search(index=self.index_name, body=body)
            hits_data = res.get("hits", {})
            total_hits = hits_data.get("total", {}).get("value", 0)
            hits = [
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    **hit.get("_source", {}),
                }
                for hit in hits_data.get("hits", [])
            ]

            aggs = res.get("aggregations", {})
            sentiment_dist = {
                bucket["key"]: bucket["doc_count"]
                for bucket in aggs.get("sentiment_distribution", {}).get("buckets", [])
            }

            return {
                "status": "success",
                "total": total_hits,
                "page": page,
                "page_size": page_size,
                "results": hits,
                "aggregations": {"sentiment_distribution": sentiment_dist},
                "took_ms": res.get("took", 0),
            }
        except Exception as e:
            return {
                "status": "error",
                "total": 0,
                "page": page,
                "page_size": page_size,
                "results": [],
                "error": str(e),
            }
