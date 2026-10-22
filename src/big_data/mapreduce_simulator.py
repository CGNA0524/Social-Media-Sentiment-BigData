"""
MapReduce Simulator and Parallel Aggregation Engine for Big Data Sentiment Analysis.

Demonstrates the core MapReduce computational model:
1. Input Splitting: Partitioning the dataset across simulated worker nodes.
2. Mapping Phase: Emitting intermediate Key-Value pairs (k1, v1) -> list(k2, v2).
3. Combiner (Optimization): Local aggregation on mapper nodes to reduce shuffle overhead.
4. Shuffle & Sort: Grouping and sorting intermediate pairs by key across reducer partitions.
5. Reducing Phase: Aggregating grouped values into final output records (k2, list(v2)) -> (k2, v3).
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd


class MapReduceEngine:
    """
    In-memory, multi-worker MapReduce processing engine implementing
    the formal Hadoop MapReduce programming model.
    """

    def __init__(self, num_mappers: int = 4, num_reducers: int = 2, use_combiner: bool = True):
        self.num_mappers = max(1, num_mappers)
        self.num_reducers = max(1, num_reducers)
        self.use_combiner = use_combiner

    def split_input(self, data: List[Any]) -> List[List[Any]]:
        """Divide input records into balanced splits for mapper tasks."""
        if not data:
            return []
        chunk_size = (len(data) + self.num_mappers - 1) // self.num_mappers
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    def run_job(
        self,
        data: List[Any],
        mapper_fn: Callable[[Any], Iterator[Tuple[str, int]]],
        reducer_fn: Callable[[str, List[int]], Tuple[str, int]],
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end MapReduce job.

        Args:
            data: List of raw records (e.g., tweet text lines or (text, sentiment) tuples).
            mapper_fn: Function yielding (key, value) pairs from a single record.
            reducer_fn: Function reducing (key, [values]) to (key, final_value).

        Returns:
            Dictionary containing final results, intermediate stats, and performance timings.
        """
        start_time = time.time()
        splits = self.split_input(data)

        # 1. MAP PHASE
        map_start = time.time()
        mapper_outputs: List[List[Tuple[str, int]]] = []

        def execute_mapper(split_data: List[Any]) -> List[Tuple[str, int]]:
            pairs = []
            for record in split_data:
                for k, v in mapper_fn(record):
                    pairs.append((k, v))

            # Apply local Combiner if enabled
            if self.use_combiner:
                combined = defaultdict(int)
                for k, v in pairs:
                    combined[k] += v
                return list(combined.items())
            return pairs

        with ThreadPoolExecutor(max_workers=self.num_mappers) as executor:
            mapper_outputs = list(executor.map(execute_mapper, splits))

        map_time = time.time() - map_start
        total_intermediate_pairs = sum(len(mo) for mo in mapper_outputs)

        # 2. SHUFFLE & SORT PHASE
        shuffle_start = time.time()
        reducer_partitions: List[Dict[str, List[int]]] = [defaultdict(list) for _ in range(self.num_reducers)]

        for worker_pairs in mapper_outputs:
            for key, val in worker_pairs:
                partition_id = abs(hash(key)) % self.num_reducers
                reducer_partitions[partition_id][key].append(val)

        # Sort keys within each reducer partition
        sorted_partitions = []
        for part in reducer_partitions:
            sorted_keys = sorted(part.keys())
            sorted_partitions.append([(k, part[k]) for k in sorted_keys])

        shuffle_time = time.time() - shuffle_start

        # 3. REDUCE PHASE
        reduce_start = time.time()

        def execute_reducer(partition_items: List[Tuple[str, List[int]]]) -> List[Tuple[str, int]]:
            results = []
            for k, val_list in partition_items:
                results.append(reducer_fn(k, val_list))
            return results

        with ThreadPoolExecutor(max_workers=self.num_reducers) as executor:
            reducer_outputs = list(executor.map(execute_reducer, sorted_partitions))

        reduce_time = time.time() - reduce_start

        # Flatten output
        final_results = {}
        for ro in reducer_outputs:
            for k, v in ro:
                final_results[k] = v

        total_time = time.time() - start_time

        return {
            "total_input_records": len(data),
            "num_mappers": len(splits),
            "num_reducers": self.num_reducers,
            "use_combiner": self.use_combiner,
            "intermediate_pairs_count": total_intermediate_pairs,
            "unique_keys_emitted": len(final_results),
            "map_time_seconds": round(map_time, 4),
            "shuffle_sort_time_seconds": round(shuffle_time, 4),
            "reduce_time_seconds": round(reduce_time, 4),
            "total_time_seconds": round(total_time, 4),
            "results": final_results,
        }


# Standard MapReduce implementations for Social Media Sentiment

def sentiment_count_mapper(record: Tuple[str, str]) -> Iterator[Tuple[str, int]]:
    """
    Map record (text, sentiment_label) -> (sentiment_label, 1).
    """
    _, sentiment = record
    if sentiment:
        yield (str(sentiment).lower(), 1)


def word_count_mapper(record: str) -> Iterator[Tuple[str, int]]:
    """
    Map tweet text -> (token, 1) for all tokens of length > 2.
    """
    if isinstance(record, str):
        for token in record.lower().split():
            clean_tok = "".join(c for c in token if c.isalnum())
            if len(clean_tok) > 2:
                yield (clean_tok, 1)


def sentiment_word_mapper(record: Tuple[str, str]) -> Iterator[Tuple[str, int]]:
    """
    Map (text, sentiment) -> ('sentiment:token', 1).
    """
    text, sentiment = record
    if isinstance(text, str) and sentiment:
        for token in text.lower().split():
            clean_tok = "".join(c for c in token if c.isalnum())
            if len(clean_tok) > 2:
                yield (f"{sentiment.lower()}:{clean_tok}", 1)


def sum_reducer(key: str, values: List[int]) -> Tuple[str, int]:
    """Standard addition reducer aggregating counts for a specific key."""
    return (key, sum(values))
