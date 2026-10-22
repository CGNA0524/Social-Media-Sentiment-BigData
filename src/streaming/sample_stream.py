"""
Sample Social Media Tweet Stream Generator.

Simulates a continuous incoming stream of social media micro-posts with:
- Unique event tweet IDs
- Timestamps (ISO 8601 with millisecond precision)
- Varied sentiment categories (positive, negative, neutral, mixed, and aspect-rich)
- Realistic usernames and handles
"""

from datetime import datetime, timezone
import itertools
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional

# Realistic social media sample tweets for stream replay
SAMPLE_TWEET_STREAM_POOL = [
    {"user": "@tech_guru", "text": "The camera is excellent, but the battery life is disappointing."},
    {"user": "@traveler_jane", "text": "Loved the flight! Super friendly cabin crew and on-time arrival. #happy"},
    {"user": "@angry_customer", "text": "Worst customer support ever! Waited 45 mins and they hung up. Not acceptable!"},
    {"user": "@gadget_lover", "text": "The delivery was fast, and the product quality is excellent."},
    {"user": "@daily_commuter", "text": "Flight delay for 3 hours again... seat comfort is terrible and wifi broken."},
    {"user": "@foodie_dan", "text": "Delicious food on board today, very surprised by the quality and taste!"},
    {"user": "@mobile_tester", "text": "Screen resolution is stunning, but the phone gets really hot while charging."},
    {"user": "@eco_shopper", "text": "Great packaging and fast shipping speed. Will definitely order again."},
    {"user": "@random_poster", "text": "Just another regular Tuesday morning. Coffee in hand, heading to office."},
    {"user": "@support_needed", "text": "Website is broken, cannot access my account or download my ticket. Plz fix!"},
]


def create_stream_event(
    text: str,
    user: str = "@stream_user",
    tweet_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create a structured raw stream event message."""
    ts = timestamp or datetime.now(timezone.utc)
    return {
        "tweet_id": tweet_id or f"tweet_{uuid.uuid4().hex[:10]}",
        "user": user,
        "text": text,
        "timestamp": ts.isoformat(),
        "event_time_epoch_ms": int(ts.timestamp() * 1000),
    }


def get_sample_tweet_records(count: int = 10) -> List[Dict[str, Any]]:
    """Retrieve a fixed list of sample tweet events for deterministic pipeline testing."""
    records = []
    pool_len = len(SAMPLE_TWEET_STREAM_POOL)
    for i in range(count):
        item = SAMPLE_TWEET_STREAM_POOL[i % pool_len]
        rec = create_stream_event(
            text=item["text"],
            user=item["user"],
            tweet_id=f"stream_ev_{i+1:04d}",
        )
        records.append(rec)
    return records


def generate_tweet_stream(
    max_records: Optional[int] = None,
    delay_seconds: float = 0.5,
) -> Iterator[Dict[str, Any]]:
    """
    Generator that continuously yields streaming tweet records.

    Args:
        max_records: Maximum number of events to yield (None for infinite).
        delay_seconds: Pause between successive event emissions to simulate rate.
    """
    counter = 0
    pool_cycle = itertools.cycle(SAMPLE_TWEET_STREAM_POOL)

    for item in pool_cycle:
        if max_records is not None and counter >= max_records:
            break

        counter += 1
        event = create_stream_event(
            text=item["text"],
            user=item["user"],
            tweet_id=f"stream_ev_{counter:05d}",
        )
        yield event

        if delay_seconds > 0:
            time.sleep(delay_seconds)
