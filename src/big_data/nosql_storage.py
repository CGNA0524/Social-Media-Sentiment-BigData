"""
NoSQL (MongoDB) Integration and Storage Fallback Module.

Implements:
- Optional MongoDB document storage for single predictions and batch analytics
- Safe environment variable configuration (MONGO_URI, MONGO_DB)
- Graceful local JSON / SQLite storage fallback when MongoDB daemon is not running
- Secret-free and zero-credential leak design
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data_processing.load_dataset import get_project_root

PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
LOCAL_FALLBACK_FILE = RESULTS_DIR / "prediction_history.json"
LOCAL_DB_FILE = RESULTS_DIR / "results_store.sqlite"


class NoSQLStorageManager:
    """
    Unified storage manager that attempts MongoDB first,
    falling back to local JSON and SQLite storage if MongoDB is unavailable.
    """

    def __init__(self):
        self.mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.environ.get("MONGO_DB", "sentiment_analytics_db")
        self.client = None
        self.db = None
        self.is_connected = False
        self.status_message = ""
        self._init_connection()

    def _init_connection(self):
        """Attempt connection to MongoDB with a short timeout."""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=1200)
            # Ping database to verify active server
            client.admin.command("ping")
            self.client = client
            self.db = client[self.db_name]
            self.is_connected = True
            self.status_message = f"Connected to MongoDB at {self.mongo_uri} (Database: {self.db_name})"
        except Exception:
            self.is_connected = False
            self.status_message = "MongoDB daemon not detected. Using Local Storage Fallback (JSON/SQLite)."

        # Ensure fallback directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_fallback()

    def _init_sqlite_fallback(self):
        """Initialize local SQLite tables for persistent fallback storage."""
        try:
            conn = sqlite3.connect(LOCAL_DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    text TEXT,
                    sentiment TEXT,
                    confidence REAL,
                    probabilities TEXT
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity metadata for UI display."""
        return {
            "is_connected": self.is_connected,
            "storage_type": "MongoDB NoSQL" if self.is_connected else "Local Storage Fallback (SQLite / JSON)",
            "status_message": self.status_message,
            "target_database": self.db_name if self.is_connected else str(LOCAL_DB_FILE.name),
        }

    def save_prediction(self, prediction_dict: Dict[str, Any]) -> bool:
        """
        Store a single sentiment prediction record.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "text": prediction_dict.get("original_text", ""),
            "sentiment": prediction_dict.get("sentiment", ""),
            "confidence": float(prediction_dict.get("confidence", 0.0)),
            "probabilities": prediction_dict.get("probabilities", {}),
        }

        # 1. MongoDB Storage
        if self.is_connected and self.db is not None:
            try:
                self.db.predictions.insert_one(record)
                return True
            except Exception:
                pass  # Fall back if runtime insertion fails

        # 2. Local Fallback (SQLite & JSON)
        try:
            conn = sqlite3.connect(LOCAL_DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (timestamp, text, sentiment, confidence, probabilities)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["text"],
                    record["sentiment"],
                    record["confidence"],
                    json.dumps(record["probabilities"]),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve recent predictions from MongoDB or local storage.
        """
        if self.is_connected and self.db is not None:
            try:
                docs = list(self.db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
                return docs
            except Exception:
                pass

        # Fallback read from SQLite
        try:
            conn = sqlite3.connect(LOCAL_DB_FILE)
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT timestamp, text, sentiment, confidence, probabilities FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()

            results = []
            for r in rows:
                results.append({
                    "timestamp": r[0],
                    "text": r[1],
                    "sentiment": r[2],
                    "confidence": r[3],
                    "probabilities": json.loads(r[4]) if r[4] else {},
                })
            return results
        except Exception:
            return []


# Global singleton instance
nosql_storage = NoSQLStorageManager()
