# System Architecture Specification

## 1. Architectural Overview

The **Social Media Sentiment Analysis Using Big Data** system is structured around a decoupled, multi-tiered architecture designed to balance large-scale data ingestion, distributed/parallel batch computation, predictive modeling, and user presentation.

```text
+-------------------------------------------------------------------------+
|                          PRESENTATION TIER                              |
|   Streamlit UI (Dark Mode): Single Analyzer, Batch CSV, MapReduce Lab    |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           INFERENCE TIER                                |
|   Scikit-Learn Inference Pipeline (TF-IDF Vectorizer + MultinomialNB)   |
|   Confidence Probability Calibration & Decision Boundary Engine         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                         BIG DATA COMPUTE TIER                           |
|   - MapReduce Engine: Splits -> Mappers -> Combiner -> Shuffle -> Reducer|
|   - Spark Manager: PySpark DataFrame / Vectorized Local Batch Fallback   |
|   - Analytics Engine: N-Gram Mining, Temporal Aggregations              |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                         PREPROCESSING & ETL                             |
|   Regex Cleaning, HTML Unescape, Negation Preservation, Contractions     |
|   Stratified Sampling & Parquet Caching Layer                           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           PERSISTENCE TIER                              |
|   - Raw Storage: Sentiment140 (239MB), Tweets.csv, database.sqlite      |
|   - Processed Cache: Parquet Columnar Files                             |
|   - Operational DB: MongoDB NoSQL Document Store / SQLite Fallback      |
+-------------------------------------------------------------------------+
```

---

## 2. Component Responsibilities

| Module | Location | Primary Responsibilities |
| :--- | :--- | :--- |
| **Dataset Ingestion** | `src/data_processing/load_dataset.py` | Headerless CSV parsing (`latin-1`), target label normalization ($0 \to \text{neg}, 4 \to \text{pos}$), stratified sampling, parquet caching. |
| **Text Preprocessor** | `src/data_processing/preprocess.py` | Strip handles and links, unpack hashtags, expand contractions, retain negation tokens (`not`, `never`), optional stemming. |
| **Model Pipeline** | `src/models/train.py`, `evaluate.py`, `predict.py` | Fit `Pipeline(TF-IDF + MNB/LR)`, compute accuracy/F1/confusion matrix, serialize artifacts with `joblib`. |
| **MapReduce Simulator** | `src/big_data/mapreduce_simulator.py` | Demonstrates input splitting, multi-worker mapper execution, combiner optimization, hash-based shuffle & sort, and reduce aggregation. |
| **Hadoop Streaming** | `src/big_data/hadoop_streaming_*.py` | Standard Unix STDIN/STDOUT mapper and reducer scripts for physical Hadoop clusters. |
| **Spark Manager** | `src/big_data/spark_engine.py` | Coordinates distributed Spark processing or manages transparent vectorized local fallback with runtime telemetry. |
| **NoSQL Manager** | `src/big_data/nosql_storage.py` | Connects to MongoDB, falls back to SQLite/JSON, persists user predictions and batch query logs. |
| **Visualization Hub** | `src/visualization/charts.py` | Renders dark-themed donut charts, horizontal n-gram bars, confusion matrices, and timeline dynamics. |
| **Web Dashboard** | `app.py` | 8-page Streamlit interactive application. |

---

## 3. Storage Design

### 3.1 Raw Datasets
1. **Sentiment140 (`training.1600000.processed.noemoticon.csv`):**
   - Size: 238.8 MB
   - 1,600,000 rows, 6 columns, no header, `latin-1` encoded.
   - Binary classes: 800,000 negative (0), 800,000 positive (4).
2. **US Airline Tweets (`Tweets.csv`):**
   - Size: 3.4 MB
   - 14,640 rows, 15 columns with header.
   - 3 classes: negative (9,178), neutral (3,099), positive (2,363).
3. **Relational Database (`database.sqlite`):**
   - Size: 5.0 MB
   - Contains `Tweets` table (14,485 rows) accessed in read-only URI mode.

### 3.2 HDFS Data Layout (Cluster Deployment Design)
When deployed to a genuine Hadoop cluster:
```text
/user/hadoop/sentiment_project/
├── raw/
│   └── training.1600000.processed.noemoticon.csv   (Block Size: 128MB, 2 Blocks, Replication: 3)
├── processed/
│   └── clean_tweets.parquet
└── output/
    └── wordcount_results/
        ├── _SUCCESS
        ├── part-00000
        └── part-00001
```

### 3.3 NoSQL Schema Design (MongoDB)
Database: `sentiment_analytics_db`  
Collection: `predictions`
```json
{
  "_id": "ObjectId('67112470a1b2c3d4e5f67890')",
  "timestamp": "2026-10-17T18:50:54.515998",
  "text": "Antigravity big data project is working exceptionally well!",
  "sentiment": "positive",
  "confidence": 0.892,
  "probabilities": {
    "negative": 0.108,
    "positive": 0.892
  }
}
```

---

## 4. Local vs. Distributed Execution Tradeoffs

| Factor | Local Vectorized Mode (Active Environment) | Distributed Mode (Hadoop / Spark) |
| :--- | :--- | :--- |
| **Target Scale** | Laptop memory ($< 16$ GB RAM) | Multi-node cluster (TB / PB Scale) |
| **Fault Tolerance** | Single point of process failure | Automated block replication and task re-execution |
| **Network Overhead** | Zero network latency (shared memory) | Shuffle data transfers across cluster network |
| **Complexity** | Simple setup, no external daemons | Requires ZooKeeper, NameNode, ResourceManagers |
| **When to Use** | Rapid development, testing, college demos | Production web-scale data pipelines |
