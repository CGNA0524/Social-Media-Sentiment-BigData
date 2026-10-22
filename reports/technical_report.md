# Technical Project Report: Social Media Sentiment Analysis Using Big Data

**Course:** Big Data Computation (BDC)  
**Academic Degree:** Bachelor of Technology (B.Tech)  
**System Implementation:** Python 3.11, Scikit-Learn, Streamlit, MapReduce Architecture  
**Date of Evaluation:** October 2026  

---

## 1. Abstract

With the explosive growth of modern social networks, massive streams of user-generated text are published every second. Capturing, preprocessing, and extracting actionable intelligence from this unstructured data represents a hallmark challenge in Big Data Computation (BDC). This project presents an end-to-end sentiment classification system that scales to 1.6 Million real-world social media posts (the Sentiment140 corpus) along with multi-class domain datasets (US Airline Sentiment and SQLite storage). 

The computational pipeline integrates raw data ingestion, schema normalization, custom text preprocessing (negation preservation, contraction expansion, regex token cleaning), an in-memory multi-worker MapReduce aggregation engine, and a TF-IDF n-gram feature extraction pipeline. Classification is benchmarked between **Multinomial Naive Bayes** (the curriculum baseline) and **Logistic Regression**. On a held-out stratified test set of 9,960 samples, Naive Bayes achieved **76.22% accuracy (F1: 0.7622)** in 0.96 seconds of training time, while Logistic Regression achieved **77.53% accuracy (F1: 0.7753)** in 1.36 seconds with an ROC-AUC of **0.8566**. The system features an interactive, dark-themed Streamlit dashboard providing single-tweet inference with confidence gauges, bulk CSV prediction, MapReduce step-by-step simulations, and persistent storage with MongoDB and SQLite fallbacks.

---

## 2. Introduction & Background

Social media platforms like Twitter (X) are prime examples of the **3 V's of Big Data**:
- **Volume:** Millions of user status updates created continuously.
- **Velocity:** High-frequency incoming data streams requiring low-latency sentiment scoring.
- **Variety:** Unstructured text littered with conversational contractions, hashtags, @mentions, URLs, spelling variations, and emojis.

Traditional relational database management systems (RDBMS) degrade significantly when asked to execute full-text scans or maintain vocabulary frequency indices over millions of rows. Big Data architectures address this through distributed file storage (Hadoop HDFS) and parallel computing frameworks (Hadoop MapReduce and Apache Spark).

This project demonstrates how core BDC principles translate into a demonstrable, high-performance software system on modern compute platforms.

---

## 3. Related Work Placeholder

> *[Student Note for Faculty/Viva: Insert 3–4 literature citations here during final project submission, e.g., Go, Bhayani, and Huang (2009) on Sentiment140 distant supervision; Pang and Lee (2008) on opinion mining; and White (2012) on Hadoop: The Definitive Guide.]*

---

## 4. Problem Definition & Objectives

### Problem Definition
Given a raw, noisy social media text stream $T = \{t_1, t_2, \dots, t_N\}$, construct an automated computational pipeline capable of ingesting the corpus at scale, cleaning and vectorizing the text without information loss or test-set leakage, classifying sentiment into polarized classes $C \in \{\text{negative}, \text{positive}\}$, and aggregating metrics using parallel Big Data paradigms.

### Core Objectives
1. **Scalable Ingestion:** Ingest the 239 MB Sentiment140 dataset (1.6M rows) and normalize schema headers, encodings (`latin-1`), and target labels ($0 \to \text{negative}, 4 \to \text{positive}$).
2. **Text Normalization:** Strip URLs, user mentions, and punctuation while explicitly retaining sentiment-critical negations (`not`, `cannot`, `never`).
3. **Big Data Implementation:** Model the MapReduce computational cycle (Map, Combiner, Shuffle & Sort, Reduce) and provide standalone Hadoop Streaming scripts.
4. **Machine Learning Pipeline:** Benchmark Multinomial Naive Bayes and Logistic Regression using TF-IDF n-gram vectors.
5. **Interactive UI:** Deliver an interactive web dashboard for real-time inference, batch CSV scoring, and architecture visualization.
6. **Persistence & Telemetry:** Provide NoSQL (MongoDB) persistence with resilient local SQLite and JSON fallbacks.

---

## 5. System Architecture & Methodology

```mermaid
flowchart TD
    subgraph Storage Layer
        S1[Sentiment140 CSV - 239MB]
        S2[Airline Tweets CSV - 3.4MB]
        S3[database.sqlite - 5MB]
    end

    subgraph Ingestion & Preprocessing
        P1[load_dataset.py<br>Stratified Sampling & Validation]
        P2[preprocess.py<br>URLs, Mentions, Contractions, Negations]
    end

    subgraph Big Data Parallel Engine
        MR[MapReduce Simulator<br>Splits -> Map -> Combiner -> Shuffle -> Reduce]
        SP[Spark / Vectorized Engine<br>Batch Transformations & Telemetry]
    end

    subgraph Machine Learning Pipeline
        TF[TfidfVectorizer<br>Unigrams & Bigrams, sublinear TF]
        NB[Multinomial Naive Bayes<br>alpha=1.0 Baseline]
        LR[Logistic Regression<br>L-BFGS Comparison]
    end

    subgraph Storage & Presentation
        DB[(NoSQL MongoDB / SQLite)]
        UI[Streamlit Web Application<br>Single / Batch Inference & Analytics]
    end

    Storage Layer --> Ingestion & Preprocessing
    Ingestion & Preprocessing --> Big Data Parallel Engine
    Ingestion & Preprocessing --> Machine Learning Pipeline
    Machine Learning Pipeline --> DB
    Machine Learning Pipeline --> UI
    Big Data Parallel Engine --> UI
```

### 5.1 Data Preprocessing Pipeline
Social media text exhibits heavy noise. The preprocessing pipeline applies:
1. **HTML Entity Decoding:** Unescapes `&amp;`, `&quot;`, `&lt;`.
2. **Entity Stripping:** Removes URLs (`https?://\S+`) and user handles (`@\w+`).
3. **Hashtag Decomposition:** Extracts textual content (`#delayed` $\to$ `delayed`).
4. **Contraction Expansion:** Normalizes conversational abbreviations (`can't` $\to$ `cannot`, `won't` $\to$ `will not`).
5. **Negation Protection:** Unlike standard NLP stopword lists that drop `not`, `no`, and `never`, our stopword filter explicitly protects negation terms to preserve semantic valence.

### 5.2 TF-IDF Feature Extraction
Term Frequency-Inverse Document Frequency evaluates the relative importance of token $t$ in tweet $d$ within corpus $D$:

$$\text{TF}(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

We extract both unigrams and bigrams ($1, 2$) up to 25,000 features, applying sublinear term frequency scaling to attenuate the impact of repetitive tokens.

### 5.3 Classification Algorithms

#### Multinomial Naive Bayes
Applies Bayes' Theorem under the conditional independence assumption:

$$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^{V} P(x_i \mid C_k)$$

To handle vocabulary terms unseen during training, Laplace smoothing is applied:

$$\hat{P}(x_i \mid C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha V}$$

where $\alpha = 1.0$, $N_{ki}$ is the count of term $i$ in class $C_k$, and $V$ is the vocabulary size.

#### Logistic Regression
A linear model optimizing class posterior probabilities using the logistic sigmoid function:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Weights $\mathbf{w}$ are learned by minimizing the binary cross-entropy loss with $L_2$ regularization:

$$\mathcal{L}(\mathbf{w}) = -\sum_{j=1}^{M} \left[ y_j \log \hat{y}_j + (1 - y_j) \log(1 - \hat{y}_j) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

---

## 6. Big Data Modules

### 6.1 MapReduce Implementation
The project implements a parallel multi-worker MapReduce architecture in `src/big_data/mapreduce_simulator.py`:
- **Input Splitting:** Evenly splits input lines across $K$ worker partitions.
- **Map Function:** Emits intermediate `(key, value)` pairs (`(token, 1)` or `(sentiment, 1)`).
- **Combiner (Optimization):** Performs local in-memory reduction per mapper split, minimizing intermediate data volume.
- **Shuffle & Sort:** Employs hash partitioning (`abs(hash(key)) % num_reducers`) and sorts keys before delivery to reducers.
- **Reduce Function:** Aggregates lists of integers per key $\sum v_i$.
- **Hadoop Streaming:** Standalone Python scripts (`hadoop_streaming_mapper.py` and `hadoop_streaming_reducer.py`) allow native submission to Apache Hadoop via `hadoop-streaming.jar`.

### 6.2 Spark & Local Batch Engine
The `SparkEngineManager` module evaluates whether PySpark and a compatible JVM environment are present. If PySpark is active, it coordinates distributed DataFrame transformations; when executing in environments with incompatible Java versions (such as Java 24 on Windows lacking `winutils.exe`), it transparently transitions to a Vectorized Local Engine, openly reporting telemetry to prevent false claims of distributed execution.

### 6.3 NoSQL Persistence
The `NoSQLStorageManager` targets MongoDB for persisting prediction histories and batch summaries. If the MongoDB daemon is offline, it activates a local SQLite transactional store (`data/results/results_store.sqlite`) with zero downtime.

---

## 7. Experimental Setup & Results

### 7.1 Dataset Configuration
- **Corpus:** Sentiment140 (`training.1600000.processed.noemoticon.csv`)
- **Sample Size:** 50,000 balanced records (25,000 negative, 25,000 positive)
- **Usable Records After Cleaning:** 49,799
- **Train/Test Split:** Stratified 80% Train (39,839 records) / 20% Test (9,960 records)
- **Random Seed:** 42

### 7.2 Benchmark Comparison Table

| Metric | Multinomial Naive Bayes | Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | **76.22%** | **77.53%** |
| **Macro Precision** | 76.24% | 77.54% |
| **Macro Recall** | 76.22% | 77.53% |
| **Macro F1-Score** | 0.7622 | 0.7753 |
| **ROC-AUC** | 0.8433 | 0.8566 |
| **Training Time** | **0.96 seconds** | 1.36 seconds |
| **Inference Latency** | **< 1.5 ms / tweet** | < 1.8 ms / tweet |

### 7.3 Detailed Class Breakdown

#### Multinomial Naive Bayes:
- **Negative Class:** Precision = 0.7554, Recall = 0.7760, F1 = 0.7656
- **Positive Class:** Precision = 0.7695, Recall = 0.7484, F1 = 0.7588
- **Confusion Matrix:**
  $$\begin{pmatrix} 3867 & 1116 \\ 1252 & 3725 \end{pmatrix}$$
  - True Negatives: 3,867 | False Positives: 1,116
  - False Negatives: 1,252 | True Positives: 3,725

#### Logistic Regression:
- **Negative Class:** Precision = 0.7795, Recall = 0.7682, F1 = 0.7738
- **Positive Class:** Precision = 0.7712, Recall = 0.7824, F1 = 0.7768
- **Confusion Matrix:**
  $$\begin{pmatrix} 3828 & 1155 \\ 1083 & 3894 \end{pmatrix}$$
  - True Negatives: 3,828 | False Positives: 1,155
  - False Negatives: 1,083 | True Positives: 3,894

---

## 8. Limitations & Constraints

1. **Bag-of-Words Independence:** TF-IDF with Naive Bayes ignores deep syntactic dependencies and word ordering beyond 2-word n-grams.
2. **Sarcasm & Subtle Irony:** Tweets featuring contradictory sentiment (e.g., *"Oh wonderful, another flight delay!"*) remain challenging for linear n-gram classifiers.
3. **Environment Incompatibilities:** PySpark requires Java 8/11/17 and POSIX-compatible binaries; on Windows workstations running Java 24 without `winutils.exe`, local vectorized fallback is required.

---

## 9. Conclusion & Future Work

This project demonstrates the complete lifecycle of a Big Data NLP application, transitioning from raw 1.6M-record CSV ingestion to parallel MapReduce processing, machine learning classification, and an interactive presentation dashboard. Logistic Regression marginally outperformed Naive Bayes (77.53% vs. 76.22%), while Naive Bayes exhibited faster training throughput.

### Future Work
1. Deploying a live Apache Kafka streaming ingestion pipeline.
2. Integrating transformer-based language models (e.g., DistilBERT) using Spark NLP for contextual sentiment understanding.
3. Hosting the cluster on Google Cloud Dataproc or AWS EMR for multi-node distributed benchmarks.
