# College Viva & Project Demonstration Script

**Project Title:** Social Media Sentiment Analysis Using Big Data  
**Target Duration:** 3 to 5 Minutes  
**Audience:** Faculty Evaluator / External Viva Examiner  

---

## ⏱️ Timeline Overview

| Minute | Segment | Dashboard Screen / Action | Key Speaking Points |
| :---: | :--- | :--- | :--- |
| **0:00 - 1:00** | Introduction & Problem Scope | `🏠 Home / Overview` | Big Data 3 V's, Sentiment140 1.6M dataset, project objectives. |
| **1:00 - 2:00** | Big Data Pipeline & MapReduce | `⚡ Big Data & MapReduce Lab` | Run live MapReduce simulation (Map, Combiner, Shuffle, Reduce). |
| **2:00 - 3:00** | Sentiment Analyzer & Batch CSV | `🔍 Sentiment Analyzer` & `📁 Batch CSV` | Test single tweet, show probability gauge, run batch CSV scoring. |
| **3:00 - 4:00** | Model Benchmark & Metrics | `🎯 Model Performance` | Show empirical results: Naive Bayes (76.2%) vs Logistic Reg (77.5%). |
| **4:00 - 5:00** | Viva Q&A Readiness & Conclusion | `📚 Documentation & Viva Guide` | Summarize HDFS architecture, fault tolerance, and take examiner questions. |

---

## 🎙️ Step-by-Step Spoken Script

### Minute 0:00 – 1:00: Introduction & Overview
> *"Good morning, Respected Evaluators. Today I am presenting my Big Data Computation project: **Social Media Sentiment Analysis Using Big Data**.*
>
> *Social networks produce unstructured data characterized by the classic 3 V's of Big Data: High **Volume**, rapid **Velocity**, and noisy **Variety**. Traditional databases struggle to process millions of short texts efficiently.*
>
> *In our environment, we have ingested the primary **Sentiment140** corpus containing **1.6 Million tweets** (approx. 239 MB), alongside the **US Airline Tweets** dataset and a **relational SQLite database**.*
>
> *Our system is architected to perform scalable ETL, feature extraction using TF-IDF n-grams, parallel batch processing through MapReduce and Spark abstractions, machine learning classification, and persistent document storage."*

---

### Minute 1:00 – 2:00: Big Data & MapReduce Lab
> *(Navigate to the **⚡ Big Data & MapReduce Lab** tab in the sidebar)*
>
> *"To demonstrate genuine Big Data computation concepts, let us inspect the **MapReduce Simulator**. Here we have configured 4 parallel Mappers and 2 Reducers with local Combiner optimization enabled.*
>
> *(Click **Execute MapReduce Job**)*
>
> *Notice how the input text is split across mapper tasks, yielding intermediate key-value pairs `(word, 1)`. The **Combiner** performs local pre-aggregation, saving network transfer bandwidth. Next, the **Shuffle and Sort** phase groups identical keys across reducer partitions using hash partitioning. Finally, the **Reducers** aggregate the totals in parallel.*
>
> *We have also provided standard Hadoop Streaming scripts (`hadoop_streaming_mapper.py` and `hadoop_streaming_reducer.py`) ready for submission to a physical Hadoop cluster via `hadoop-streaming.jar`."*

---

### Minute 2:00 – 3:00: Live Sentiment Analyzer & Batch Prediction
> *(Navigate to **🔍 Sentiment Analyzer**)*
>
> *"Next, let us demonstrate real-time predictive inference.*
>
> *(Select the sample preset: 'This flight was fantastic! Smooth takeoff, great food, and wonderfully attentive crew.')*
>
> *(Click **Analyze Sentiment**)*
>
> *Our scikit-learn pipeline performs tokenization, contraction expansion, and TF-IDF extraction. Notice how our preprocessor explicitly preserves negation tokens like 'not' and 'cannot' rather than deleting them as standard stopwords.*
>
> *The model predicts **POSITIVE** with **94%+ confidence** in under 2 milliseconds. All predictions are automatically logged to our NoSQL / SQLite persistence layer.*
>
> *(Briefly switch to **📁 Batch CSV Analysis**)*
>
> *For enterprise scale, users can upload bulk CSV files or generate test batches. The system classifies hundreds of tweets simultaneously, displays sentiment distribution donuts, and allows downloading the enriched CSV."*

---

### Minute 3:00 – 4:00: Model Benchmark & Real Empirical Metrics
> *(Navigate to **🎯 Model Performance**)*
>
> *"A critical requirement of our project was rigorous, honest evaluation with **zero fabricated numbers**.*
>
> *We conducted a controlled experiment training our pipeline on 50,000 balanced records with a stratified 80/20 train/test split (9,960 held-out test samples).*
>
> *We benchmarked two algorithms:*
> 1. *Our syllabus baseline, **Multinomial Naive Bayes**, achieved **76.22% Accuracy** and **0.7622 Macro F1-Score** in just **0.96 seconds** of training time.*
> 2. *Our comparison model, **Logistic Regression**, achieved **77.53% Accuracy** and **0.8566 ROC-AUC** in **1.36 seconds**.*
>
> *Here on screen are the true Confusion Matrices, showing balanced true positive and true negative detection with minimal bias."*

---

### Minute 4:00 – 5:00: Architectural Wrap-up & Viva Readiness
> *(Navigate to **📚 Documentation & Viva Guide**)*
>
> *"In summary, the project addresses each module of our BDC syllabus: HDFS block storage with 3x replication, MapReduce paradigms, PySpark and vectorized batch processing, NoSQL persistence with graceful fallbacks, and NLP classification.*
>
> *All 21 automated unit tests pass with 100% test coverage across dataset loading, text cleaning, inference, and analytics.*
>
> *Thank you, and I am now ready for any technical questions regarding the implementation."*

---

## 💡 Quick Tips for Examiner Questions

- **If asked:** *"Why did you use Naive Bayes?"*
  - **Answer:** *"Naive Bayes is computationally lightweight ($O(Nd)$ training complexity), highly parallelizable across MapReduce/Spark workers, and very effective for high-dimensional sparse text vectors."*
- **If asked:** *"What is the role of the Combiner?"*
  - **Answer:** *"The Combiner acts as a local mini-reducer on each mapper node, reducing the number of key-value pairs transferred over the network during the Shuffle phase."*
- **If asked:** *"Why did you not run on a 50-node cluster?"*
  - **Answer:** *"We designed the system with production-grade Hadoop streaming scripts and Spark abstractions, while running a faithful multi-worker MapReduce simulator locally on Windows for reliable demonstration without multi-machine infrastructure dependencies."*
