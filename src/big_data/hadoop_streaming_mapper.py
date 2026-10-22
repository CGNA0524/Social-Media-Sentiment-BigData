#!/usr/bin/env python3
"""
Hadoop Streaming Mapper for Social Media Sentiment Word Frequency.

Reads raw tweet lines from STDIN, cleans tokens, and emits tab-separated:
<word>\t1

Can be tested locally via:
    cat tweets.txt | python hadoop_streaming_mapper.py | sort | python hadoop_streaming_reducer.py
Or executed on a Hadoop Cluster via:
    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
      -input /user/data/tweets.csv \
      -output /user/data/wordcount_out \
      -mapper hadoop_streaming_mapper.py \
      -reducer hadoop_streaming_reducer.py
"""

import sys
import re

ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # Remove non-alphabetical characters and lowercase
    cleaned = ALPHA_PATTERN.sub(" ", line).lower()
    words = cleaned.split()

    for word in words:
        if len(word) > 2:
            print(f"{word}\t1")
