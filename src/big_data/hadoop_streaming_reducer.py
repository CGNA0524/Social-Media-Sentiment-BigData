#!/usr/bin/env python3
"""
Hadoop Streaming Reducer for Social Media Sentiment Word Frequency.

Reads sorted (key, count) pairs from STDIN and aggregates frequency per key:
Emits:
<word>\t<total_count>
"""

import sys

current_word = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split("\t")
    if len(parts) != 2:
        continue

    word, count_str = parts
    try:
        count = int(count_str)
    except ValueError:
        continue

    if current_word == word:
        current_count += count
    else:
        if current_word is not None:
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

if current_word is not None:
    print(f"{current_word}\t{current_count}")
