#!/usr/bin/env python3
"""
Find minimum set of words to filter all transaction messages.
Uses greedy set cover algorithm with WHOLE WORD MATCHING (case insensitive).
"""

import json
import re
from collections import defaultdict
import argparse

def load_transaction_messages(jsonl_path: str) -> list[tuple[int, str]]:
    """Load 'relevant' and 'error' messages from JSONL file."""
    messages = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            classification = entry.get('output', {}).get('classification', '')
            # Include both 'relevant' and 'error' entries
            if classification in ('relevant', 'error'):
                messages.append((entry['index'], entry['source']['body']))
    return messages

def load_non_transaction_messages(csv_path: str) -> list[str]:
    """Load non-transaction messages from CSV for false positive analysis."""
    import csv
    messages = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row['body'])
    return messages

def tokenize(text: str, min_len: int = 2) -> set[str]:
    """Extract lowercase words from text."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return set(w for w in words if len(w) >= min_len)

def get_all_words(messages: list[tuple[int, str]], min_len: int = 2) -> set[str]:
    """Get all unique words across all messages."""
    all_words = set()
    for _, body in messages:
        all_words.update(tokenize(body, min_len))
    return all_words

def build_word_index(messages: list[tuple[int, str]], candidate_words: set[str]) -> dict[str, set[int]]:
    """Build word -> message indices using WHOLE WORD matching (case insensitive)."""
    index = defaultdict(set)
    for idx, (_, body) in enumerate(messages):
        body_words = set(re.findall(r'[a-zA-Z]+', body.lower()))
        for word in candidate_words:
            if word in body_words:  # Whole word match
                index[word].add(idx)
    return index

def greedy_set_cover(n_messages: int, index: dict[str, set[int]]) -> list[tuple[str, int]]:
    """Find minimum words using greedy set cover. Returns (word, coverage) pairs."""
    uncovered = set(range(n_messages))
    selected = []

    while uncovered:
        best_word = None
        best_coverage = 0

        for word, msg_indices in index.items():
            coverage = len(msg_indices & uncovered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_word = word

        if best_word is None or best_coverage == 0:
            print(f"Warning: {len(uncovered)} messages cannot be covered")
            break

        selected.append((best_word, best_coverage))
        uncovered -= index[best_word]
        print(f"Selected '{best_word}': covers {best_coverage}, {len(uncovered)} remaining")

    return selected

def verify_coverage(messages: list[tuple[int, str]], words: list[str]) -> tuple[int, list]:
    """Verify coverage using whole word matching and return uncovered messages."""
    uncovered = []
    words_set = set(words)
    for idx, (msg_id, body) in enumerate(messages):
        body_words = set(re.findall(r'[a-zA-Z]+', body.lower()))
        if not body_words.intersection(words_set):
            uncovered.append((msg_id, body))
    return len(messages) - len(uncovered), uncovered

def analyze_false_positives(non_tx_messages: list[str], words: list[str]) -> tuple[int, int]:
    """Count how many non-transaction messages would be matched (false positives)."""
    matched = 0
    words_set = set(words)
    for body in non_tx_messages:
        body_words = set(re.findall(r'[a-zA-Z]+', body.lower()))
        if body_words.intersection(words_set):
            matched += 1
    return matched, len(non_tx_messages)

def main():
    parser = argparse.ArgumentParser(description='Find minimum words to filter transaction messages')
    parser.add_argument('--min-len', type=int, default=2, help='Minimum word length (default: 2)')
    parser.add_argument('--analyze-fp', action='store_true', help='Analyze false positive rate')
    args = parser.parse_args()

    jsonl_path = "dataset-creation/output/transaction_messages.output.final.jsonl"
    non_tx_path = "dataset-creation/input/non_transaction_messages.csv"

    print(f"Configuration: min_word_length={args.min_len}")
    print("="*60)

    print("\nLoading transaction messages (relevant + error)...")
    messages = load_transaction_messages(jsonl_path)
    print(f"Loaded {len(messages)} messages to cover")

    print(f"\nExtracting all unique words (min length: {args.min_len})...")
    all_words = get_all_words(messages, args.min_len)
    print(f"Found {len(all_words)} unique words")

    print("\nBuilding word index (whole word matching)...")
    index = build_word_index(messages, all_words)

    print("\nRunning greedy set cover algorithm...")
    selected = greedy_set_cover(len(messages), index)

    words_only = [w for w, _ in selected]
    print(f"\n{'='*60}")
    print(f"RESULT: {len(selected)} words needed (min_len={args.min_len})")
    print(f"{'='*60}")
    print("Words:", words_only)

    # Verify
    covered, uncovered = verify_coverage(messages, words_only)
    print(f"\nVerification: {covered}/{len(messages)} covered ({100*covered/len(messages):.2f}%)")

    if uncovered:
        print(f"\nUncovered messages ({len(uncovered)}):")
        for msg_id, body in uncovered[:5]:
            print(f"  [{msg_id}] {body[:80]}...")

    # False positive analysis
    if args.analyze_fp:
        print("\n" + "="*60)
        print("FALSE POSITIVE ANALYSIS")
        print("="*60)
        non_tx = load_non_transaction_messages(non_tx_path)
        matched, total = analyze_false_positives(non_tx, words_only)
        print(f"Non-transaction messages matched: {matched}/{total} ({100*matched/total:.2f}%)")
        print(f"True negatives (correctly filtered out): {total - matched}/{total}")

if __name__ == "__main__":
    main()
