"""
Generate ground truth relevance labels for evaluation.
Uses keyword heuristics to auto-label passages, with support for manual review.
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOPICS, TOPIC_QUERIES, CORPUS_PATH, RELEVANCE_LABELS_PATH


def keyword_relevance(text, topic):
    """Score relevance of a passage to a topic using keyword matching.
    Returns 0 (not relevant), 1 (somewhat relevant), or 2 (highly relevant).
    """
    text_lower = text.lower()
    seeds = TOPIC_QUERIES[topic]["expansion_seeds"]
    primary_terms = TOPIC_QUERIES[topic]["primary"].lower().split()

    # Count how many seed terms appear
    matches = sum(1 for term in seeds if term.lower() in text_lower)
    primary_matches = sum(1 for term in primary_terms if term in text_lower)

    if matches >= 3 or (matches >= 1 and primary_matches >= 1):
        return 2  # Highly relevant
    elif matches >= 1 or primary_matches >= 1:
        return 1  # Somewhat relevant
    return 0  # Not relevant


def generate_labels():
    """Generate relevance labels for all passages using keyword heuristics."""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    labels = {}
    stats = {topic: {"highly": 0, "somewhat": 0} for topic in TOPICS}

    for passage in corpus:
        pid = passage["passage_id"]
        text = passage["text"]
        passage_labels = {}

        for topic in TOPICS:
            score = keyword_relevance(text, topic)
            if score > 0:
                passage_labels[topic] = score
                if score == 2:
                    stats[topic]["highly"] += 1
                else:
                    stats[topic]["somewhat"] += 1

        if passage_labels:
            labels[pid] = passage_labels

    # Save labels
    with open(RELEVANCE_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("Relevance label statistics:")
    print(f"{'Topic':<20} {'Highly (2)':<12} {'Somewhat (1)':<14} {'Total'}")
    print("-" * 60)
    for topic in TOPICS:
        h = stats[topic]["highly"]
        s = stats[topic]["somewhat"]
        print(f"{topic:<20} {h:<12} {s:<14} {h + s}")

    labeled_count = len(labels)
    total_count = len(corpus)
    print(f"\nLabeled {labeled_count}/{total_count} passages ({labeled_count/total_count*100:.1f}%)")
    print(f"Saved to {RELEVANCE_LABELS_PATH}")

    return labels


if __name__ == "__main__":
    generate_labels()
