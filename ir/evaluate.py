"""
Evaluation metrics for information retrieval: Precision@K and Mean Average Precision.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOPICS, RELEVANCE_LABELS_PATH


def load_relevance_labels(path=RELEVANCE_LABELS_PATH):
    """Load ground truth relevance labels."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_relevant_ids(labels, topic, min_score=1):
    """Get set of passage IDs relevant to a topic."""
    return {pid for pid, topic_scores in labels.items()
            if topic_scores.get(topic, 0) >= min_score}


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for pid in top_k if pid in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant documents found in top-k."""
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    relevant_in_top_k = sum(1 for pid in top_k if pid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def average_precision(retrieved_ids, relevant_ids):
    """Average precision for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, pid in enumerate(retrieved_ids):
        if pid in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_ids)


def evaluate_retriever(retriever, labels, topics=TOPICS, k_values=[5, 10]):
    """Run full evaluation of a retriever across all topics."""
    results = {}
    for topic in topics:
        relevant_ids = get_relevant_ids(labels, topic)
        retrieved = retriever.retrieve_all_ranked(topic)
        retrieved_ids = [p["passage_id"] for p in retrieved]

        topic_results = {}
        for k in k_values:
            topic_results[f"P@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            topic_results[f"R@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        topic_results["AP"] = average_precision(retrieved_ids, relevant_ids)
        topic_results["num_relevant"] = len(relevant_ids)

        results[topic] = topic_results

    # Compute MAP
    aps = [results[t]["AP"] for t in topics]
    results["MAP"] = sum(aps) / len(aps) if aps else 0.0

    # Compute average P@K
    for k in k_values:
        vals = [results[t][f"P@{k}"] for t in topics]
        results[f"avg_P@{k}"] = sum(vals) / len(vals) if vals else 0.0

    return results
