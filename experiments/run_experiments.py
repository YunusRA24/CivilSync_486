"""
Run full evaluation experiments comparing TF-IDF, embeddings, and embeddings + query expansion.
Outputs metrics tables and charts for the report.
"""
import json
import os
import csv
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOPICS, TOPIC_DISPLAY_NAMES, RESULTS_DIR
from ir.evaluate import evaluate_retriever, load_relevance_labels
from ir.tfidf_retriever import TFIDFRetriever
from ir.embedding_retriever import EmbeddingRetriever

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_all_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = load_relevance_labels()

    # Initialize retrievers
    print("Initializing TF-IDF retriever...")
    t0 = time.time()
    tfidf = TFIDFRetriever()
    tfidf_init_time = time.time() - t0

    print("Initializing Embedding retriever (no expansion)...")
    t0 = time.time()
    emb = EmbeddingRetriever(use_query_expansion=False)
    emb_init_time = time.time() - t0

    print("Initializing Embedding retriever (with expansion)...")
    t0 = time.time()
    emb_qe = EmbeddingRetriever(use_query_expansion=True)
    emb_qe_init_time = time.time() - t0

    methods = {
        "TF-IDF": (tfidf, tfidf_init_time),
        "SBERT": (emb, emb_init_time),
        "SBERT+QE": (emb_qe, emb_qe_init_time),
    }

    # Run evaluations
    all_results = {}
    for name, (retriever, init_time) in methods.items():
        print(f"\nEvaluating {name}...")
        t0 = time.time()
        results = evaluate_retriever(retriever, labels)
        query_time = time.time() - t0
        results["init_time_s"] = init_time
        results["query_time_s"] = query_time
        all_results[name] = results

    # Print results table
    print("\n" + "=" * 90)
    print(f"{'Topic':<20} {'Metric':<10} {'TF-IDF':<12} {'SBERT':<12} {'SBERT+QE':<12}")
    print("=" * 90)

    for topic in TOPICS:
        display = TOPIC_DISPLAY_NAMES.get(topic, topic)
        for metric in ["P@5", "P@10", "AP"]:
            vals = [f"{all_results[m][topic][metric]:.3f}" for m in methods]
            label = display if metric == "P@5" else ""
            print(f"{label:<20} {metric:<10} {'  '.join(f'{v:<12}' for v in vals)}")
        print("-" * 90)

    print(f"\n{'MAP':<30} " + "  ".join(
        f"{all_results[m]['MAP']:.3f}       " for m in methods))
    print(f"{'Avg P@5':<30} " + "  ".join(
        f"{all_results[m]['avg_P@5']:.3f}       " for m in methods))
    print(f"{'Avg P@10':<30} " + "  ".join(
        f"{all_results[m]['avg_P@10']:.3f}       " for m in methods))

    print(f"\n{'Init Time (s)':<30} " + "  ".join(
        f"{all_results[m]['init_time_s']:.2f}        " for m in methods))
    print(f"{'Query Time (s)':<30} " + "  ".join(
        f"{all_results[m]['query_time_s']:.2f}        " for m in methods))

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "Metric", "TF-IDF", "SBERT", "SBERT+QE"])
        for topic in TOPICS:
            for metric in ["P@5", "P@10", "AP", "num_relevant"]:
                row = [topic, metric]
                for m in methods:
                    row.append(f"{all_results[m][topic][metric]:.4f}"
                               if isinstance(all_results[m][topic][metric], float)
                               else str(all_results[m][topic][metric]))
                writer.writerow(row)
        # Add summary rows
        for metric in ["MAP", "avg_P@5", "avg_P@10"]:
            row = ["OVERALL", metric]
            for m in methods:
                row.append(f"{all_results[m][metric]:.4f}")
            writer.writerow(row)
        # Timing
        for metric in ["init_time_s", "query_time_s"]:
            row = ["TIMING", metric]
            for m in methods:
                row.append(f"{all_results[m][metric]:.4f}")
            writer.writerow(row)
    print(f"\nSaved metrics to {csv_path}")

    # Save full results as JSON
    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate charts
    generate_charts(all_results)

    return all_results


def generate_charts(all_results):
    """Generate matplotlib charts for the report."""
    method_names = ["TF-IDF", "SBERT", "SBERT+QE"]
    topic_labels = [TOPIC_DISPLAY_NAMES.get(t, t) for t in TOPICS]

    # Chart 1: P@5 per topic grouped by method
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TOPICS))
    width = 0.25
    colors = ["#6c757d", "#0d6efd", "#198754"]

    for i, method in enumerate(method_names):
        values = [all_results[method][t]["P@5"] for t in TOPICS]
        ax.bar(x + i * width, values, width, label=method, color=colors[i])

    ax.set_xlabel("Topic")
    ax.set_ylabel("Precision@5")
    ax.set_title("Precision@5 by Topic and Method")
    ax.set_xticks(x + width)
    ax.set_xticklabels(topic_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_at_5.png"), dpi=150)
    plt.close()

    # Chart 2: P@10 per topic grouped by method
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(method_names):
        values = [all_results[method][t]["P@10"] for t in TOPICS]
        ax.bar(x + i * width, values, width, label=method, color=colors[i])

    ax.set_xlabel("Topic")
    ax.set_ylabel("Precision@10")
    ax.set_title("Precision@10 by Topic and Method")
    ax.set_xticks(x + width)
    ax.set_xticklabels(topic_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_at_10.png"), dpi=150)
    plt.close()

    # Chart 3: MAP comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    map_values = [all_results[m]["MAP"] for m in method_names]
    bars = ax.bar(method_names, map_values, color=colors)
    ax.set_ylabel("Mean Average Precision")
    ax.set_title("MAP Comparison Across Methods")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, map_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "map_comparison.png"), dpi=150)
    plt.close()

    # Chart 4: Average Precision per topic (line chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(method_names):
        values = [all_results[method][t]["AP"] for t in TOPICS]
        ax.plot(topic_labels, values, marker="o", label=method, color=colors[i], linewidth=2)

    ax.set_xlabel("Topic")
    ax.set_ylabel("Average Precision")
    ax.set_title("Average Precision by Topic")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ap_by_topic.png"), dpi=150)
    plt.close()

    print(f"Charts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_all_experiments()
