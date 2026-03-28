"""
Process userinput.csv and run the CivilSync pipeline from the command line.
Reads topic weights from CSV, retrieves quotes, and outputs alignment results.
"""
import csv
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TOPICS, TOPIC_DISPLAY_NAMES, CANDIDATE_DISPLAY_NAMES, CANDIDATES
from ir.embedding_retriever import EmbeddingRetriever


def load_user_input(csv_path="userinput.csv"):
    """Load topic weights from userinput.csv."""
    weights = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["Topic"].strip()
            points = int(row["points"].strip())
            weights[topic] = points
    return weights


def run_pipeline(csv_path="userinput.csv"):
    """Run the full CivilSync pipeline from CSV input."""
    weights = load_user_input(csv_path)
    print("Topic weights loaded from", csv_path)
    for topic in TOPICS:
        w = weights.get(topic, 0)
        print(f"  {TOPIC_DISPLAY_NAMES.get(topic, topic)}: {w}")

    print("\nInitializing retriever...")
    retriever = EmbeddingRetriever(use_query_expansion=True)

    print("\nRetrieving quotes for each topic...\n")
    candidate_scores = {c: 0 for c in CANDIDATES}

    for topic in TOPICS:
        results = retriever.retrieve_for_topic(topic, top_k=5)
        weight = weights.get(topic, 0)
        display = TOPIC_DISPLAY_NAMES.get(topic, topic)

        print(f"=== {display} (importance: {weight}) ===")
        for cand in CANDIDATES:
            cand_display = CANDIDATE_DISPLAY_NAMES[cand]
            quotes = results.get(cand, [])
            print(f"\n  {cand_display}:")
            for i, q in enumerate(quotes[:3], 1):
                text = q["text"][:150].replace("\n", " ")
                print(f"    {i}. {text}...")

        # Simulate a random choice for demo
        chosen = random.choice(CANDIDATES)
        candidate_scores[chosen] += weight
        print(f"\n  [Demo: randomly selected {CANDIDATE_DISPLAY_NAMES[chosen]}]\n")

    total = sum(candidate_scores.values())
    print("\n" + "=" * 50)
    print("ALIGNMENT RESULTS (demo - random selections)")
    print("=" * 50)
    for cand in CANDIDATES:
        name = CANDIDATE_DISPLAY_NAMES[cand]
        score = candidate_scores[cand]
        pct = (score / total * 100) if total > 0 else 0
        print(f"  {name}: {score} points ({pct:.1f}%)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "userinput.csv"
    run_pipeline(csv_path)
