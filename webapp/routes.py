"""
Flask route handlers for CivilSync's three-page flow:
1. Topic importance form
2. Blinded quote comparison
3. Results pie chart
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify
from config import TOPICS, TOPIC_DISPLAY_NAMES, CANDIDATE_DISPLAY_NAMES, CANDIDATES, BASE_DIR
from ir.embedding_retriever import EmbeddingRetriever

bp = Blueprint("main", __name__)

# Initialize retriever once at import time
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = EmbeddingRetriever(use_query_expansion=True)
    return _retriever


def load_default_weights():
    """Load default topic weights from userinput.csv."""
    weights = {}
    csv_path = os.path.join(BASE_DIR, "userinput.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic = row["Topic"].strip()
                points = int(row["points"].strip())
                weights[topic] = points
    return weights


@bp.route("/", methods=["GET", "POST"])
def topics():
    """Page 1: Topic importance form."""
    if request.method == "POST":
        weights = {}
        for topic in TOPICS:
            val = request.form.get(topic, "5")
            weights[topic] = int(val)
        session["weights"] = weights

        # Generate random candidate shuffle mapping
        candidates = CANDIDATES.copy()
        random.shuffle(candidates)
        session["shuffle"] = {"A": candidates[0], "B": candidates[1]}

        return redirect(url_for("main.compare"))

    # GET: show form with default values
    defaults = load_default_weights()
    topics_data = []
    for topic in TOPICS:
        topics_data.append({
            "key": topic,
            "name": TOPIC_DISPLAY_NAMES.get(topic, topic),
            "default": defaults.get(topic, 5),
        })
    return render_template("topics.html", topics=topics_data)


@bp.route("/compare", methods=["GET", "POST"])
def compare():
    """Page 2: Blinded quote comparison."""
    if "weights" not in session or "shuffle" not in session:
        return redirect(url_for("main.topics"))

    if request.method == "POST":
        choices = {}
        for topic in TOPICS:
            choice = request.form.get(topic)
            if choice:
                choices[topic] = choice  # "A" or "B"
        session["choices"] = choices
        return redirect(url_for("main.results"))

    # GET: retrieve quotes for each topic
    retriever = get_retriever()
    shuffle = session["shuffle"]
    comparisons = []

    for topic in TOPICS:
        results = retriever.retrieve_for_topic(topic, top_k=5)
        candidate_a = shuffle["A"]
        candidate_b = shuffle["B"]

        quotes_a = results.get(candidate_a, [])
        quotes_b = results.get(candidate_b, [])

        comparisons.append({
            "topic_key": topic,
            "topic_name": TOPIC_DISPLAY_NAMES.get(topic, topic),
            "quotes_a": [q["text"] for q in quotes_a[:5]],
            "quotes_b": [q["text"] for q in quotes_b[:5]],
        })

    return render_template("compare.html", comparisons=comparisons)


@bp.route("/results")
def results():
    """Page 3: Results pie chart with candidate reveal."""
    if "choices" not in session or "weights" not in session or "shuffle" not in session:
        return redirect(url_for("main.topics"))

    weights = session["weights"]
    choices = session["choices"]
    shuffle = session["shuffle"]

    # Map choices back to real candidates
    topic_results = []
    candidate_scores = {c: 0 for c in CANDIDATES}

    for topic in TOPICS:
        choice = choices.get(topic, "A")
        chosen_candidate = shuffle[choice]
        weight = weights.get(topic, 5)
        candidate_scores[chosen_candidate] += weight

        topic_results.append({
            "topic": TOPIC_DISPLAY_NAMES.get(topic, topic),
            "weight": weight,
            "chosen": choice,
            "candidate": CANDIDATE_DISPLAY_NAMES.get(chosen_candidate, chosen_candidate),
            "candidate_key": chosen_candidate,
        })

    total_points = sum(candidate_scores.values())
    percentages = {
        CANDIDATE_DISPLAY_NAMES.get(c, c): round(s / total_points * 100, 1) if total_points > 0 else 0
        for c, s in candidate_scores.items()
    }

    # Pie chart data
    chart_labels = [r["topic"] + f" ({r['weight']})" for r in topic_results]
    chart_data = [r["weight"] for r in topic_results]
    chart_colors = [
        "#3b82f6" if r["candidate_key"] == CANDIDATES[0] else "#ef4444"
        for r in topic_results
    ]

    reveal = {
        "A": CANDIDATE_DISPLAY_NAMES.get(shuffle["A"], shuffle["A"]),
        "B": CANDIDATE_DISPLAY_NAMES.get(shuffle["B"], shuffle["B"]),
    }

    return render_template(
        "results.html",
        topic_results=topic_results,
        percentages=percentages,
        chart_labels=chart_labels,
        chart_data=chart_data,
        chart_colors=chart_colors,
        reveal=reveal,
        candidate_display=CANDIDATE_DISPLAY_NAMES,
        candidates=CANDIDATES,
    )
