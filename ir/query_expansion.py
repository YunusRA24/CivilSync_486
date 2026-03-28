"""
Query expansion using predefined seed terms from config.
Expands topic queries with semantically related terms.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOPIC_QUERIES


def expand_query(topic):
    """Expand a topic query by combining primary query with seed terms."""
    query_info = TOPIC_QUERIES[topic]
    primary = query_info["primary"]
    seeds = query_info["expansion_seeds"]

    # Combine primary query with all seed terms (deduplicated)
    all_terms = primary.split() + seeds
    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in all_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    return " ".join(unique_terms)


def get_expanded_queries():
    """Return expanded queries for all topics."""
    return {topic: expand_query(topic) for topic in TOPIC_QUERIES}


if __name__ == "__main__":
    queries = get_expanded_queries()
    for topic, query in queries.items():
        print(f"{topic}: {query}")
