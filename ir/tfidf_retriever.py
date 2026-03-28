"""
TF-IDF baseline retriever using scikit-learn.
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CORPUS_PATH, TOPIC_QUERIES, TOPICS


class TFIDFRetriever:
    def __init__(self, corpus_path=CORPUS_PATH):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.texts = [p["text"] for p in self.corpus]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=10):
        """Retrieve top-k passages for a query string."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            passage = self.corpus[idx].copy()
            passage["score"] = float(scores[idx])
            results.append(passage)
        return results

    def retrieve_for_topic(self, topic, top_k=10):
        """Retrieve top-k passages for a topic, split by candidate."""
        query = TOPIC_QUERIES[topic]["primary"]
        # Retrieve all to ensure both candidates are represented
        results = self.retrieve(query, top_k=len(self.corpus))

        by_candidate = {}
        for r in results:
            cand = r["candidate"]
            if cand not in by_candidate:
                by_candidate[cand] = []
            if len(by_candidate[cand]) < top_k:
                by_candidate[cand].append(r)

        return by_candidate

    def retrieve_all_ranked(self, topic, top_k=None):
        """Retrieve all passages ranked by relevance to a topic."""
        query = TOPIC_QUERIES[topic]["primary"]
        if top_k is None:
            top_k = len(self.corpus)
        return self.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    retriever = TFIDFRetriever()
    for topic in TOPICS:
        results = retriever.retrieve_for_topic(topic, top_k=3)
        print(f"\n=== {topic.upper()} ===")
        for cand, passages in results.items():
            print(f"  {cand}:")
            for p in passages[:2]:
                print(f"    [{p['score']:.3f}] {p['text'][:100]}...")
