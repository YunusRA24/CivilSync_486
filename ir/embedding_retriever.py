"""
Sentence-BERT embedding retriever using sentence-transformers.
"""
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CORPUS_PATH, TOPIC_QUERIES, TOPICS, EMBEDDING_MODEL, EMBEDDINGS_PATH
from ir.query_expansion import expand_query


class EmbeddingRetriever:
    def __init__(self, corpus_path=CORPUS_PATH, use_query_expansion=False):
        self.use_query_expansion = use_query_expansion

        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Load cached embeddings or compute new ones
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) != len(self.corpus):
                self._compute_embeddings()
        else:
            self._compute_embeddings()

    def _compute_embeddings(self):
        """Compute and cache passage embeddings."""
        print("Computing passage embeddings...")
        texts = [p["text"] for p in self.corpus]
        self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                             convert_to_numpy=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        print(f"Cached embeddings to {EMBEDDINGS_PATH}")

    def retrieve(self, query, top_k=10):
        """Retrieve top-k passages for a query string."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            passage = self.corpus[idx].copy()
            passage["score"] = float(scores[idx])
            results.append(passage)
        return results

    def retrieve_for_topic(self, topic, top_k=10):
        """Retrieve top-k passages for a topic, split by candidate."""
        if self.use_query_expansion:
            query = expand_query(topic)
        else:
            query = TOPIC_QUERIES[topic]["primary"]

        # Retrieve enough results to ensure both candidates are represented
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
        if self.use_query_expansion:
            query = expand_query(topic)
        else:
            query = TOPIC_QUERIES[topic]["primary"]
        if top_k is None:
            top_k = len(self.corpus)
        return self.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    print("Testing without query expansion:")
    retriever = EmbeddingRetriever(use_query_expansion=False)
    for topic in TOPICS[:2]:
        results = retriever.retrieve_for_topic(topic, top_k=3)
        print(f"\n=== {topic.upper()} ===")
        for cand, passages in results.items():
            print(f"  {cand}:")
            for p in passages[:2]:
                print(f"    [{p['score']:.3f}] {p['text'][:100]}...")

    print("\n\nTesting with query expansion:")
    retriever_qe = EmbeddingRetriever(use_query_expansion=True)
    for topic in TOPICS[:2]:
        results = retriever_qe.retrieve_for_topic(topic, top_k=3)
        print(f"\n=== {topic.upper()} (expanded) ===")
        for cand, passages in results.items():
            print(f"  {cand}:")
            for p in passages[:2]:
                print(f"    [{p['score']:.3f}] {p['text'][:100]}...")
