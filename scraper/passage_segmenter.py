"""
Split cleaned text into quote-sized passages for IR retrieval.
"""
import os
import re
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, CANDIDATES_DIR, CANDIDATES, WAYBACK_DATE_FROM, WAYBACK_DATE_TO
from scraper.html_parser import parse_candidate_pages, filter_opponent_references


def split_into_sentences(text):
    """Split text into sentences using regex."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for sent in sentences:
        parts = sent.split("\n\n")
        result.extend(parts)
    return [s.strip() for s in result if s.strip()]


def create_passages(sentences, min_words=10, max_words=100, sentences_per_passage=3):
    """Group sentences into passages of 1-3 sentences."""
    passages = []
    i = 0
    while i < len(sentences):
        # Take up to sentences_per_passage sentences
        group = []
        word_count = 0
        for j in range(i, min(i + sentences_per_passage, len(sentences))):
            sent = sentences[j]
            words = len(sent.split())
            if word_count + words > max_words and group:
                break
            group.append(sent)
            word_count += words

        passage = " ".join(group)
        if len(passage.split()) >= min_words:
            passages.append(passage)

        i += max(len(group), 1)

    return passages


def process_candidate(candidate):
    """Process all scraped pages for a candidate into passages."""
    print(f"\nProcessing {candidate}...")

    # Parse HTML files
    parsed_pages = parse_candidate_pages(candidate, RAW_DIR)

    if not parsed_pages:
        print(f"  No parsed pages for {candidate}")
        return []

    all_passages = []
    passage_id = 0

    for page in parsed_pages:
        text = page["text"]
        source = page["source_file"]

        # Split into sentences
        sentences = split_into_sentences(text)

        # Create passages
        passages = create_passages(sentences)

        # Filter opponent references
        passages = filter_opponent_references(passages, candidate)

        for passage_text in passages:
            all_passages.append({
                "passage_id": f"{candidate}_{passage_id:03d}",
                "candidate": candidate,
                "text": passage_text,
                "source_file": source,
            })
            passage_id += 1

    print(f"  Created {len(all_passages)} passages for {candidate}")
    return all_passages


def run_segmenter():
    """Run the passage segmentation pipeline for all candidates."""
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    all_corpus = []
    for candidate in CANDIDATES:
        passages = process_candidate(candidate)
        # Save per-candidate
        output_path = os.path.join(CANDIDATES_DIR, f"{candidate}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(passages, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(passages)} passages to {output_path}")
        all_corpus.extend(passages)

    # Save combined corpus
    corpus_path = os.path.join(CANDIDATES_DIR, "..", "corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_corpus, f, indent=2, ensure_ascii=False)
    print(f"\nTotal corpus: {len(all_corpus)} passages saved to {corpus_path}")

    return all_corpus


if __name__ == "__main__":
    run_segmenter()
