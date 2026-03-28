"""
Parse raw HTML from Wayback Machine snapshots.
Extract main content text and filter out opponent references.
"""
import os
import re
from bs4 import BeautifulSoup


# Terms to filter out — keyed by candidate, listing opponent references
OPPONENT_TERMS = {
    "trump": ["harris", "kamala", "biden", "joe biden", "democrat party",
              "the democrats", "radical left", "crooked"],
    "harris": ["trump", "donald", "maga", "republican party",
               "the republicans", "project 2025"],
}


def extract_text_from_html(html_content):
    """Extract clean text from HTML, removing boilerplate."""
    soup = BeautifulSoup(html_content, "lxml")

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                               "noscript", "iframe", "svg"]):
        tag.decompose()

    # Try to find main content area
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", {"role": "main"}) or
        soup.find("div", class_=re.compile(r"content|main|body", re.I))
    )

    target = main if main else soup.body if soup.body else soup

    # Get text
    text = target.get_text(separator="\n", strip=True)

    # Clean up
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"Paid for by.*",
        r"Privacy Policy.*",
        r"Terms of Service.*",
        r"Donate\s*(Now)?",
        r"Sign Up.*",
        r"Subscribe.*",
        r"Follow us.*",
        r"Share (this|on).*",
        r"Copyright ©.*",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def filter_opponent_references(passages, candidate):
    """Remove passages that reference the opposing candidate."""
    block_terms = OPPONENT_TERMS.get(candidate, [])
    filtered = []
    for passage in passages:
        text_lower = passage.lower()
        if not any(term in text_lower for term in block_terms):
            filtered.append(passage)
    return filtered


def parse_candidate_pages(candidate, raw_dir):
    """Parse all raw HTML files for a candidate and return clean text passages."""
    candidate_dir = os.path.join(raw_dir, candidate)
    if not os.path.exists(candidate_dir):
        print(f"  No raw data directory for {candidate}")
        return []

    all_texts = []
    html_files = [f for f in os.listdir(candidate_dir) if f.endswith(".html")]

    if not html_files:
        print(f"  No HTML files found for {candidate}")
        return []

    for filename in sorted(html_files):
        filepath = os.path.join(candidate_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html = f.read()
            text = extract_text_from_html(html)
            if text and len(text) > 50:  # Skip near-empty pages
                all_texts.append({
                    "text": text,
                    "source_file": filename,
                })
        except Exception as e:
            print(f"  Error parsing {filename}: {e}")

    print(f"  Parsed {len(all_texts)} pages for {candidate}")
    return all_texts
