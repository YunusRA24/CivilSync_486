"""
Political Issues Quiz
Compare candidates on issues that matter to you — without knowing who's who.
"""

import sys
import re
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

NUMBER_OF_SHOWN_QUOTES = 4

def load_text(path: str) -> str:
    full_path = os.path.join("database", path)

    if path.endswith(".pdf"):
        reader = PdfReader(full_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    with open(full_path, encoding="utf-8") as f:
        return f.read()


def split_sentences(text: str) -> list[str]:
    abbreviations = [
        "mr.", "mrs.", "ms.", "dr.", "sen.", "rep.", "u.s.", "u.k.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
        "sep.", "sept.", "oct.", "nov.", "dec."
    ]
    t = text.lower()
    for abbr in abbreviations:
        t = t.replace(abbr, abbr.replace(".", "<PERIOD>"))
    t = re.sub(r'\n+', ' ', t)
    sentences = re.split(r'(?<=[.!?])\s+', t)
    sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]
    return sentences


def build_name_variants(name_a: str, name_b: str) -> list[tuple[str, str]]:
  
    entries = [] 

    titles = [
        "president", "vice president", "vp", "senator", "sen",
        "governor", "gov", "secretary", "attorney general",
    ]

    for name, label in [(name_a, "Candidate A"), (name_b, "Candidate B")]:
        parts = name.strip().split()
        base_variants = [name] + parts

        variants = list(base_variants)
        for chunk in base_variants:
            for title in titles:
                variants.append(f"{title} {chunk}")

        seen: set[str] = set()
        unique: list[str] = []
        for v in variants:
            k = v.strip().lower()
            if k not in seen and len(k) > 2:
                seen.add(k)
                unique.append(v.strip())

        for v in unique:
            entries.append((re.escape(v), label))

    entries.sort(key=lambda x: len(x[0]), reverse=True)
    return entries


_PARTY_REPLACEMENTS = [
    # Long phrases first
    (r'\bthe\s+republican\s+party\b',          "the candidate's party"),
    (r'\bthe\s+democrat(?:ic)?\s+party\b',     "the candidate's party"),
    (r'\brepublican\s+party\b',                "the candidate's party"),
    (r'\bdemocrat(?:ic)?\s+party\b',           "the candidate's party"),
    (r'\bgop\b',                               "the candidate's party"),
    (r'\brepublicans\b',                       "members of the candidate's party"),
    (r'\bdemocrats\b',                         "members of the candidate's party"),
    (r'\brepublican\b',                        "the candidate's party"),
    (r'\bdemocrat(?:ic)?\b',                   "the candidate's party"),
    # Ticket / running-mate references
    (r'\btim walz\b',                          "the candidate's running mate"),
    (r'\bwalz\b',                              "the candidate's running mate"),
    (r'\bgovernor walz\b',                     "the candidate's running mate"),
    (r'\bsenator vance\b',                     "the candidate's running mate"),
    (r'\bjd vance\b',                          "the candidate's running mate"),
    (r'\bvance\b',                             "the candidate's running mate"),
    (r'\bjoe biden\b',                         "the candidate's running mate"),
    (r'\bbiden\b',                             "the candidate's running mate"),
    (r'\btim kaine\b',                         "the candidate's running mate"),
    # Generic political labels
    (r'\bright-wing\b',                        "[political]"),
    (r'\bleft-wing\b',                         "[political]"),
    (r'\bconservative\b',                      "[political]"),
    (r'\bliberal\b',                           "[political]"),
    (r'\bprogressive\b',                       "[political]"),
    (r'\bradical left\b',                      "[political]"),
    (r'\bhard right\b',                        "[political]"),
    (r'\bmaga\b',                              "[political]"),
    (r'\bfirst lady\b',                        "[political]"),
    (r'\btrump administration\b',              "the previous administration"),
    (r'\bbiden.harris administration\b',       "the previous administration"),
    (r'\bbiden administration\b',              "the previous administration"),
    (r'\bbidenharris\b',                       "the previous administration"),
]

# Compile once at module level for speed
_PARTY_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _PARTY_REPLACEMENTS
]


def anonymize(text: str, name_a: str, name_b: str,
              _cache: dict = {}) -> str:

    cache_key = (name_a, name_b)
    if cache_key not in _cache:
        _cache[cache_key] = build_name_variants(name_a, name_b)
    name_patterns = _cache[cache_key]

    result = text

    for pat, label in name_patterns:
        result = re.sub(r'(?i)(?<![a-zA-Z])' + pat + r'(?![a-zA-Z])', label, result)

    for compiled_pat, repl in _PARTY_PATTERNS:
        result = compiled_pat.sub(repl, result)

    result = re.sub(r'\bthe\s+the\b', 'the', result, flags=re.IGNORECASE)


    pronoun_map = [
        (r'\bshe\b',    'they'),
        (r'\bhe\b',     'they'),
        (r'\bhers\b',   'theirs'),
        (r'\bhis\b',    'their'),
        (r'\bher\b',    'their'),   
        (r'\bhim\b',    'them'),
        (r'\bbrother\b',   'sibling'),
        (r'\bsister\b',    'sibling'),
        (r'\bmom\b',    'parent'),
        (r'\bdad\b',    'parent'),
        (r'\bmother\b',    'parent'),
        (r'\bfather\b',    'parent'),
        (r'\bhim\b',    'them'),
        (r'\bhim\b',    'them'),
        (r'\bhim\b',    'them'),
        (r'\bherself\b','themselves'),
        (r'\bhimself\b','themselves'),
    ]
    for pat, repl in pronoun_map:
        result = re.sub(pat, repl, result, flags=re.IGNORECASE)

    return result


def build_issues(issues_file: str) -> dict:
    issues = {}
    with open(issues_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            title, _, term_line = line.partition(":")
            terms = [t.strip() for t in term_line.split(",") if t.strip()]
            issues[title.strip()] = terms
    return issues


_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

_JUNK_PATTERNS = re.compile(
    r'(https?://\S+|www\.\S+|\S+\.php\S*|\S+\.pdf\S*'
    r'|\bread\s+the\s+(plan|speech|op.?ed)\b'
    r'|\bfor\s+more\s+(information|info)\b'
    r'|\bat\s+a\s+glance\b'
    r'|\bmessage\s+from\b'
    r'|\bclick\s+here\b|\bsee\s+more\b'
    r'|^\s*\d[\d\s]{0,8}$'
    r')',
    re.IGNORECASE
)

_MIN_LEN        = 60
_MAX_LEN        = 500
_MIN_WORD_RATIO = 0.55


def _is_clean(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < _MIN_LEN:
        return False
    if _URL_RE.search(s):
        return False
    if _JUNK_PATTERNS.search(s):
        return False
    letters = sum(1 for c in s if c.isalpha())
    if letters / max(len(s), 1) < _MIN_WORD_RATIO:
        return False
    numbers = re.findall(r'\b\d+\b', s)
    words   = re.findall(r'[a-zA-Z]{3,}', s)
    if len(numbers) > 4 and len(words) < len(numbers) * 2:
        return False
    return True


def _trim_to_relevant(sentence: str, term: str, window: int = 35) -> str:
    if len(sentence) <= _MAX_LEN:
        return sentence
    words      = sentence.split()
    term_words = term.split()
    hit        = -1
    for i in range(len(words) - len(term_words) + 1):
        if words[i : i + len(term_words)] == term_words:
            hit = i
            break
    if hit == -1:
        return sentence[:_MAX_LEN].rsplit(' ', 1)[0] + '\u2026'
    start   = max(0, hit - window // 2)
    end     = min(len(words), hit + len(term_words) + window // 2)
    snippet = ' '.join(words[start:end])
    if start > 0:
        snippet = '\u2026' + snippet
    if end < len(words):
        snippet = snippet + '\u2026'
    return snippet


def find_relevant(sentences: list[str], terms: list[str]) -> list[str]:
    seen    = set()
    results = []
    for s in sentences:
        if not _is_clean(s):
            continue
        for term in terms:
            if term in s and s not in seen:
                seen.add(s)
                results.append(_trim_to_relevant(s, term))
                break
    return results


def load_politicians(filename="politicians.txt"):
    politicians = []

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # skip bad lines

            first = parts[0]
            last = parts[1]
            file = parts[2]

            full_name = f"{first} {last}"
            politicians.append((full_name, file))

    return politicians


def choose_politicians(politicians):
    print("\nAvailable politicians:")
    for i, (name, _) in enumerate(politicians):
        print(f"{i + 1}. {name}")

    choices = input("\nEnter two numbers (e.g., 1 2): ").split()

    if len(choices) != 2:
        print("Please select exactly two politicians.")
        sys.exit(1)

    try:
        idx1 = int(choices[0]) - 1
        idx2 = int(choices[1]) - 1
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

    return politicians[idx1], politicians[idx2]


DIVIDER  = "─" * 60
DIVIDER2 = "═" * 60

def clear(): print("\033[H\033[J", end="")

def print_header():
    print(DIVIDER2)
    print("  🗳   POLITICAL ISSUES QUIZ — UNBIASED EDITION   🗳")
    print(DIVIDER2)
    print("  Rank issues, read anonymized quotes, pick your side.")
    print(DIVIDER2)
    print()


def wrap(text: str, width: int = 76, indent: str = "   ") -> str:
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > width:
            lines.append(indent + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


def bullet_quotes(quotes: list[str], label: str) -> None:
    if not quotes:
        print(f"   (No quotes found for {label}.)")
        return
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(quotes)

    centroid = tfidf.mean(axis=0)
    scores = (tfidf @ centroid.T).A.flatten()

    scored_quotes = list(zip(quotes, scores))
    shown = [q for q, _ in sorted(scored_quotes, key=lambda x: x[1], reverse=True)[:NUMBER_OF_SHOWN_QUOTES]]

    for q in shown:
        if q.startswith('\u2026'):
            display = '\u2026' + q[1:2].upper() + q[2:]
        else:
            display = q[0].upper() + q[1:] if q else q

        print(wrap('\u2022 ' + display))
        print()
    

def get_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if lo <= val <= hi:
                return val
            print(f"   Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("   Please enter a whole number.")
            

#  STEP 1: RANK ISSUES

def rank_issues(issues: dict) -> dict:
    print_header()
    print("STEP 1 OF 3 — RANK THE ISSUES")
    print("How important is each issue to you?  (0 = don't care, 10 = top priority)\n")
    rankings = {}
    issue_list = list(issues.keys())
    for i, issue in enumerate(issue_list, 1):
        score = get_int(f"  [{i:2}/{len(issue_list)}] {issue:<22}: ", 0, 10)
        rankings[issue] = score
    active = {k: v for k, v in rankings.items() if v > 0}
    if not active:
        print("\n  You ranked every issue 0. Exiting.")
        sys.exit(0)
    return active


#  STEP 2 AND 3: PRESENT QUOTES + COLLECT CHOICES

def run_quiz(
    active_issues: dict,
    sents_a: list[str],
    sents_b: list[str],
    issues: dict,
    name_a: str,
    name_b: str,
) -> dict:

    results = {}


    if random.random() < 0.5:
        label_a, label_b = "A", "B"
        s_a, s_b = sents_a, sents_b
    else:
        label_a, label_b = "B", "A"
        s_a, s_b = sents_b, sents_a

    total = len(active_issues)
    for idx, (issue, points) in enumerate(active_issues.items(), 1):
        terms = issues[issue]
        quotes_a = find_relevant(s_a, terms)
        quotes_b = find_relevant(s_b, terms)

        if not quotes_a and not quotes_b:
            continue

        clear()
        print(DIVIDER2)
        print(f"  ISSUE {idx}/{total}:  {issue.upper()}   (weight: {points}/10)")
        print(DIVIDER2)
        print()
        print(f"  ── CANDIDATE {label_a} ──────────────────────────────────")
        print()
        bullet_quotes(
        [anonymize(q, name_a, name_b) for q in quotes_a],
        f"Candidate {label_a}")

        print(f"  ── CANDIDATE {label_b} ──────────────────────────────────")
        print()
        bullet_quotes(
        [anonymize(q, name_a, name_b) for q in quotes_b],
        f"Candidate {label_b}")

        print(DIVIDER)
        print(f"  Which position do you agree with more?")
        print(f"  [A] Candidate A    [B] Candidate B    [S] Skip / Tie")
        print()
        while True:
            choice = input("  Your choice: ").strip().upper()
            if choice in ("A", "B", "S"):
                break
            print("  Please enter A, B, or S.")

        if choice == "A":
            winner = name_a if label_a == "A" else name_b
        elif choice == "B":
            winner = name_a if label_b == "A" else name_b
        else:
            winner = None

        results[issue] = (winner, points)

    return results


#  STEP 4: RESULTS INTO A PIE CHART


def show_results(results: dict, name_a: str, name_b: str, out_path: str = "results.png"):
    scores = {name_a: 0, name_b: 0, "Skipped / Tied": 0}
    detail = {name_a: [], name_b: [], "Skipped / Tied": []}

    for issue, (winner, pts) in results.items():
        if winner is None:
            scores["Skipped / Tied"] += pts
            detail["Skipped / Tied"].append((issue, pts))
        else:
            scores[winner] += pts
            detail[winner].append((issue, pts))

    total = sum(scores.values()) or 1

    clear()
    print(DIVIDER2)
    print("  YOUR RESULTS")
    print(DIVIDER2)
    print()
    for name in (name_a, name_b, "Skipped / Tied"):
        pct = scores[name] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<22} {bar:<50} {pct:5.1f}%  ({scores[name]} pts)")
        for issue, pts in detail[name]:
            print(f"               • {issue} ({pts} pts)")
        print()

    # ── pie chart ──────────────────────────────
    colors = ["#9124BC", "#189A25", "#94A3B8"]
    labels, sizes, clrs = [], [], []
    for name, color in zip((name_a, name_b, "Skipped / Tied"), colors):
        if scores[name] > 0:
            labels.append(f"{name}\n({scores[name]} pts)")
            sizes.append(scores[name])
            clrs.append(color)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=clrs,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title("Which candidate matches your views?\n(weighted by your issue rankings)",
                 fontsize=13, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(DIVIDER)
    print(f"  Pie chart saved → {out_path}")
    print(DIVIDER)


#  MAIN

def main():
    if len(sys.argv) != 2:
        print('Usage: python political_quiz.py <issues_file>')
        sys.exit(1)

    iss_file = sys.argv[1]

    politicians = load_politicians()

    (name_a, file_a), (name_b, file_b) = choose_politicians(politicians)

    print("\nLoading files…", end=" ", flush=True)

    text_a = load_text(file_a)
    text_b = load_text(file_b)
    issues = build_issues(iss_file)
    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)

    print("done.\n")

    # Step 1: ranking
    active = rank_issues(issues)

    # Steps 2 and 3: quiz
    results = run_quiz(active, sents_a, sents_b, issues, name_a, name_b)

    # Step 4: results
    show_results(results, name_a, name_b)

if __name__ == "__main__":
    main()