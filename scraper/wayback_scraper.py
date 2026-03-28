"""
Scrape candidate campaign pages from the Wayback Machine.
Uses the CDX API to discover archived snapshots, then fetches HTML.
"""
import os
import time
import hashlib
import requests
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WAYBACK_CDX_API, CANDIDATE_URLS, WAYBACK_DATE_FROM,
    WAYBACK_DATE_TO, RAW_DIR, MAX_PAGES_PER_CANDIDATE
)


def query_cdx(url_pattern, date_from=WAYBACK_DATE_FROM, date_to=WAYBACK_DATE_TO):
    """Query the Wayback Machine CDX API for archived snapshots of a URL pattern."""
    params = {
        "url": url_pattern,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "from": date_from,
        "to": date_to,
        "collapse": "urlkey",
        "limit": 50,
    }
    try:
        resp = requests.get(WAYBACK_CDX_API, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if len(rows) < 2:
            return []
        header = rows[0]
        return [dict(zip(header, row)) for row in rows[1:]]
    except Exception as e:
        print(f"  CDX query failed for {url_pattern}: {e}")
        return []


def fetch_archived_page(timestamp, original_url):
    """Fetch an archived page from the Wayback Machine."""
    wayback_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
    try:
        resp = requests.get(wayback_url, timeout=30, headers={
            "User-Agent": "CivilSync-486-CourseProject/1.0"
        })
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  Failed to fetch {wayback_url}: {e}")
        return None


def deduplicate_snapshots(snapshots):
    """Keep only one snapshot per unique URL path (ignore query params)."""
    seen_paths = set()
    deduped = []
    for snap in snapshots:
        parsed = urlparse(snap["original"])
        path = parsed.netloc + parsed.path.rstrip("/")
        if path not in seen_paths:
            seen_paths.add(path)
            deduped.append(snap)
    return deduped


def scrape_candidate(candidate, url_patterns, max_pages=MAX_PAGES_PER_CANDIDATE):
    """Scrape archived pages for a candidate, up to max_pages."""
    candidate_dir = os.path.join(RAW_DIR, candidate)
    os.makedirs(candidate_dir, exist_ok=True)

    all_snapshots = []
    for pattern in url_patterns:
        print(f"  Querying CDX for {pattern}...")
        snapshots = query_cdx(pattern + "*")
        if not snapshots:
            snapshots = query_cdx(pattern)
        all_snapshots.extend(snapshots)
        time.sleep(1)

    # Filter out non-page URLs
    skip_exts = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".ico", ".pdf", ".woff", ".woff2")
    all_snapshots = [s for s in all_snapshots
                     if not any(s["original"].lower().endswith(ext) for ext in skip_exts)]

    # Deduplicate by URL path (ignore query params)
    all_snapshots = deduplicate_snapshots(all_snapshots)

    # Limit to max_pages
    all_snapshots = all_snapshots[:max_pages]

    print(f"  Found {len(all_snapshots)} unique snapshots for {candidate} (limit {max_pages})")

    fetched = 0
    for snap in all_snapshots:
        timestamp = snap["timestamp"]
        original = snap["original"]
        slug = original.replace("https://", "").replace("http://", "").replace("/", "_")
        if len(slug) > 80:
            slug = slug[:60] + "_" + hashlib.md5(slug.encode()).hexdigest()[:12]
        filename = f"{timestamp}_{slug}.html"
        filepath = os.path.join(candidate_dir, filename)

        if os.path.exists(filepath):
            print(f"  Already have {filename}, skipping")
            fetched += 1
            continue

        print(f"  Fetching {original} ({timestamp})...")
        html = fetch_archived_page(timestamp, original)
        if html:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            fetched += 1
        time.sleep(1.5)

    print(f"  Fetched {fetched}/{len(all_snapshots)} pages for {candidate}")
    return fetched


def run_scraper():
    """Run the full scraping pipeline."""
    os.makedirs(RAW_DIR, exist_ok=True)
    # Clear old data to start fresh
    for candidate in CANDIDATE_URLS:
        candidate_dir = os.path.join(RAW_DIR, candidate)
        if os.path.exists(candidate_dir):
            for f in os.listdir(candidate_dir):
                os.remove(os.path.join(candidate_dir, f))

    total = 0
    for candidate, urls in CANDIDATE_URLS.items():
        print(f"\nScraping {candidate}...")
        count = scrape_candidate(candidate, urls)
        total += count
    print(f"\nDone. Total pages fetched: {total}")
    return total


if __name__ == "__main__":
    run_scraper()
