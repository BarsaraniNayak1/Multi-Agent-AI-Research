from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup


def extract_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(part.strip() for part in soup.get_text(" ", strip=True).split())
    return text


def _same_domain(seed: str, candidate: str) -> bool:
    try:
        seed_host = urlparse(seed).netloc.lower()
        candidate_host = urlparse(candidate).netloc.lower()
        return not candidate_host or candidate_host == seed_host
    except Exception:
        return False


def crawl_urls(
    urls: list[str],
    source_type: str,
    max_depth: int = 2,
    respect_robots: bool = True,
    timeout: int = 15,
    storage: Any | None = None,
    errors: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not urls:
        return []

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque((url, 0) for url in urls if url)
    results: list[dict[str, Any]] = []
    robots_cache: dict[str, robotparser.RobotFileParser] = {}
    min_delay = float(os.getenv("RESEARCH_CRAWL_DELAY", "0.25"))
    last_request_at = 0.0

    for seed_url in urls:
        parsed = urlparse(seed_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid seed URL: {seed_url}")

    while queue:
        current_url, depth = queue.popleft()
        if not current_url or current_url in seen:
            continue
        seen.add(current_url)

        parsed_url = urlparse(current_url)
        if respect_robots:
            origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
            if origin not in robots_cache:
                parser = robotparser.RobotFileParser(f"{origin}/robots.txt")
                try:
                    parser.read()
                except Exception:
                    parser = robotparser.RobotFileParser()
                    parser.parse([])
                robots_cache[origin] = parser
            if not robots_cache[origin].can_fetch("research-copilot", current_url):
                if errors is not None:
                    errors.append(f"Blocked by robots.txt: {current_url}")
                continue

        response = None
        try:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
            response = requests.get(current_url, timeout=timeout, allow_redirects=True)
            last_request_at = time.monotonic()
            response.raise_for_status()
        except Exception as error:
            if errors is not None:
                if response is not None and getattr(response, "status_code", None):
                    status = response.status_code
                    reason = getattr(response, "reason", "")
                    detail = f"HTTP {status} {reason}".strip()
                    errors.append(f"{detail} while fetching {current_url}.")
                else:
                    errors.append(f"Could not fetch {current_url}: {error}")
            continue

        text = extract_clean_text(response.text)
        if not text:
            if errors is not None:
                errors.append(f"No readable text was extracted from {current_url}.")
            continue

        doc_id = hashlib.sha256(current_url.encode("utf-8")).hexdigest()[:16]
        result = {
            "id": doc_id,
            "url": current_url,
            "source_type": source_type,
            "text": text,
            "title": current_url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }
        if storage is not None:
            result["storage_id"] = storage.upsert(result)
        results.append(result)

        if depth >= max_depth:
            continue

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                if not href or href.startswith("mailto:"):
                    continue
                next_url = urljoin(current_url, href)
                if not _same_domain(current_url, next_url):
                    continue
                if next_url not in seen:
                    queue.append((next_url, depth + 1))
        except Exception:
            continue

    return results
