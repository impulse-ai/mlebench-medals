"""Policy-gated web research hooks for EES strategy discovery."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from html import unescape
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile
from typing import Callable, Protocol
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bench.adapters.task_detection import IMAGE_EXTS
from bench.ees_core.strategy import StrategyHypothesis, StrategyProbeResult, TaskEvidence


@dataclass(frozen=True)
class WebResearchResult:
    status: str
    query_count: int
    sources: list[dict[str, str]]
    notes: list[str]
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SourceCacheBuildResult:
    status: str
    sources_written: int
    manifest_path: Path
    notes: list[str]
    error: str | None = None
    labels_with_aliases: int = 0
    labels_with_sources: int = 0

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


class WebResearchProvider(Protocol):
    def search(
        self,
        evidence: TaskEvidence,
        hypothesis: StrategyHypothesis,
    ) -> WebResearchResult:
        ...


@dataclass(frozen=True)
class NoopWebResearchProvider:
    reason: str = "web_research_disabled"

    def search(
        self,
        evidence: TaskEvidence,
        hypothesis: StrategyHypothesis,
    ) -> WebResearchResult:
        return WebResearchResult(
            status="skipped",
            query_count=0,
            sources=[],
            notes=[self.reason],
            error=None,
        )


def build_research_queries(
    evidence: TaskEvidence,
    hypothesis: StrategyHypothesis,
    *,
    max_queries: int = 3,
) -> list[str]:
    hints = set(evidence.hints)
    queries = [
        f"{evidence.task_id} Kaggle competition data source winning solution",
    ]
    if "public_domain" in hints or hypothesis.strategy_id == "external_provenance":
        queries.append(f"{evidence.task_id} public domain source texts")
    if "scientific_structure" in hints:
        queries.append(f"{evidence.task_id} scientific feature engineering geometry")
    if "filename_labels" in hints or evidence.has_image_files or evidence.has_audio_files:
        queries.append(f"{evidence.task_id} filename metadata leakage")
    description_terms = " ".join(evidence.description.split()[:20])
    if description_terms:
        queries.append(f"{evidence.task_id} {description_terms}")
    deduped = list(dict.fromkeys(query for query in queries if query.strip()))
    return deduped[:max_queries]


def _clean_html_text(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _clean_source_text(text: str) -> str:
    without_scripts = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if "<" in without_scripts and ">" in without_scripts:
        return _clean_html_text(without_scripts)
    return re.sub(r"\s+", " ", unescape(without_scripts)).strip()


def _match_normalize(text: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "source"


def _candidate_labels(evidence: TaskEvidence) -> list[str]:
    labels = [label for label in evidence.target_cols if label != evidence.id_col]
    if not labels:
        labels = [
            label
            for label in evidence.sample_submission_cols
            if label != evidence.id_col and str(label).lower() not in {"id"}
        ]
    return [str(label) for label in labels if str(label).strip()]


def _source_label(source: dict[str, str], evidence: TaskEvidence) -> str | None:
    labels = _candidate_labels(evidence)
    if not labels:
        return None
    try:
        from bench.ees_core.operators.text_provenance import infer_label_aliases

        aliases = infer_label_aliases(evidence.description, labels)
    except Exception:
        aliases = {}

    haystack = _match_normalize(
        " ".join([source.get("title", ""), source.get("url", ""), source.get("snippet", "")])
    )
    matches: list[str] = []
    for label in labels:
        tokens = list(aliases.get(label, []))
        tokens.extend(_description_parenthetical_aliases(evidence.description, label))
        normalized_label = _match_normalize(label)
        if len(normalized_label) >= 3:
            tokens.append(normalized_label)
        if any(token and token in haystack for token in tokens):
            matches.append(label)
    return matches[0] if len(matches) == 1 else None


def _description_parenthetical_aliases(description: str, label: str) -> list[str]:
    aliases: list[str] = []
    if not description or not label:
        return aliases
    pattern = re.compile(
        rf"([A-Za-z][A-Za-z.\s'-]{{2,100}}?)\s*\(\s*{re.escape(str(label))}\s*\)",
        flags=re.IGNORECASE,
    )
    stop_words = {"predict", "authors", "author", "by", "and", "or", "the", "of"}
    for match in pattern.finditer(description):
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", match.group(1))
        tail = [word for word in words[-5:] if word.lower() not in stop_words]
        for size in range(min(4, len(tail)), 0, -1):
            alias = _match_normalize(" ".join(tail[-size:]))
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def _source_score(source: dict[str, str], evidence: TaskEvidence) -> float:
    label = _source_label(source, evidence)
    haystack = _match_normalize(
        " ".join([source.get("title", ""), source.get("url", ""), source.get("snippet", "")])
    )
    score = 0.0
    if label:
        score += 10.0
    for token in ["complete works", "collected works", "public domain", "source", "dataset"]:
        if token in haystack:
            score += 1.0
    for hint in evidence.hints:
        normalized = _match_normalize(hint)
        if normalized and normalized in haystack:
            score += 0.5
    return score


def parse_source_links(html: str, *, base_url: str, max_links: int = 20) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>", html, flags=re.IGNORECASE):
        url = urljoin(base_url, unescape(match.group(1)).strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized = parsed._replace(fragment="").geturl()
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
        if len(links) >= max_links:
            break
    return links


def rank_and_expand_sources(
    sources: list[dict[str, str]],
    evidence: TaskEvidence,
    *,
    fetcher: Callable[[str], str] | None = None,
    max_sources: int = 20,
    max_links_per_source: int = 3,
) -> list[dict[str, str]]:
    fetcher = fetcher or fetch_web_source_text
    ranked = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", "")).strip()
        if not url:
            continue
        scored = dict(source)
        scored["source_rank_score"] = f"{_source_score(scored, evidence):.3f}"
        ranked.append(scored)
    ranked = sorted(ranked, key=lambda source: float(source["source_rank_score"]), reverse=True)

    expanded: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for source in ranked:
        url = source["url"]
        if url not in seen_urls:
            expanded.append(source)
            seen_urls.add(url)
        if len(expanded) >= max_sources:
            break
        if max_links_per_source <= 0:
            continue
        try:
            html = fetcher(url)
        except Exception:
            continue
        link_candidates = []
        for link in parse_source_links(html, base_url=url, max_links=max_links_per_source * 5):
            linked_source = {
                "title": link.rsplit("/", 1)[-1].replace("-", " ") or link,
                "url": link,
                "snippet": source.get("title", ""),
            }
            linked_score = _source_score(linked_source, evidence)
            if linked_score <= 0.0:
                continue
            linked_source["source_rank_score"] = f"{linked_score:.3f}"
            linked_source["expanded_from"] = url
            link_candidates.append(linked_source)
        for linked_source in sorted(link_candidates, key=lambda item: float(item["source_rank_score"]), reverse=True):
            if linked_source["url"] in seen_urls:
                continue
            expanded.append(linked_source)
            seen_urls.add(linked_source["url"])
            if len([item for item in expanded if item.get("expanded_from") == url]) >= max_links_per_source:
                break
            if len(expanded) >= max_sources:
                break
        if len(expanded) >= max_sources:
            break
    return expanded[:max_sources]


def fetch_web_source_text(url: str) -> str:
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": "impulse-ees-source-cache/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def build_source_cache_from_web_manifest(
    manifest_path: Path,
    output_dir: Path,
    evidence: TaskEvidence,
    *,
    fetcher: Callable[[str], str] = fetch_web_source_text,
    min_chars: int = 20,
    max_sources: int = 20,
) -> SourceCacheBuildResult:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    source_manifest_path = output_dir / "source_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        return SourceCacheBuildResult(
            status="skipped",
            sources_written=0,
            manifest_path=source_manifest_path,
            notes=["missing_web_manifest"],
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = payload.get("research", {}).get("sources", [])
    except Exception as exc:
        return SourceCacheBuildResult(
            status="failed",
            sources_written=0,
            manifest_path=source_manifest_path,
            notes=["unreadable_web_manifest"],
            error=f"{type(exc).__name__}: {exc}",
        )

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    ranked_sources = rank_and_expand_sources(
        sources,
        evidence,
        fetcher=fetcher,
        max_sources=max_sources,
        max_links_per_source=2,
    )
    for index, source in enumerate(ranked_sources):
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", "")).strip()
        if not url:
            continue
        label = _source_label(source, evidence)
        if not label:
            continue
        try:
            raw = fetcher(url)
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        text = _clean_source_text(raw)
        if not _is_plausible_plain_text(raw, text):
            errors.append(f"{url}: rejected_non_plain_text")
            continue
        if len(text) < min_chars:
            continue
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        title = str(source.get("title", "") or url)
        local_path = label_dir / f"{index:03d}-{_slugify(title)}.txt"
        local_path.write_text(text + "\n", encoding="utf-8")
        rows.append(
            {
                "author": label,
                "local_path": str(local_path),
                "url": url,
                "title": title,
            }
        )

    if rows:
        with source_manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["author", "local_path", "url", "title"])
            writer.writeheader()
            writer.writerows(rows)
        return SourceCacheBuildResult(
            status="success",
            sources_written=len(rows),
            manifest_path=source_manifest_path,
            notes=["web_source_cache_built"],
            error="; ".join(errors) if errors else None,
            labels_with_sources=len({row["author"] for row in rows}),
        )

    return SourceCacheBuildResult(
        status="skipped" if not errors else "failed",
        sources_written=0,
        manifest_path=source_manifest_path,
        notes=["no_labelled_sources_cached"],
        error="; ".join(errors) if errors else None,
    )


# ---------------------------------------------------------------------------
# Image source-corpus acquisition (second-modality sibling of the label-source
# text acquisition above). Where text infers a per-label author and fetches
# their public-domain works, an image task typically draws ALL its classes from
# ONE public source dataset (e.g. a canine subset of ImageNet drawn from the
# Stanford Dogs dataset). We infer that dataset's name + a fetch URL from the
# task description, download the archive, and extract it into a class-labelled
# directory that ``discover_corpus_class_images`` already knows how to read.
#
# Honesty is NOT enforced here: the acquired corpus is only ever a HYPOTHESIS.
# The same train-side precision>=0.995 / coverage>=0.2 image_provenance probe
# gates promotion, so a wrong source (wrong dataset, mismatched classes, or
# unrelated images) can never fake a medal.
# ---------------------------------------------------------------------------

_IMAGE_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".zip")
_FRAME_SRC_PATTERN = re.compile(r"<frame\b[^>]*\bsrc=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
# Words that mark a hyperlink's anchor text as naming a source dataset rather
# than a paper, competition page, or inline image.
_IMAGE_SOURCE_NAME_TOKENS = ("dataset", "corpus", "database", "imagenet", "collection")


@dataclass(frozen=True)
class ImageSourceCandidate:
    name: str
    url: str
    score: float

    def to_json(self) -> dict:
        return asdict(self)


def _url_basename(url: str) -> str:
    path = urlparse(url).path
    return path.rsplit("/", 1)[-1].lower()


def _has_image_extension(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in IMAGE_EXTS)


def _has_archive_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _IMAGE_ARCHIVE_EXTS)


def infer_image_source_candidates(description: str) -> list[ImageSourceCandidate]:
    """Infer (dataset name, fetch URL) candidates for a public image source.

    Evidence-only and generic: it reads markdown links ``[Name](url)`` (and, as
    a fallback, bare URLs) from the task description and ranks them by how much
    they look like a downloadable source-dataset reference. There is NO
    task-id or dataset-name allowlist -- ranking is by anchor-text tokens
    ("dataset"/"corpus"/...), host, and URL shape, so any task whose
    description links its source dataset can trigger acquisition.
    """
    if not description:
        return []
    candidates: dict[str, ImageSourceCandidate] = {}

    def _consider(name: str, url: str) -> None:
        url = unescape(url).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return
        basename = _url_basename(url)
        if _has_image_extension(basename):
            return  # inline figure, not a dataset
        name = re.sub(r"\s+", " ", name or "").strip()
        name_lower = name.lower()
        host = parsed.netloc.lower()
        score = 0.0
        for token in _IMAGE_SOURCE_NAME_TOKENS:
            if token in name_lower:
                score += 3.0 if token in ("dataset", "corpus", "database") else 1.0
        # Kaggle competition/wiki pages are landing pages, not raw source data.
        if "kaggle.com" in host:
            score -= 4.0
        if _has_archive_extension(url):
            score += 4.0
        if host.endswith(".edu") or "archive.org" in host or "vision." in host:
            score += 1.0
        existing = candidates.get(url)
        if existing is None or score > existing.score:
            candidates[url] = ImageSourceCandidate(name=name or basename or url, url=url, score=score)

    for match in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", description):
        _consider(match.group(1), match.group(2))
    if not any(candidate.score > 0 for candidate in candidates.values()):
        # No markdown link scored as a dataset; fall back to bare URLs, using
        # the preceding words on the line as a best-effort name.
        for match in re.finditer(r"(https?://[^\s)\]]+)", description):
            url = match.group(1)
            prefix = description[max(0, match.start() - 60) : match.start()]
            name = re.sub(r"[^A-Za-z0-9 ]+", " ", prefix).split()[-6:]
            _consider(" ".join(name), url)

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, len(candidate.url)),
    )
    return [candidate for candidate in ranked if candidate.score > 0]


def _collect_archive_links(
    url: str,
    fetcher: Callable[[str], str],
    *,
    max_pages: int = 6,
) -> list[str]:
    """Breadth-first crawl of a landing page (and its <frame> children) for
    archive links. Stanford Dogs, for example, serves a frameset whose real
    download links live in ``main.html``."""
    seen_pages: set[str] = set()
    to_visit: list[str] = [url]
    archives: list[str] = []
    seen_archives: set[str] = set()
    while to_visit and len(seen_pages) < max_pages:
        page = to_visit.pop(0)
        if page in seen_pages:
            continue
        seen_pages.add(page)
        try:
            html = fetcher(page)
        except Exception:
            continue
        for match in _FRAME_SRC_PATTERN.finditer(html):
            child = urljoin(page, unescape(match.group(1)).strip())
            if child not in seen_pages:
                to_visit.append(child)
        for link in parse_source_links(html, base_url=page, max_links=200):
            if _has_archive_extension(link) and link not in seen_archives:
                seen_archives.add(link)
                archives.append(link)
    return archives


def resolve_image_archive_url(url: str, fetcher: Callable[[str], str]) -> str | None:
    """Resolve a source landing URL to its downloadable image archive.

    If the URL already points at an archive it is returned as-is; otherwise the
    landing page (and any frames it embeds) is crawled for archive links,
    preferring one whose filename mentions "image" (the images payload, not the
    annotations/lists sidecars)."""
    if _has_archive_extension(url):
        return url
    archives = _collect_archive_links(url, fetcher)
    if not archives:
        return None
    archives.sort(key=lambda link: (0 if "image" in _url_basename(link) else 1, len(link)))
    return archives[0]


def _safe_extract_target(dest_dir: Path, member_name: str) -> Path | None:
    """Resolve an archive member to a path strictly inside ``dest_dir`` (guards
    against ``..`` traversal and absolute-path members)."""
    member_path = Path(member_name)
    if member_path.is_absolute():
        return None
    target = (dest_dir / member_path).resolve()
    try:
        target.relative_to(dest_dir.resolve())
    except ValueError:
        return None
    return target


def extract_image_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_images: int = 200_000,
) -> int:
    """Extract only the image members of a tar/zip archive into ``dest_dir``,
    preserving their relative directory layout (so ``Images/<class>/x.jpg`` maps
    to a class directory). Returns the number of images written."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def _accept(name: str) -> bool:
        return Path(name).suffix.lower() in IMAGE_EXTS

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive:
                if not member.isfile() or not _accept(member.name):
                    continue
                target = _safe_extract_target(dest_dir, member.name)
                if target is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with extracted, target.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                count += 1
                if count >= max_images:
                    break
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not _accept(info.filename):
                    continue
                target = _safe_extract_target(dest_dir, info.filename)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                count += 1
                if count >= max_images:
                    break
    else:
        raise ValueError(f"unsupported_archive_format:{archive_path.name}")
    return count


def stream_download_archive(
    url: str,
    dest_path: Path,
    *,
    max_bytes: int = 5_000_000_000,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream a (potentially large) archive to disk, aborting if it exceeds
    ``max_bytes`` so a mis-inferred source cannot blow up local storage."""
    import requests

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with requests.get(
        url,
        stream=True,
        headers={"User-Agent": "impulse-ees-source-cache/1.0"},
        timeout=120,
    ) as response:
        response.raise_for_status()
        with dest_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("archive_exceeds_max_bytes")
                handle.write(chunk)
    return dest_path


def _extracted_image_count(corpus_dir: Path) -> int:
    """Count class-labelled images already present under ``corpus_dir`` (an
    image whose immediate parent is a subdirectory, not the corpus root)."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return 0
    count = 0
    for path in corpus_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parent = path.parent
        if parent == corpus_dir or parent.name.startswith(".") or parent.name == ".downloads":
            continue
        count += 1
    return count


def acquire_image_source_corpus(
    evidence: TaskEvidence,
    cache_dir: Path,
    *,
    landing_fetcher: Callable[[str], str] = fetch_web_source_text,
    archive_downloader: Callable[..., Path] = stream_download_archive,
    max_archive_bytes: int = 5_000_000_000,
    max_corpus_images: int = 200_000,
    max_candidates: int = 4,
) -> SourceCacheBuildResult:
    """Infer, download, and extract a public image source dataset into
    ``cache_dir`` as a class-labelled corpus.

    Mirrors ``acquire_label_source_texts``: evidence-only source inference, a
    cached download (re-runs reuse the extracted corpus and the on-disk
    archive), and an honesty-neutral result -- the image_provenance probe's
    train-precision gate is what decides whether the corpus is trustworthy.
    """
    cache_dir = Path(cache_dir)
    candidates = infer_image_source_candidates(evidence.description)
    if not candidates:
        return SourceCacheBuildResult(
            status="skipped",
            sources_written=0,
            manifest_path=cache_dir,
            notes=["no_image_source_inferred"],
        )

    # Cache hit: a previous run already extracted a class-labelled corpus here.
    already = _extracted_image_count(cache_dir)
    if already > 0:
        return SourceCacheBuildResult(
            status="success",
            sources_written=already,
            manifest_path=cache_dir,
            notes=["image_corpus_cache_hit", f"images={already}"],
            labels_with_sources=len([p for p in cache_dir.iterdir() if p.is_dir()]),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = cache_dir / ".downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for candidate in candidates[:max_candidates]:
        try:
            archive_url = resolve_image_archive_url(candidate.url, landing_fetcher)
        except Exception as exc:
            errors.append(f"{candidate.url}: resolve: {type(exc).__name__}: {exc}")
            continue
        if not archive_url:
            errors.append(f"{candidate.url}: no_archive_link_found")
            continue
        archive_name = _url_basename(archive_url) or f"{_slugify(candidate.name)}.tar"
        archive_path = downloads_dir / archive_name
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            try:
                archive_downloader(archive_url, archive_path, max_bytes=max_archive_bytes)
            except Exception as exc:
                errors.append(f"{archive_url}: download: {type(exc).__name__}: {exc}")
                if archive_path.exists():
                    archive_path.unlink()
                continue
        try:
            extracted = extract_image_archive(
                archive_path, cache_dir, max_images=max_corpus_images
            )
        except Exception as exc:
            errors.append(f"{archive_url}: extract: {type(exc).__name__}: {exc}")
            continue
        if extracted > 0:
            class_dirs = [
                path
                for path in cache_dir.rglob("*")
                if path.is_dir() and not path.name.startswith(".")
            ]
            return SourceCacheBuildResult(
                status="success",
                sources_written=extracted,
                manifest_path=cache_dir,
                notes=[
                    "image_source_corpus_acquired",
                    f"source={candidate.name}",
                    f"archive_url={archive_url}",
                    f"images={extracted}",
                ],
                error="; ".join(errors) if errors else None,
                labels_with_sources=len(class_dirs),
            )
        errors.append(f"{archive_url}: no_images_in_archive")

    return SourceCacheBuildResult(
        status="failed" if errors else "skipped",
        sources_written=0,
        manifest_path=cache_dir,
        notes=["no_image_source_corpus_acquired"],
        error="; ".join(errors) if errors else None,
    )


_PUBLIC_DOMAIN_TEXT_HOSTS = ("gutenberg.org", "archive.org", "wikisource.org")
_GUTENBERG_ID_PATTERNS = (
    re.compile(r"/ebooks/(\d+)", flags=re.IGNORECASE),
    re.compile(r"/cache/epub/(\d+)", flags=re.IGNORECASE),
    re.compile(r"/files/(\d+)", flags=re.IGNORECASE),
)


def _public_domain_host_rank(url: str) -> int:
    netloc = urlparse(url).netloc.lower()
    for rank, host in enumerate(_PUBLIC_DOMAIN_TEXT_HOSTS):
        if netloc == host or netloc.endswith("." + host):
            return rank
    return len(_PUBLIC_DOMAIN_TEXT_HOSTS)


def expand_gutenberg_text_url(url: str) -> str:
    """Rewrite gutenberg.org ebook/landing URLs to their plain-text asset."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc != "gutenberg.org" and not netloc.endswith(".gutenberg.org"):
        return url
    if parsed.path.lower().endswith(".txt"):
        return url
    for pattern in _GUTENBERG_ID_PATTERNS:
        match = pattern.search(parsed.path)
        if match:
            ebook_id = match.group(1)
            return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
    return url


def build_label_source_queries(alias: str) -> list[str]:
    alias = alias.strip()
    return [
        f'"{alias}" full text',
        f'"{alias}" complete works plain text',
        f'"{alias}" site:gutenberg.org',
    ]


def _search_provider_query(
    provider: WebResearchProvider,
    query: str,
    *,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Issue a raw-query search through whatever the provider supports."""
    search_query = getattr(provider, "search_query", None)
    if callable(search_query):
        return [dict(result) for result in search_query(query) if isinstance(result, dict)]
    query_fetcher = getattr(provider, "fetcher", None)
    if callable(query_fetcher):
        return parse_duckduckgo_html(query_fetcher(query), max_results=max_results)
    return []


_BINARY_MAGIC_PREFIXES = ("%PDF", "PK\x03\x04", "GIF8", "\x89PNG", "PNG\r\n")
_PLAIN_TEXT_GOOD_CHARS = set(".,;:'\"!?()-")


def _is_plausible_plain_text(
    raw: str,
    cleaned: str,
    *,
    sample_chars: int = 50000,
    min_ratio: float = 0.85,
) -> bool:
    """Reject binary payloads (PDF/zip/image magics) and OCR/stream garbage.

    A corpus source must read like prose: over the first ``sample_chars``
    characters, ASCII letters, whitespace, and basic punctuation must dominate.
    """
    head = raw.lstrip()[:8]
    if any(head.startswith(magic) for magic in _BINARY_MAGIC_PREFIXES):
        return False
    sample = cleaned[:sample_chars]
    if not sample:
        return False
    good = sum(
        1
        for char in sample
        if char.isascii() and (char.isalpha() or char.isspace() or char in _PLAIN_TEXT_GOOD_CHARS)
    )
    return good / len(sample) >= min_ratio


def fetch_gutendex_catalog(search: str) -> dict:
    """Query the Project Gutenberg catalog (gutendex.com) for an author/work."""
    import requests

    response = requests.get(
        "https://gutendex.com/books",
        params={"search": search},
        headers={"User-Agent": "impulse-ees-source-cache/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _gutendex_candidates(
    alias: str,
    catalog_fetcher: Callable[[str], dict],
    errors: list[str],
) -> list[dict[str, str]]:
    """Catalog-first source discovery: every text-format Gutenberg book whose
    author matches the alias surname, ranked by download_count.

    Breadth beats precision here — matching against the corpus is cheap, and
    the provenance probe needs coverage across an author's works, so no
    title filtering (poetry vs prose is unknowable in general).
    """
    surname = alias.strip().split()[-1].lower() if alias.strip() else ""
    if not surname:
        return []
    try:
        payload = catalog_fetcher(alias)
    except Exception as exc:
        errors.append(f"gutendex:{alias}: {type(exc).__name__}: {exc}")
        return []
    books = payload.get("results", []) if isinstance(payload, dict) else []
    matching = []
    for book in books:
        if not isinstance(book, dict):
            continue
        book_id = book.get("id")
        if not str(book_id).isdigit():
            continue
        authors = book.get("authors") or []
        names = " ".join(
            str(author.get("name", "")) for author in authors if isinstance(author, dict)
        ).lower()
        if surname not in names:
            continue
        formats = book.get("formats") or {}
        if not any(str(key).lower().startswith("text/plain") for key in formats):
            continue
        matching.append(book)
    matching.sort(key=lambda book: -(book.get("download_count") or 0))
    return [
        {
            "title": str(book.get("title") or f"gutenberg-{book.get('id')}"),
            "url": f"https://www.gutenberg.org/cache/epub/{book.get('id')}/pg{book.get('id')}.txt",
            "snippet": "gutendex_catalog",
        }
        for book in matching
    ]


def _ddg_candidates(
    alias: str,
    provider: WebResearchProvider,
    errors: list[str],
) -> list[dict[str, str]]:
    """Fallback source discovery via generic web search, public-domain hosts first."""
    candidates: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for query in build_label_source_queries(alias):
        try:
            results = _search_provider_query(provider, query)
        except Exception as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
            continue
        for source in results:
            url = expand_gutenberg_text_url(str(source.get("url", "")).strip())
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            candidate = dict(source)
            candidate["url"] = url
            candidates.append(candidate)
    candidates.sort(key=lambda source: _public_domain_host_rank(source["url"]))
    return candidates


def acquire_label_source_texts(
    evidence: TaskEvidence,
    cache_dir: Path,
    *,
    provider: WebResearchProvider,
    fetcher: Callable[[str], str] = fetch_web_source_text,
    catalog_fetcher: Callable[[str], dict] = fetch_gutendex_catalog,
    max_sources_per_label: int = 8,
    min_chars: int = 20000,
    max_fetches_per_label: int = 10,
) -> SourceCacheBuildResult:
    """Search for the class labels' SOURCE TEXTS (not competition writeups) and cache them.

    General, evidence-only: it fires only when ``infer_label_aliases`` can map
    abbreviated labels to named entities from the task description. Per-label
    files land in ``cache_dir/<label>/source<k>.txt`` plus a
    ``source_manifest.csv`` in the ``author,local_path`` format that
    ``text_provenance._iter_corpus_sources`` reads.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "source_manifest.csv"
    labels = _candidate_labels(evidence)
    try:
        from bench.ees_core.operators.text_provenance import infer_label_aliases

        aliases = infer_label_aliases(evidence.description, labels)
    except Exception:
        aliases = {}
    aliases = {label: names for label, names in aliases.items() if names}
    if not aliases:
        return SourceCacheBuildResult(
            status="skipped",
            sources_written=0,
            manifest_path=manifest_path,
            notes=["no_label_aliases_inferred"],
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    written_urls: set[str] = set()
    sources_per_label: dict[str, int] = {}
    for label in labels:
        label_aliases = aliases.get(label)
        if not label_aliases:
            continue
        alias = label_aliases[0]
        candidates = _gutendex_candidates(alias, catalog_fetcher, errors)
        if not candidates:
            candidates = _ddg_candidates(alias, provider, errors)
        written = 0
        label_fetches = 0
        for source in candidates:
            if written >= max_sources_per_label or label_fetches >= max_fetches_per_label:
                break
            url = source["url"]
            if url in written_urls:
                continue
            label_fetches += 1
            try:
                raw = fetcher(url)
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                continue
            text = _clean_source_text(raw)
            if not _is_plausible_plain_text(raw, text):
                errors.append(f"{url}: rejected_non_plain_text")
                continue
            if len(text) < min_chars:
                continue
            label_dir = cache_dir / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            written += 1
            local_path = label_dir / f"source{written}.txt"
            local_path.write_text(text + "\n", encoding="utf-8")
            written_urls.add(url)
            rows.append(
                {
                    "author": str(label),
                    "local_path": str(local_path),
                    "url": url,
                    "title": str(source.get("title", "") or url),
                }
            )
        sources_per_label[str(label)] = written

    labels_with_aliases = len(aliases)
    labels_with_sources = sum(1 for count in sources_per_label.values() if count > 0)
    per_label_notes = [f"{label}_sources={count}" for label, count in sources_per_label.items()]
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["author", "local_path", "url", "title"])
            writer.writeheader()
            writer.writerows(rows)
        return SourceCacheBuildResult(
            status="success",
            sources_written=len(rows),
            manifest_path=manifest_path,
            notes=["label_source_texts_acquired", *per_label_notes],
            error="; ".join(errors) if errors else None,
            labels_with_aliases=labels_with_aliases,
            labels_with_sources=labels_with_sources,
        )
    return SourceCacheBuildResult(
        status="skipped" if not errors else "failed",
        sources_written=0,
        manifest_path=manifest_path,
        notes=["no_label_sources_acquired", *per_label_notes],
        error="; ".join(errors) if errors else None,
        labels_with_aliases=labels_with_aliases,
        labels_with_sources=labels_with_sources,
    )


def _decode_duckduckgo_url(url: str) -> str:
    url = unescape(url)
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def parse_duckduckgo_html(html: str, *, max_results: int = 5) -> list[dict[str, str]]:
    result_pattern = re.compile(
        r"<a[^>]+class=[\"'][^\"']*result__a[^\"']*[\"'][^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = [
        _clean_html_text(match)
        for match in re.findall(
            r"<a[^>]+class=[\"'][^\"']*result__snippet[^\"']*[\"'][^>]*>(.*?)</a>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    sources: list[dict[str, str]] = []
    for index, match in enumerate(result_pattern.finditer(html)):
        url = _decode_duckduckgo_url(match.group(1))
        title = _clean_html_text(match.group(2))
        if not url or not title:
            continue
        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippets[index] if index < len(snippets) else "",
            }
        )
        if len(sources) >= max_results:
            break
    return sources


def _duckduckgo_fetch(query: str) -> str:
    import requests

    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "impulse-ees-web-research/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


@dataclass(frozen=True)
class DuckDuckGoHtmlWebResearchProvider:
    fetcher: Callable[[str], str] = _duckduckgo_fetch
    max_queries: int = 3
    max_results_per_query: int = 5

    def search_query(self, query: str) -> list[dict[str, str]]:
        html = self.fetcher(query)
        return parse_duckduckgo_html(html, max_results=self.max_results_per_query)

    def search(
        self,
        evidence: TaskEvidence,
        hypothesis: StrategyHypothesis,
    ) -> WebResearchResult:
        queries = build_research_queries(evidence, hypothesis, max_queries=self.max_queries)
        sources: list[dict[str, str]] = []
        errors: list[str] = []
        for query in queries:
            try:
                html = self.fetcher(query)
                sources.extend(parse_duckduckgo_html(html, max_results=self.max_results_per_query))
            except Exception as exc:
                errors.append(f"{query}: {type(exc).__name__}: {exc}")
        deduped_sources = list({source["url"]: source for source in sources}.values())
        if deduped_sources:
            return WebResearchResult(
                status="success",
                query_count=len(queries),
                sources=deduped_sources,
                notes=["duckduckgo_html"],
                error=None,
            )
        return WebResearchResult(
            status="failed" if errors else "skipped",
            query_count=len(queries),
            sources=[],
            notes=["duckduckgo_html_no_sources"],
            error="; ".join(errors) if errors else None,
        )


def run_web_research_probe(
    provider: WebResearchProvider,
    evidence: TaskEvidence,
    hypothesis: StrategyHypothesis,
    output_dir: Path,
) -> StrategyProbeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    research = provider.search(evidence, hypothesis)
    manifest_path = output_dir / "web_research_manifest.json"
    manifest = {
        "task_id": evidence.task_id,
        "strategy_id": hypothesis.strategy_id,
        "research": research.to_json(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    promote = research.status == "success" and bool(research.sources)
    rationale = research.error or "; ".join(research.notes) or research.status
    return StrategyProbeResult(
        strategy_id=hypothesis.strategy_id,
        status=research.status,
        promote=promote,
        evidence_score=1.0 if promote else 0.0,
        coverage=None,
        precision_estimate=None,
        rationale=rationale,
        risks=[] if promote else ["no_web_sources"],
        artifact_paths=[str(manifest_path)],
    )
