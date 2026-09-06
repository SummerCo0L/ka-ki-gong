"""Scrape all SupportGoWhere topics' schemes/services into a local knowledge base.

Data source: SupportGoWhere (supportgowhere.life.gov.sg), a Singapore Government
website. This script calls two of its public, unauthenticated JSON APIs:

1. List API (found via a topic page's embedded widget, e.g. "Caregiving support"):
     https://www.ccube.gov.sg/coordinator/api/resources/v1
   Returns every scheme/service ("resource") tagged under a domain, e.g.
   CAREGIVING_SUPPORT, with its id/type/title/short description.

2. Detail API (found via the scheme/service detail pages):
     https://api.supportgowhere.life.gov.sg/v2/sr/{schemes|services}/{id}
   Returns the full record, including rich-text fields (About this support,
   Who is this for, How to apply, etc.) encoded as TipTap ProseMirror JSON docs.

One resource (e.g. a scheme) commonly belongs to several topics (domains), so it
will appear in multiple output folders below - this mirrors the site's own
categorisation. Detail fetches are cached across topics to avoid duplicate requests.

Usage:
    python scripts/scrape_supportgowhere_schemes.py

Output (relative to repo root), one folder per topic:
    knowledge_base/<topic_slug>/schemes.json               - one combined JSON array
    knowledge_base/<topic_slug>/md/<title-slug>_<id>.md    - one markdown file per scheme/service
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Maps each topic's domain friendlyId (as used by the list API) to the output
# folder slug. "Budget 2026" is shown alongside these on the site's homepage but
# links to a calculator page, not a resource-listing topic, so it's excluded.
DOMAINS = {
    "CAREGIVING_SUPPORT": "caregiving_support",
    "CITIZENSHIP_RESIDENCY": "citizenship_residency",
    "COUNSELLING_CRISIS": "counselling_crisis",
    "DISABILITY_SUPPORT": "disability_support",
    "EDUCATION_LEARNING": "education_learning",
    "FAMILY_PARENTING": "family_parenting",
    "FINANCIAL_SUPPORT": "financial_support",
    "HEALTHCARE": "healthcare",
    "HOUSING_SHELTERS": "housing_shelters",
    "MENTAL_HEALTH": "mental_health",
    "RETIREMENT_LEGACY": "retirement_legacy",
    "WORK_EMPLOYMENT": "work_employment",
}

LIST_API = "https://www.ccube.gov.sg/coordinator/api/resources/v1"
DETAIL_API = "https://api.supportgowhere.life.gov.sg/v2/sr"
USER_AGENT = "voice-agents-tracing-kb-scraper/1.0 (personal research; contact via repo)"
REQUEST_DELAY_SECONDS = 0.3

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base"

# Human-friendly labels for the rich-text fields we know about. Any other
# TipTap field we encounter is still rendered, just with its raw key as heading.
FIELD_LABELS = {
    "aboutThisSupport": "About this support",
    "whoIsThisFor": "Who is this for?",
    "howToApply": "How to apply?",
    "whatToExpect": "What to expect?",
    "address": "Where to go",
    "agencyWebsite": "Agency website",
    "emailOrForm": "Email / Form",
    "phoneNumber": "Phone number",
    "eligibility": "Eligibility",
}


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_resource_list(domain: str) -> list[dict]:
    url = f"{LIST_API}?domains={domain}&lang=en&serviceId=sgw-cms&limit=100000"
    data = fetch_json(url)
    return data["data"]["resources"]


def fetch_detail(resource_type: str, resource_id: str) -> dict:
    plural = "schemes" if resource_type == "scheme" else "services"
    encoded_id = urllib.parse.quote(resource_id, safe="")
    url = f"{DETAIL_API}/{plural}/{encoded_id}?lang=en"
    return fetch_json(url)


def _decode_href(href: str) -> str:
    """Unwrap supportgowhere.life.gov.sg/track?encodedUrl=... redirect links."""
    try:
        parsed = urllib.parse.urlparse(href)
        if parsed.path == "/track":
            qs = urllib.parse.parse_qs(parsed.query)
            if "encodedUrl" in qs:
                return urllib.parse.unquote(qs["encodedUrl"][0])
    except Exception:
        pass
    return href


def _render_inline(nodes: list[dict]) -> str:
    parts = []
    for node in nodes:
        ntype = node.get("type")
        if ntype == "text":
            text = node.get("text", "")
            marks = {m.get("type") for m in node.get("marks", [])}
            link_mark = next((m for m in node.get("marks", []) if m.get("type") == "link"), None)
            if "bold" in marks:
                text = f"**{text}**"
            if link_mark:
                href = _decode_href(link_mark.get("attrs", {}).get("href", ""))
                if href and href != text:
                    text = f"[{text}]({href})"
            parts.append(text)
        elif ntype == "hardBreak":
            parts.append("\n")
        else:
            parts.append(_render_inline(node.get("content", []) or []))
    return "".join(parts)


def _render_block(node: dict, indent: int = 0) -> str:
    ntype = node.get("type")
    content = node.get("content", []) or []
    prefix = "  " * indent

    if ntype == "paragraph":
        text = _render_inline(content).strip()
        return f"{prefix}{text}" if text else ""
    if ntype in ("bulletList", "orderedList"):
        lines = []
        for i, item in enumerate(content, start=1):
            marker = "-" if ntype == "bulletList" else f"{i}."
            item_lines = [
                _render_block(child, indent + 1) for child in item.get("content", []) or []
            ]
            item_lines = [line for line in item_lines if line]
            if item_lines:
                first, *rest = item_lines
                first = f"{prefix}{marker} {first.strip()}"
                lines.append("\n".join([first, *rest]))
        return "\n".join(lines)
    if ntype == "listItem":
        return "\n".join(_render_block(child, indent) for child in content)
    # Fallback: recurse into unknown block types (e.g. blockquote, heading).
    return "\n".join(filter(None, (_render_block(child, indent) for child in content)))


def tiptap_to_markdown(raw: str) -> str:
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw)
    blocks = doc.get("content", []) or []
    rendered = [_render_block(b) for b in blocks]
    return "\n\n".join(line for line in rendered if line).strip()


def _simplify_ref_list(items: list[dict] | None) -> list[str]:
    if not items:
        return []
    return [item.get("name_en") or item.get("name") for item in items if item]


def normalize_detail(resource_type: str, summary: dict, detail: dict) -> dict:
    normalized = {
        "id": summary["id"],
        "type": resource_type,
        "title": detail.get("title") or detail.get("title_en") or summary.get("title"),
        "description": detail.get("description") or summary.get("description"),
        "url": f"https://supportgowhere.life.gov.sg/{'schemes' if resource_type == 'scheme' else 'services'}/{summary['id']}",
        "keywords": summary.get("keywords") or [k.get("name_en") for k in detail.get("keywordsV2", [])],
        "domains": _simplify_ref_list(detail.get("domains")),
        "profiles": _simplify_ref_list(detail.get("profiles")),
        "highlights": detail.get("highlights") or [],
        "citizenship": detail.get("citizenship"),
        "min_age": detail.get("minAge"),
        "max_age": detail.get("maxAge"),
        "cost": detail.get("cost"),
        "info_last_updated": detail.get("infoLastUpdated"),
        "sections": {},
    }
    for key, label in FIELD_LABELS.items():
        raw = detail.get(key)
        if raw:
            normalized["sections"][label] = tiptap_to_markdown(raw)
    return normalized


def to_markdown_doc(entry: dict) -> str:
    lines = [f"# {entry['title']}", ""]
    lines.append(f"*Source: {entry['url']}*")
    lines.append("")
    if entry.get("description"):
        lines.append(entry["description"].strip())
        lines.append("")
    if entry.get("highlights"):
        lines.append("## Highlights")
        lines.extend(f"- {h}" for h in entry["highlights"])
        lines.append("")
    meta_bits = []
    if entry.get("profiles"):
        meta_bits.append(f"**Who it targets:** {', '.join(entry['profiles'])}")
    if entry.get("citizenship"):
        meta_bits.append(f"**Citizenship:** {', '.join(entry['citizenship'])}")
    if entry.get("keywords"):
        meta_bits.append(f"**Keywords:** {', '.join(entry['keywords'])}")
    if meta_bits:
        lines.extend(meta_bits)
        lines.append("")
    for label, text in entry["sections"].items():
        lines.append(f"## {label}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def slugify(resource_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", resource_id)


def slugify_title(title: str, max_length: int = 60) -> str:
    """Turn a title into a filename-safe slug (site's resource ids are opaque, e.g. 'Sucth2ke')."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or "untitled"


def main() -> None:
    detail_cache: dict[tuple[str, str], dict] = {}
    total_placements = 0
    total_fetch_failures = 0

    for domain, slug in DOMAINS.items():
        output_dir = OUTPUT_ROOT / slug
        md_dir = output_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)
        for stale in md_dir.glob("*.md"):
            stale.unlink()

        print(f"\n=== {domain} -> {slug} ===")
        summaries = fetch_resource_list(domain)
        print(f"Found {len(summaries)} resources.")

        entries = []
        for i, summary in enumerate(summaries, start=1):
            rtype = summary["type"]  # "scheme" or "service"
            rid = summary["id"]
            cache_key = (rtype, rid)

            if cache_key in detail_cache:
                detail = detail_cache[cache_key]
            else:
                print(f"[{i}/{len(summaries)}] Fetching detail for {rtype}:{rid} ...")
                try:
                    detail = fetch_detail(rtype, rid)
                except Exception as exc:  # noqa: BLE001 - log and continue scraping the rest
                    print(f"  ! Failed to fetch {rtype}:{rid}: {exc}")
                    total_fetch_failures += 1
                    continue
                detail_cache[cache_key] = detail
                time.sleep(REQUEST_DELAY_SECONDS)

            entry = normalize_detail(rtype, summary, detail)
            entries.append(entry)

            title_slug = slugify_title(entry.get("title") or rid)
            md_path = md_dir / f"{title_slug}_{slugify(rid)}.md"
            md_path.write_text(to_markdown_doc(entry), encoding="utf-8")

        json_path = output_dir / "schemes.json"
        json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        total_placements += len(entries)
        print(f"Wrote {len(entries)} entries to {json_path}")

    print("\nDone.")
    print(f"  {len(detail_cache)} unique schemes/services fetched")
    print(f"  {total_placements} total placements across {len(DOMAINS)} topic folders")
    if total_fetch_failures:
        print(f"  {total_fetch_failures} fetch failures (see log above)")


if __name__ == "__main__":
    main()
