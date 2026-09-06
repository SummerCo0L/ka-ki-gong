#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nearest Social Service Office (SSO) lookup by free-text area name.

Data source: ``knowledge_base/social_service_offices/sso_locations.json``,
transcribed from MSF's "Social Service Office Contact List" (Social Support
Group, updated August 2026).

This is a location-matching problem, not a semantic search problem, so it is
deliberately kept separate from the embedding-based ``SchemeIndex``: a user
says a town/neighbourhood name, and we match it against the 24 SSO areas
(exact match) or a best-effort alias table of nearby neighbourhoods that are
not themselves home to an SSO (approximate match, lower confidence).

The alias table is a rough geographic approximation, not an official MSF
catchment-area mapping — callers should always be offered the ComCare
Hotline (1800-222-0000) to confirm, especially for approximate matches.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

COMCARE_HOTLINE = "1800-222-0000"

# Best-effort mapping of neighbourhoods/MRT names not themselves covered by an
# SSO to the geographically nearest SSO area. Deliberately approximate.
_AREA_ALIASES: dict[str, str] = {
    "amk": "Ang Mo Kio",
    "yio chu kang": "Ang Mo Kio",
    "thomson": "Ang Mo Kio",
    "upper thomson": "Ang Mo Kio",
    "tanah merah": "Bedok",
    "kembangan": "Bedok",
    "simei": "Tampines",
    "changi": "Pasir Ris",
    "loyang": "Pasir Ris",
    "jurong west": "Boon Lay",
    "pioneer": "Boon Lay",
    "tuas": "Boon Lay",
    "bukit gombak": "Bukit Batok",
    "hillview": "Bukit Batok",
    "tiong bahru": "Bukit Merah",
    "redhill": "Bukit Merah",
    "telok blangah": "Bukit Merah",
    "harbourfront": "Bukit Merah",
    "alexandra": "Bukit Merah",
    "bukit timah": "Bukit Panjang",
    "fajar": "Bukit Panjang",
    "segar": "Bukit Panjang",
    "west coast": "Clementi",
    "dover": "Queenstown",
    "ghim moh": "Queenstown",
    "holland village": "Queenstown",
    "buona vista": "Queenstown",
    "tanglin": "Queenstown",
    "choa chu kang": "Chua Chu Kang",  # alternate spelling
    "cck": "Chua Chu Kang",
    "geylang": "Geylang Serai",
    "paya lebar": "Geylang Serai",
    "eunos": "Geylang Serai",
    "marine parade": "Geylang Serai",
    "katong": "Geylang Serai",
    "joo chiat": "Geylang Serai",
    "ubi": "Geylang Serai",
    "macpherson": "Geylang Serai",
    "aljunied": "Geylang Serai",
    "buangkok": "Hougang",
    "kovan": "Hougang",
    "bugis": "Jalan Besar",
    "kampong glam": "Jalan Besar",
    "little india": "Jalan Besar",
    "rochor": "Jalan Besar",
    "kallang": "Jalan Besar",
    "whampoa": "Jalan Besar",
    "chinatown": "Kreta Ayer",
    "outram": "Kreta Ayer",
    "tanjong pagar": "Kreta Ayer",
    "cbd": "Kreta Ayer",
    "raffles place": "Kreta Ayer",
    "marina bay": "Kreta Ayer",
    "downtown": "Kreta Ayer",
    "potong pasir": "Toa Payoh",
    "bishan": "Toa Payoh",
    "novena": "Toa Payoh",
    "newton": "Toa Payoh",
    "marsiling": "Woodlands",
    "admiralty": "Woodlands",
    "bukit canberra": "Sembawang",
    "canberra": "Sembawang",
    "serangoon gardens": "Serangoon",
}


@dataclass
class SSOLocation:
    """One Social Service Office record."""

    area: str
    name: str
    address: str | None
    postal_code: str | None
    remote_only: bool
    remarks: str | None


@dataclass
class SSOMatch:
    """Result of a nearest-SSO lookup."""

    location: SSOLocation | None
    confidence: str
    """One of "high" (exact area name match), "approximate" (matched via a
    nearby-neighbourhood alias), or "none" (no match found)."""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


class SSODirectory:
    """In-memory lookup of SSOs by free-text area name."""

    def __init__(self, locations: list[SSOLocation]):
        self._locations = locations
        self._by_area = {_normalize(loc.area): loc for loc in locations}

    @classmethod
    def load(cls, path: Path | str) -> "SSODirectory":
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        locations = [
            SSOLocation(
                area=entry["area"],
                name=entry["name"],
                address=entry.get("address"),
                postal_code=entry.get("postal_code"),
                remote_only=bool(entry.get("remote_only", False)),
                remarks=entry.get("remarks"),
            )
            for entry in entries
        ]
        return cls(locations)

    def find_nearest(self, area_text: str) -> SSOMatch:
        """Find the SSO nearest to a free-text area/neighbourhood description.

        Args:
            area_text: The area, town, or neighbourhood the user mentioned
                (e.g. "Toa Payoh", "near Bishan", "AMK").

        Returns:
            An ``SSOMatch`` with the best guess and a confidence level.
        """
        normalized = _normalize(area_text)
        if not normalized:
            return SSOMatch(location=None, confidence="none")

        # Exact area match, e.g. "toa payoh" in "i stay near toa payoh".
        for area_key, location in self._by_area.items():
            if area_key and area_key in normalized:
                return SSOMatch(location=location, confidence="high")

        # Alias match against nearby neighbourhoods not covered by name.
        for alias, canonical_area in _AREA_ALIASES.items():
            if alias in normalized:
                location = self._by_area.get(_normalize(canonical_area))
                if location:
                    return SSOMatch(location=location, confidence="approximate")

        return SSOMatch(location=None, confidence="none")
