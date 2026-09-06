#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Optional ``find_nearest_sso`` follow-up tool for the grounded-answer LLM.

Lets the LLM look up the Social Service Office (SSO) nearest to an area the
user mentions. See ``retrieval/sso_lookup.py`` for the matching logic and its
approximate-match caveats.

Same caveat applies here as elsewhere: Pipecat's CHANGELOG documents a broken-
streaming-tool-call bug specific to ``openai/gpt-oss-120b`` on at least one
OpenRouter upstream provider. This tool is gated behind its own
``ENABLE_SSO_LOOKUP_TOOL`` flag (default on) so it can be disabled
independently if that bug is hit.
"""

from __future__ import annotations

from retrieval.sso_lookup import COMCARE_HOTLINE, SSODirectory


def format_sso_tool_result(match) -> dict:
    """Shape an ``SSOMatch`` into the tool-result JSON returned to the LLM."""
    if match.location is None:
        return {
            "matched": False,
            "confidence": "none",
            "comcare_hotline": COMCARE_HOTLINE,
            "message": (
                "No Social Service Office could be matched to that area. Tell "
                f"the user to call the ComCare Hotline {COMCARE_HOTLINE} to find "
                "the SSO nearest to them."
            ),
        }

    location = match.location
    result = {
        "matched": True,
        "confidence": match.confidence,
        "sso_name": location.name,
        "address": location.address,
        "remote_only": location.remote_only,
        "remarks": location.remarks,
        "comcare_hotline": COMCARE_HOTLINE,
    }
    if match.confidence == "approximate":
        result["message"] = (
            f"{location.name} is the closest match on file, but this is an "
            "approximate neighbourhood match, not an official catchment area. "
            f"Tell the user this and also give the ComCare Hotline {COMCARE_HOTLINE} "
            "so they can confirm the nearest SSO."
        )
    elif location.remote_only:
        result["message"] = (
            f"{location.name} does not accept walk-ins — relay its remarks "
            "field to the user plainly, including the ComCare Hotline."
        )
    else:
        result["message"] = (
            f"{location.name} is a confident match. Give its name and address."
        )
    return result


def make_find_nearest_sso_tool(directory: SSODirectory):
    """Build the direct-function tool handler for nearest-SSO lookup.

    Returns an async function suitable for ``LLMContext(tools=[...])`` — Pipecat
    derives the tool's JSON schema from this function's signature and docstring.
    """

    async def find_nearest_sso(params, area: str):
        """Find the Social Service Office (SSO) nearest to an area in Singapore.

        Use this when the user mentions needing help from a Social Service
        Office, ComCare, or asks where to go in person, and has stated or
        been asked which town/neighbourhood they stay in.

        Args:
            area: The town, neighbourhood, or MRT station name the user
                mentioned (e.g. "Toa Payoh", "near Bishan", "AMK").
        """
        match = directory.find_nearest(area)
        await params.result_callback(format_sso_tool_result(match))

    return find_nearest_sso
