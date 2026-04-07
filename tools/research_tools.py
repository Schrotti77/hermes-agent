#!/usr/bin/env python3
"""
Research Tools — Hermes Agent tool wrapper for research_hub.

Provides: research_tool (unified multi-engine search + deep extraction)

This module is imported by model_tools.py → registry → available to all agents.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add autonomy path so we can import research_hub
_AUTONOMY_PATH = Path.home() / ".hermes" / "autonomy"
if str(_AUTONOMY_PATH) not in sys.path:
    sys.path.insert(0, str(_AUTONOMY_PATH))

from tools.registry import registry

logger = logging.getLogger(__name__)

# ── Lazy-loaded research hub ───────────────────────────────────────────────────

_research_hub = None


def _get_research_hub():
    """Lazy import to avoid loading research_hub unless tool is actually called."""
    global _research_hub
    if _research_hub is None:
        try:
            from autonomy.research_hub import research, format_research
            _research_hub = {"research": research, "format": format_research}
            logger.debug("research_hub loaded successfully")
        except Exception as e:
            logger.error("Failed to import research_hub: %s", e)
            return None
    return _research_hub


# ── Tool Handlers ──────────────────────────────────────────────────────────────

def research_tool_handler(args: dict, task_id: str = None) -> str:
    """
    Unified research tool — multi-engine search with automatic routing.

    Args:
        query: Search/research query (required)
        engines: Comma-separated engine names (optional, auto-routes if omitted)
               Available: whoogle, ddgs, github, arxiv, crawl4ai, firecrawl, tavily
        max_results: Results per engine (default: 8)
        save: Whether to save results to research_memory.db (default: true)

    Returns:
        Formatted markdown with research results.
    """
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"success": False, "error": "query is required"})

    engines_str = args.get("engines", "")
    engines = [e.strip() for e in engines_str.split(",") if e.strip()] if engines_str else None
    max_results = min(int(args.get("max_results", 8)), 20)
    save = args.get("save", "true").lower() != "false"

    hub = _get_research_hub()
    if not hub:
        return json.dumps({
            "success": False,
            "error": "research_hub not available. Check research-venv installation."
        })

    try:
        rh = hub["research"]
        result = rh(
            query=query,
            engines=engines,
            max_results_per_engine=max_results,
            save_to_memory=save,
        )
        formatted = hub["format"](result)
        return json.dumps({
            "success": True,
            "query": query,
            "classification": result.get("classification"),
            "primary_type": result.get("primary_type"),
            "engines_used": result.get("engines_used", []),
            "total_results": result.get("total_results", 0),
            "elapsed_s": result.get("elapsed"),
            "formatted": formatted,
            "results": result.get("results", [])[:10],  # Raw results for agents
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("research_tool failed")
        return json.dumps({"success": False, "error": str(e)})


def research_save_handler(args: dict, task_id: str = None) -> str:
    """
    Manually save a finding to research_memory.db.

    Args:
        topic: Short topic name (required)
        answer: The finding/content (required)
        question: Original question (optional)
        sources: JSON list of {url, title} (optional)
        tags: JSON list of string tags (optional)
        quality: high/medium/low (default: medium)
    """
    topic = args.get("topic", "").strip()
    answer = args.get("answer", "").strip()
    if not topic or not answer:
        return json.dumps({"success": False, "error": "topic and answer are required"})

    try:
        from autonomy.research_saver import save_research
        sources = args.get("sources", "[]")
        if isinstance(sources, str):
            sources = json.loads(sources)
        tags = args.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)
        fid = save_research(
            topic=topic,
            question=args.get("question", ""),
            answer=answer,
            sources=sources or [],
            tags=tags or [],
            quality=args.get("quality", "medium"),
            role=args.get("role", "manual"),
        )
        return json.dumps({"success": True, "finding_id": fid})
    except Exception as e:
        logger.exception("research_save failed")
        return json.dumps({"success": False, "error": str(e)})


def research_recall_handler(args: dict, task_id: str = None) -> str:
    """
    Search saved research findings from research_memory.db.

    Args:
        query: Search query (required)
        role: Filter by role (optional, e.g. "research", "backend", "manual")
        limit: Max results (default: 10)
    """
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"success": False, "error": "query is required"})

    try:
        from autonomy.research_memory import search, format_results
        role = args.get("role") or None
        limit = min(int(args.get("limit", 10)), 50)
        results = search(query, role=role, limit=limit)
        formatted = format_results(results, query)
        return json.dumps({
            "success": True,
            "query": query,
            "count": len(results),
            "formatted": formatted,
            "results": results,
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("research_recall failed")
        return json.dumps({"success": False, "error": str(e)})


# ── Registry ───────────────────────────────────────────────────────────────────

RESEARCH_VENV = Path.home() / ".hermes" / "research-venv"


def _check_requirements() -> bool:
    """Check if research-venv exists and has required packages."""
    return RESEARCH_VENV.exists()


registry.register(
    name="research",
    toolset="web",
    schema={
        "name": "research",
        "description": """Unified multi-engine research tool — search + deep extraction.

**Available engines:** whoogle (privacy search), ddgs (DuckDuckGo), github (code search),
  arxiv (academic papers), crawl4ai (deep page extraction), firecrawl (API extraction),
  tavily (AI-optimized search).

**Auto-routing:** If no engines specified, automatically selects best engine for query type:
  - code queries → github, whoogle
  - academic → arxiv, whoogle
  - specs/Protocol (BOLT, BLIP, NUT) → whoogle, ddgs, github
  - bitcoin/lightning → whoogle, ddgs
  - deep extraction → crawl4ai, firecrawl

**Examples:**
- research(query="LNURLw service provider specification")
- research(query="github.com/lightningnetwork/lnd RouteDiagnostic", engines="github")
- research(query="https://github.com/lightning/bolts/blob/master/04-onion-routing.md", engines="crawl4ai")
- research(query="mpc wallet construction bitcoin", engines="arxiv,whoogle")
""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or URL to research. For extraction, pass a full URL."
                },
                "engines": {
                    "type": "string",
                    "description": "Comma-separated engine names (optional, auto-routes if omitted). Options: whoogle,ddgs,github,arxiv,crawl4ai,firecrawl,tavily"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results per engine (default: 8, max: 20)"
                },
                "save": {
                    "type": "string",
                    "description": "Save to research_memory.db (true/false, default: true)"
                },
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: research_tool_handler(args, task_id=kw.get("task_id")),
    check_fn=_check_requirements,
    requires_env=["HERMES_HOME"],
)


registry.register(
    name="research_save",
    toolset="web",
    schema={
        "name": "research_save",
        "description": "Manually save a finding to the persistent research memory (research_memory.db). Use this to bookmark important discoveries.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Short topic name (e.g. 'NUT-09', 'ZapOut Architecture')"},
                "answer": {"type": "string", "description": "The finding/content to save"},
                "question": {"type": "string", "description": "Original research question (optional)"},
                "sources": {"type": "string", "description": "JSON list of {url, title} source objects"},
                "tags": {"type": "string", "description": "JSON list of tag strings (e.g. '[\"lightning\",\"spec\"]')"},
                "quality": {"type": "string", "description": "high/medium/low quality rating"},
            },
            "required": ["topic", "answer"],
        },
    },
    handler=lambda args, **kw: research_save_handler(args, task_id=kw.get("task_id")),
    check_fn=_check_requirements,
    requires_env=["HERMES_HOME"],
)


registry.register(
    name="research_recall",
    toolset="web",
    schema={
        "name": "research_recall",
        "description": "Search previously saved research findings from research_memory.db. Use this before starting new research to avoid duplicating work.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "role": {"type": "string", "description": "Filter by role: research, backend, frontend, manual"},
                "limit": {"type": "integer", "description": "Max results (default: 10)"},
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: research_recall_handler(args, task_id=kw.get("task_id")),
    check_fn=_check_requirements,
    requires_env=["HERMES_HOME"],
)
