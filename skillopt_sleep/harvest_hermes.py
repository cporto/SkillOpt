"""SkillOpt-Sleep — Hermes Agent session harvest adapter.

Queries ~/.hermes/state.db (SQLite) for sessions in a time window,
reads messages, and normalizes into SessionDigest objects.

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, List, Optional

from skillopt_sleep.harvest import _detect_feedback, _is_meta_prompt
from skillopt_sleep.types import SessionDigest

HERMES_DB = os.path.expanduser("~/.hermes/state.db")


def _iso_from_epoch(epoch: float) -> str:
    """Convert Unix epoch float to ISO 8601 string."""
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (OSError, ValueError, TypeError):
        return ""


def _parse_tool_calls(raw: Any) -> List[str]:
    """Parse tool_calls JSON into a list of unique tool names.

    Hermes stores tool_calls as a JSON array of objects, each with a
    ``function.name`` or top-level ``name`` field.

    Returns empty list on parse failure.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    names: List[str] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        # Hermes formats: {"type":"function","function":{"name":"...","arguments":"..."}}
        # or direct {"name":"..."}
        fn = tc.get("function")
        if isinstance(fn, dict):
            n = fn.get("name")
            if isinstance(n, str) and n:
                names.append(n)
                continue
        n = tc.get("name")
        if isinstance(n, str) and n:
            names.append(n)
    return names


def _dedup(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _connect_db() -> Optional[sqlite3.Connection]:
    """Open the Hermes SQLite DB read-only."""
    if not os.path.isfile(HERMES_DB):
        return None
    try:
        conn = sqlite3.connect(HERMES_DB)
        conn.row_factory = sqlite3.Row
        # Ensure read-only by setting PRAGMA
        conn.execute("PRAGMA query_only = 1")
        return conn
    except sqlite3.Error:
        return None


def _session_min_timestamp(conn: sqlite3.Connection, session_id: str) -> Optional[float]:
    """Get the minimum (first) message timestamp for a session."""
    cur = conn.execute(
        "SELECT MIN(timestamp) AS ts FROM messages "
        "WHERE session_id = ? AND role IN ('user','assistant') "
        "AND active = 1 AND compacted = 0",
        (session_id,),
    )
    row = cur.fetchone()
    return row["ts"] if row and row["ts"] is not None else None


def _session_max_timestamp(conn: sqlite3.Connection, session_id: str) -> Optional[float]:
    """Get the maximum (last) message timestamp for a session."""
    cur = conn.execute(
        "SELECT MAX(timestamp) AS ts FROM messages "
        "WHERE session_id = ? AND role IN ('user','assistant') "
        "AND active = 1 AND compacted = 0",
        (session_id,),
    )
    row = cur.fetchone()
    return row["ts"] if row and row["ts"] is not None else None


def harvest_hermes(
    *,
    since_iso: Optional[str] = None,
    limit: int = 0,
    min_messages: int = 50,
) -> List[SessionDigest]:
    """Harvest session digests from the Hermes Agent SQLite database.

    Parameters
    ----------
    since_iso : str or None
        ISO 8601 string; only sessions *ending* after this are kept.
        If None, no temporal filter is applied.
    limit : int
        Maximum number of digests to return. 0 = no cap.
    min_messages : int
        Minimum number of user+assistant messages for a session to be
        considered (default 50). Shorter sessions are skipped.

    Returns
    -------
    list[SessionDigest]
    """
    conn = _connect_db()
    if conn is None:
        print("[harvest_hermes] ERROR: Could not open Hermes database at", HERMES_DB)
        return []

    digests: List[SessionDigest] = []

    try:
        # Build the session query
        query = """
            SELECT id, cwd, git_repo_root, git_branch, started_at, message_count
            FROM sessions
            WHERE message_count >= ?
            ORDER BY started_at DESC
        """
        params: List[Any] = [min_messages]

        # If since_iso is given, convert to epoch for comparison
        since_epoch: Optional[float] = None
        if since_iso:
            try:
                # Parse ISO 8601 to epoch
                s = since_iso
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                dt = datetime.fromisoformat(s)
                since_epoch = dt.timestamp()
            except (ValueError, TypeError):
                print(f"[harvest_hermes] WARNING: Could not parse since_iso={since_iso}, ignoring")
                pass

        cursor = conn.execute(query, params)

        for row in cursor:
            session_id = str(row["id"])
            started_epoch = row["started_at"]
            cwd = row["cwd"] or ""
            git_repo_root = row["git_repo_root"] or ""

            # Determine the project field: prefer git_repo_root, fallback to cwd
            project = git_repo_root if git_repo_root else cwd

            # Get message boundaries
            min_ts = _session_min_timestamp(conn, session_id)
            max_ts = _session_max_timestamp(conn, session_id)

            started_at = _iso_from_epoch(min_ts) if min_ts else _iso_from_epoch(started_epoch)

            ended_at = _iso_from_epoch(max_ts) if max_ts else ""

            # Temporal filter: skip if ended_at is before since_iso
            if since_epoch is not None and max_ts is not None and max_ts < since_epoch:
                continue

            # Fetch all messages for this session (user + assistant, active, not compacted)
            msg_cursor = conn.execute(
                "SELECT role, content, tool_calls, timestamp "
                "FROM messages "
                "WHERE session_id = ? AND role IN ('user','assistant') "
                "AND active = 1 AND compacted = 0 "
                "ORDER BY timestamp ASC",
                (session_id,),
            )

            user_prompts: List[str] = []
            assistant_finals: List[str] = []
            tools_seen: List[str] = []
            feedback: List[str] = []
            n_user = 0
            n_asst = 0

            for msg in msg_cursor:
                role = msg["role"]
                content = msg["content"] or ""
                tool_calls_raw = msg["tool_calls"]

                if role == "user":
                    text = content.strip()
                    if text and not _is_meta_prompt(text):
                        n_user += 1
                        user_prompts.append(text)
                        feedback.extend(_detect_feedback(text))
                elif role == "assistant":
                    n_asst += 1
                    text = content.strip()
                    if text:
                        assistant_finals.append(text)
                    # Parse tool calls from JSON
                    tools_seen.extend(_parse_tool_calls(tool_calls_raw))

            # Skip sessions that somehow ended up with no valid messages
            if n_user == 0 and n_asst == 0:
                continue

            # Git branch from the row
            git_branch = row["git_branch"] or ""

            digest = SessionDigest(
                session_id=session_id,
                project=project,
                git_branch=git_branch,
                started_at=started_at,
                ended_at=ended_at,
                user_prompts=user_prompts,
                assistant_finals=assistant_finals[-5:],  # last 5 only
                tools_used=_dedup(tools_seen),
                files_touched=[],
                feedback_signals=feedback,
                n_user_turns=n_user,
                n_assistant_turns=n_asst,
                raw_path="~/.hermes/state.db",
            )

            digests.append(digest)
            if limit and len(digests) >= limit:
                break

    finally:
        conn.close()

    return digests


def main() -> None:
    """CLI entry point: python -m skillopt_sleep.harvest_hermes [--since ISO] [--limit N]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Harvest Hermes Agent session digests for SkillOpt-Sleep"
    )
    parser.add_argument(
        "--since",
        default="",
        help="ISO 8601 datetime; only sessions ending after this (e.g. 2026-07-20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of session digests to return (0 = no cap)",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=50,
        help="Minimum user+assistant messages per session (default 50)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON lines"
    )
    args = parser.parse_args()

    since_iso = args.since if args.since else None

    digests = harvest_hermes(
        since_iso=since_iso,
        limit=args.limit,
        min_messages=args.min_messages,
    )

    if args.json:
        for d in digests:
            print(json.dumps(d.to_dict(), ensure_ascii=False))
    else:
        print(f"[harvest_hermes] Collected {len(digests)} session digests")
        for d in digests:
            print(
                f"  {d.session_id[:20]:22s} | {d.started_at[:16]:16s} | "
                f"{d.n_user_turns:3d}u/{d.n_assistant_turns:3d}a | "
                f"tools={len(d.tools_used):2d} | "
                f"{d.project[:30] if d.project else '(no project)':30s}"
            )


if __name__ == "__main__":
    main()