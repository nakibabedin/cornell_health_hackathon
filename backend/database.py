"""SQLite database for storing users, sessions, and scores."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "data" / "pd_voice.db"


async def get_db() -> aiosqlite.Connection:
    """Open a connection with row_factory enabled."""
    os.makedirs(DB_PATH.parent, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    """Create tables if they don't exist."""
    db = await get_db()
    try:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                file_path TEXT NOT NULL,
                duration_sec REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE REFERENCES sessions(id),
                pd_score REAL NOT NULL,
                pd_probability REAL,
                updrs_estimate REAL,
                trend TEXT,
                label TEXT,
                features_json TEXT,
                deviations_json TEXT,
                top_changed_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        await db.commit()
    finally:
        await db.close()


# ── Users ──────────────────────────────────────────────────────────────

async def create_user(name: str) -> dict:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO users (name) VALUES (?)", (name,)
        )
        await db.commit()
        return {"id": cursor.lastrowid, "name": name}
    finally:
        await db.close()


async def get_users() -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT u.id, u.name, u.created_at,
                   s_latest.pd_score AS last_score,
                   s_latest.trend AS last_trend,
                   s_latest.label AS last_label,
                   (SELECT COUNT(*) FROM sessions WHERE user_id = u.id) AS session_count
            FROM users u
            LEFT JOIN (
                SELECT sc.pd_score, sc.trend, sc.label, se.user_id
                FROM scores sc
                JOIN sessions se ON sc.session_id = se.id
                WHERE sc.id = (
                    SELECT sc2.id FROM scores sc2
                    JOIN sessions se2 ON sc2.session_id = se2.id
                    WHERE se2.user_id = se.user_id
                    ORDER BY sc2.created_at DESC LIMIT 1
                )
            ) s_latest ON s_latest.user_id = u.id
            ORDER BY u.created_at
            """
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Sessions ───────────────────────────────────────────────────────────

async def create_session(user_id: int, notes: str | None = None) -> dict:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO sessions (user_id, notes) VALUES (?, ?)",
            (user_id, notes),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "user_id": user_id}
    finally:
        await db.close()


async def get_user_sessions(user_id: int) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT se.id, se.created_at, se.notes,
                   sc.pd_score, sc.pd_probability, sc.updrs_estimate,
                   sc.trend, sc.label
            FROM sessions se
            LEFT JOIN scores sc ON sc.session_id = se.id
            WHERE se.user_id = ?
            ORDER BY se.created_at
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ── Recordings ─────────────────────────────────────────────────────────

async def save_recording(
    session_id: int, file_path: str, duration_sec: float | None = None
) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO recordings (session_id, file_path, duration_sec) VALUES (?, ?, ?)",
            (session_id, file_path, duration_sec),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


# ── Scores ─────────────────────────────────────────────────────────────

async def save_score(session_id: int, score_data: dict) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            INSERT INTO scores
                (session_id, pd_score, pd_probability, updrs_estimate,
                 trend, label, features_json, deviations_json, top_changed_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                score_data["score"],
                score_data.get("pd_probability"),
                score_data.get("updrs_estimate"),
                score_data.get("trend"),
                score_data.get("label"),
                json.dumps(score_data.get("features", {})),
                json.dumps(score_data.get("deviations", {})),
                json.dumps(score_data.get("top_changed_features", [])),
            ),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_user_scores(user_id: int) -> list[dict]:
    """Get all scores for a user, ordered by time."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT sc.pd_score, sc.pd_probability, sc.updrs_estimate,
                   sc.trend, sc.label, sc.features_json, sc.deviations_json,
                   sc.top_changed_json, sc.created_at, se.id AS session_id
            FROM scores sc
            JOIN sessions se ON sc.session_id = se.id
            WHERE se.user_id = ?
            ORDER BY sc.created_at
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_user_past_features(user_id: int) -> list[dict]:
    """Get feature dicts from all past sessions (for baseline computation)."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT sc.features_json
            FROM scores sc
            JOIN sessions se ON sc.session_id = se.id
            WHERE se.user_id = ?
            ORDER BY sc.created_at
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [json.loads(r["features_json"]) for r in rows if r["features_json"]]
    finally:
        await db.close()


async def get_user_past_score_values(user_id: int) -> list[float]:
    """Get just the PD Voice Index values (for trend detection)."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT sc.pd_score
            FROM scores sc
            JOIN sessions se ON sc.session_id = se.id
            WHERE se.user_id = ?
            ORDER BY sc.created_at
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [r["pd_score"] for r in rows]
    finally:
        await db.close()


async def get_user_timeline(user_id: int) -> list[dict]:
    """Get timeline data for dashboard chart."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT se.id AS session_id, se.created_at AS date,
                   sc.pd_score AS score, sc.trend, sc.label,
                   sc.pd_probability, sc.updrs_estimate,
                   sc.top_changed_json
            FROM sessions se
            JOIN scores sc ON sc.session_id = se.id
            WHERE se.user_id = ?
            ORDER BY se.created_at
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("top_changed_json"):
                d["top_changed_features"] = json.loads(d.pop("top_changed_json"))
            else:
                d.pop("top_changed_json", None)
                d["top_changed_features"] = []
            result.append(d)
        return result
    finally:
        await db.close()
