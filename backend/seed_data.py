"""
Pre-seed the database with 3 demo users and their session histories.

Run: python -m backend.seed_data
"""

import asyncio
import json
import os
from pathlib import Path

from backend.database import DB_PATH, get_db, init_db

# Demo user profiles with session trajectories
DEMO_USERS = [
    {
        "name": "Sarah M.",
        "description": "Stable patient",
        "sessions": [
            {"score": 36.2, "prob": 0.35, "updrs": 18.5, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 24.8, "jitter": 0.0041, "shimmer": 0.029, "days_ago": 42},
            {"score": 37.8, "prob": 0.37, "updrs": 19.1, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 24.5, "jitter": 0.0043, "shimmer": 0.030, "days_ago": 35},
            {"score": 35.5, "prob": 0.33, "updrs": 17.8, "trend": "stable", "label": "moderate_concern",
             "hnr": 25.1, "jitter": 0.0039, "shimmer": 0.028, "days_ago": 28},
            {"score": 38.1, "prob": 0.38, "updrs": 19.5, "trend": "stable", "label": "moderate_concern",
             "hnr": 24.3, "jitter": 0.0044, "shimmer": 0.031, "days_ago": 21},
            {"score": 36.9, "prob": 0.36, "updrs": 18.8, "trend": "stable", "label": "moderate_concern",
             "hnr": 24.6, "jitter": 0.0042, "shimmer": 0.029, "days_ago": 14},
            {"score": 37.3, "prob": 0.36, "updrs": 19.0, "trend": "stable", "label": "moderate_concern",
             "hnr": 24.5, "jitter": 0.0043, "shimmer": 0.030, "days_ago": 7},
        ],
    },
    {
        "name": "James R.",
        "description": "Improving patient (medication response)",
        "sessions": [
            {"score": 55.2, "prob": 0.62, "updrs": 38.0, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 19.5, "jitter": 0.0072, "shimmer": 0.048, "days_ago": 49},
            {"score": 52.1, "prob": 0.58, "updrs": 35.5, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 20.2, "jitter": 0.0068, "shimmer": 0.045, "days_ago": 42},
            {"score": 48.5, "prob": 0.53, "updrs": 32.0, "trend": "improving", "label": "moderate_concern",
             "hnr": 21.0, "jitter": 0.0062, "shimmer": 0.042, "days_ago": 35},
            {"score": 43.8, "prob": 0.48, "updrs": 28.5, "trend": "improving", "label": "moderate_concern",
             "hnr": 22.1, "jitter": 0.0055, "shimmer": 0.038, "days_ago": 28},
            {"score": 39.2, "prob": 0.42, "updrs": 25.0, "trend": "improving", "label": "moderate_concern",
             "hnr": 23.0, "jitter": 0.0049, "shimmer": 0.034, "days_ago": 21},
            {"score": 34.5, "prob": 0.36, "updrs": 21.5, "trend": "improving", "label": "moderate_concern",
             "hnr": 23.8, "jitter": 0.0044, "shimmer": 0.031, "days_ago": 14},
            {"score": 30.1, "prob": 0.31, "updrs": 18.0, "trend": "improving", "label": "moderate_concern",
             "hnr": 24.5, "jitter": 0.0040, "shimmer": 0.028, "days_ago": 7},
        ],
    },
    {
        "name": "Linda K.",
        "description": "Worsening patient",
        "sessions": [
            {"score": 30.5, "prob": 0.30, "updrs": 16.0, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 25.5, "jitter": 0.0038, "shimmer": 0.026, "days_ago": 42},
            {"score": 34.2, "prob": 0.35, "updrs": 19.0, "trend": "insufficient_data", "label": "moderate_concern",
             "hnr": 24.8, "jitter": 0.0042, "shimmer": 0.029, "days_ago": 35},
            {"score": 40.8, "prob": 0.42, "updrs": 24.5, "trend": "worsening", "label": "moderate_concern",
             "hnr": 23.5, "jitter": 0.0050, "shimmer": 0.035, "days_ago": 28},
            {"score": 48.5, "prob": 0.52, "updrs": 31.0, "trend": "worsening", "label": "moderate_concern",
             "hnr": 22.0, "jitter": 0.0060, "shimmer": 0.042, "days_ago": 21},
            {"score": 56.2, "prob": 0.60, "updrs": 38.0, "trend": "worsening", "label": "moderate_concern",
             "hnr": 20.5, "jitter": 0.0070, "shimmer": 0.049, "days_ago": 14},
            {"score": 64.8, "prob": 0.68, "updrs": 45.0, "trend": "worsening", "label": "high_concern",
             "hnr": 19.0, "jitter": 0.0082, "shimmer": 0.057, "days_ago": 7},
        ],
    },
]


def build_features(s: dict) -> dict:
    """Build a plausible feature dict from session summary values."""
    return {
        "MDVP:Fo(Hz)": 150.0 + (s["hnr"] - 22.0) * 5.0,
        "MDVP:Fhi(Hz)": 200.0 + (s["hnr"] - 22.0) * 8.0,
        "MDVP:Flo(Hz)": 100.0 + (s["hnr"] - 22.0) * 3.0,
        "MDVP:Jitter(%)": s["jitter"] * 100,
        "MDVP:Jitter(Abs)": s["jitter"] * 0.1,
        "MDVP:RAP": s["jitter"] * 0.6,
        "MDVP:PPQ": s["jitter"] * 0.55,
        "Jitter:DDP": s["jitter"] * 1.8,
        "MDVP:Shimmer": s["shimmer"],
        "MDVP:Shimmer(dB)": s["shimmer"] * 10.0,
        "Shimmer:APQ3": s["shimmer"] * 0.7,
        "Shimmer:APQ5": s["shimmer"] * 0.85,
        "MDVP:APQ": s["shimmer"] * 1.1,
        "Shimmer:DDA": s["shimmer"] * 2.1,
        "NHR": 0.05 - (s["hnr"] - 20.0) * 0.005,
        "HNR": s["hnr"],
    }


async def seed():
    """Wipe and re-seed the database."""
    if DB_PATH.exists():
        os.remove(DB_PATH)
    await init_db()

    db = await get_db()
    try:
        for user_data in DEMO_USERS:
            cursor = await db.execute(
                "INSERT INTO users (name) VALUES (?)", (user_data["name"],)
            )
            user_id = cursor.lastrowid

            for s in user_data["sessions"]:
                # Create session with backdated timestamp
                created_at = f"datetime('now', '-{s['days_ago']} days')"
                cursor2 = await db.execute(
                    f"INSERT INTO sessions (user_id, created_at) VALUES (?, {created_at})",
                    (user_id,),
                )
                session_id = cursor2.lastrowid

                features = build_features(s)

                # Build top_changed for non-first sessions
                top_changed = []
                if s["days_ago"] < 42:
                    top_changed = [
                        {
                            "feature": "HNR",
                            "deviation_pct": round((s["hnr"] - 24.0) / 24.0 * 100, 1),
                            "direction": "better" if s["hnr"] > 24.0 else "worse",
                            "current": s["hnr"],
                            "baseline": 24.0,
                        },
                        {
                            "feature": "MDVP:Jitter(%)",
                            "deviation_pct": round((s["jitter"] - 0.004) / 0.004 * 100, 1),
                            "direction": "worse" if s["jitter"] > 0.004 else "better",
                            "current": s["jitter"],
                            "baseline": 0.004,
                        },
                    ]

                await db.execute(
                    """
                    INSERT INTO scores
                        (session_id, pd_score, pd_probability, updrs_estimate,
                         trend, label, features_json, deviations_json, top_changed_json,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                            (SELECT created_at FROM sessions WHERE id = ?))
                    """,
                    (
                        session_id,
                        s["score"],
                        s["prob"],
                        s["updrs"],
                        s["trend"],
                        s["label"],
                        json.dumps(features),
                        json.dumps({}),
                        json.dumps(top_changed),
                        session_id,
                    ),
                )

        await db.commit()
        print(f"Seeded {len(DEMO_USERS)} demo users with session histories.")
        print(f"Database: {DB_PATH}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(seed())
