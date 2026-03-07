"""FastAPI application for PD Voice Monitor."""

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import (
    create_session,
    create_user,
    get_user_past_features,
    get_user_past_score_values,
    get_user_scores,
    get_user_sessions,
    get_user_timeline,
    get_users,
    init_db,
    save_recording,
    save_score,
)
from backend.model import PDVoiceModel
from ml.preprocess import convert_webm_to_wav
from ml.scoring import score_session

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and init DB on startup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await init_db()
    app.state.model = PDVoiceModel()
    yield


app = FastAPI(
    title="PD Voice Monitor API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────

class CreateUserRequest(BaseModel):
    name: str


class CreateSessionRequest(BaseModel):
    user_id: int
    notes: str | None = None


# ── Users ──────────────────────────────────────────────────────────────

@app.post("/api/v1/users")
async def api_create_user(req: CreateUserRequest):
    return await create_user(req.name)


@app.get("/api/v1/users")
async def api_list_users():
    return await get_users()


# ── Sessions ───────────────────────────────────────────────────────────

@app.post("/api/v1/sessions")
async def api_create_session(req: CreateSessionRequest):
    return await create_session(req.user_id, req.notes)


@app.get("/api/v1/users/{user_id}/sessions")
async def api_user_sessions(user_id: int):
    return await get_user_sessions(user_id)


# ── Upload ─────────────────────────────────────────────────────────────

@app.post("/api/v1/sessions/{session_id}/upload")
async def api_upload_audio(session_id: int, file: UploadFile = File(...)):
    """Upload an audio file for a session. Converts WebM to WAV if needed."""
    ext = Path(file.filename or "audio.webm").suffix.lower()
    file_id = uuid.uuid4().hex[:12]
    raw_path = UPLOAD_DIR / f"{file_id}{ext}"

    content = await file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    # Convert to WAV if not already
    if ext in (".webm", ".ogg", ".opus", ".mp3", ".m4a"):
        wav_path = UPLOAD_DIR / f"{file_id}.wav"
        try:
            convert_webm_to_wav(str(raw_path), str(wav_path))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")
    else:
        wav_path = raw_path

    recording_id = await save_recording(session_id, str(wav_path))
    return {
        "recording_id": recording_id,
        "session_id": session_id,
        "wav_path": str(wav_path),
    }


# ── Analyze ────────────────────────────────────────────────────────────

@app.post("/api/v1/sessions/{session_id}/analyze")
async def api_analyze_session(session_id: int):
    """Run the full ML pipeline on the most recent recording for this session."""
    from backend.database import get_db

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT file_path FROM recordings WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No recording found for this session")

        # Get user_id for this session
        cursor2 = await db.execute(
            "SELECT user_id FROM sessions WHERE id = ?", (session_id,)
        )
        session_row = await cursor2.fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")
        user_id = session_row["user_id"]
    finally:
        await db.close()

    wav_path = row["file_path"]
    model: PDVoiceModel = app.state.model

    # Run model prediction
    prediction = model.predict(wav_path)

    # Get past data for scoring
    past_features = await get_user_past_features(user_id)
    past_scores = await get_user_past_score_values(user_id)

    # Compute progression score
    scoring_result = score_session(
        current_features=prediction["features"],
        past_sessions=past_features,
        past_scores=past_scores,
        feature_weights=model.cls_feature_importances,
        pd_probability=prediction["pd_probability"],
        updrs_estimate=prediction["updrs_estimate"],
    )

    # Combine results
    result = {
        "score": scoring_result["score"],
        "trend": scoring_result["trend"],
        "label": scoring_result["label"],
        "baseline_established": scoring_result["baseline_established"],
        "top_changed_features": scoring_result["top_changed_features"],
        "pd_probability": prediction["pd_probability"],
        "pd_label": prediction["pd_label"],
        "updrs_estimate": prediction["updrs_estimate"],
        "confidence": prediction["confidence"],
        "audio_quality": prediction["audio_quality"],
        "features": prediction["features"],
        "deviations": scoring_result["deviations"],
    }

    # Save to DB
    await save_score(session_id, result)

    return result


@app.post("/api/v1/analyze")
async def api_analyze_oneshot(
    user_id: int = Form(...),
    file: UploadFile = File(...),
):
    """
    Convenience one-shot: create session + upload + analyze in one call.
    Used by the frontend for the simplest recording flow.
    """
    session = await create_session(user_id)
    session_id = session["id"]

    # Upload
    ext = Path(file.filename or "audio.webm").suffix.lower()
    file_id = uuid.uuid4().hex[:12]
    raw_path = UPLOAD_DIR / f"{file_id}{ext}"

    content = await file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    if ext in (".webm", ".ogg", ".opus", ".mp3", ".m4a"):
        wav_path = UPLOAD_DIR / f"{file_id}.wav"
        try:
            convert_webm_to_wav(str(raw_path), str(wav_path))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")
    else:
        wav_path = raw_path

    await save_recording(session_id, str(wav_path))

    # Analyze
    model: PDVoiceModel = app.state.model
    prediction = model.predict(str(wav_path))

    past_features = await get_user_past_features(user_id)
    past_scores = await get_user_past_score_values(user_id)

    scoring_result = score_session(
        current_features=prediction["features"],
        past_sessions=past_features,
        past_scores=past_scores,
        feature_weights=model.cls_feature_importances,
        pd_probability=prediction["pd_probability"],
        updrs_estimate=prediction["updrs_estimate"],
    )

    result = {
        "session_id": session_id,
        "score": scoring_result["score"],
        "trend": scoring_result["trend"],
        "label": scoring_result["label"],
        "baseline_established": scoring_result["baseline_established"],
        "top_changed_features": scoring_result["top_changed_features"],
        "pd_probability": prediction["pd_probability"],
        "pd_label": prediction["pd_label"],
        "updrs_estimate": prediction["updrs_estimate"],
        "confidence": prediction["confidence"],
        "audio_quality": prediction["audio_quality"],
    }

    await save_score(session_id, {
        **result,
        "features": prediction["features"],
        "deviations": scoring_result["deviations"],
    })

    return result


# ── Timeline ───────────────────────────────────────────────────────────

@app.get("/api/v1/users/{user_id}/timeline")
async def api_user_timeline(user_id: int):
    return await get_user_timeline(user_id)


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}
