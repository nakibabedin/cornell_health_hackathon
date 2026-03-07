# Parkinson's Voice Monitoring Hackathon MVP

This document is a ready-to-build task list for a weekend prototype.
It is split into small phases so an AI agent can execute one section at a time without large context overhead.

## Project Goal
- Build a polished demo app that records short voice tasks and returns a "PD Voice Trend Score" (0-100).
- Focus on presentation quality, end-to-end flow, and believable outputs.
- Do not claim diagnosis or clinical-grade performance.

## Delivery Targets
- Working demo flow: `Record -> Upload -> Score -> Trend Dashboard`.
- One-click demo mode with seeded users/data for reliable presentations.
- Clear non-diagnostic disclaimer in UI.

## Recommended Repo Layout

```text
cornell_health/
  frontend/               # Next.js app
  backend/                # FastAPI app
  ml/                     # feature + embedding + scoring modules
  data/
    seeded/               # seeded demo sessions
  docs/
    API_CONTRACT.md
    DEMO_SCRIPT.md
  READY_TO_BUILD_TASK_LIST.md
```

---

## Phase 0: Scaffold and Contracts (Start Here)

### Objective
Create project structure and frozen interfaces before implementation.

### Tasks
- Create folders: `frontend`, `backend`, `ml`, `data/seeded`, `docs`.
- Add root `README.md` with setup and run commands.
- Define API contract in `docs/API_CONTRACT.md`.
- Define sample payloads for score responses.
- Decide a single score schema and keep it stable for all phases.

### API Endpoints to Lock
- `POST /api/v1/session/start`
- `POST /api/v1/session/{session_id}/upload`
- `POST /api/v1/session/{session_id}/score`
- `GET /api/v1/user/{user_id}/timeline`
- `GET /api/v1/demo/seeded-users`

### Acceptance Criteria
- Folder structure exists.
- API contract file is complete and specific.
- Team agrees on JSON field names and score ranges.

### Agent Prompt Chunk
"Create project scaffold and write `docs/API_CONTRACT.md` with request/response examples for all endpoints in Phase 0."

---

## Phase 1: Backend MVP API (No ML Yet)

### Objective
Make all endpoints functional using mock scoring so frontend can be developed in parallel.

### Tasks
- Initialize FastAPI service in `backend`.
- Implement all Phase 0 endpoints.
- Save uploads locally (or to temp object storage abstraction).
- Return deterministic mock score from uploaded file metadata (length + loudness proxy).
- Add OpenAPI docs and basic error handling.

### Data Model (Minimal)
- `User`: `id`, `display_name`, `created_at`
- `Session`: `id`, `user_id`, `task_type`, `med_state`, `created_at`
- `Recording`: `id`, `session_id`, `path`, `duration_sec`, `sample_rate`
- `Score`: `id`, `session_id`, `score_0_100`, `confidence_0_1`, `label`, `created_at`

### Acceptance Criteria
- Swagger docs load and endpoints are testable.
- Upload + score flow works with mocked scores.
- Timeline endpoint returns ordered session history.

### Agent Prompt Chunk
"Implement FastAPI endpoints with mock scorer and local file persistence. Add request validation and basic tests."

---

## Phase 2: Voice Processing Pipeline (Real Features)

### Objective
Replace mock scoring with a basic real signal pipeline.

### Tasks
- In `ml/`, add audio preprocessing:
- Resample to consistent rate (e.g., 16 kHz).
- Trim silence with VAD.
- Normalize amplitude.
- Extract acoustic features:
- Pitch stats, jitter/shimmer proxy, speech rate proxy, pause ratio, energy variability.
- Add embedding extractor:
- Use wav2vec2 hidden-state pooled embedding.
- Store combined feature vector per recording.

### Suggested Python Modules
- `ml/preprocess.py`
- `ml/features_acoustic.py`
- `ml/features_embedding.py`
- `ml/vectorize.py`

### Acceptance Criteria
- Pipeline runs on uploaded `.wav/.m4a`.
- Feature vector produced for every valid recording.
- Invalid/noisy clips return structured error.

### Agent Prompt Chunk
"Implement preprocessing + acoustic feature extraction + wav2vec2 pooled embedding and output one combined feature vector."

---

## Phase 3: Scoring Logic (Hackathon-Grade)

### Objective
Generate believable progression score and label for demo.

### Tasks
- Build simple scorer in `ml/scoring.py`:
- Input: feature vector + optional user baseline history.
- Output:
- `score_0_100`
- `confidence_0_1`
- `label` in `{improved, stable, worsened}`
- Use rule-based + lightweight model hybrid:
- Rule-based baseline delta for timeline behavior.
- Optional small model (`logreg`/`xgboost`) for realism.
- Add smoothing over recent sessions to reduce noise.

### Scoring Rules (Starter)
- First two sessions define baseline mean.
- Delta from baseline maps to label thresholds.
- Confidence reduced when clip quality is low or duration is short.

### Acceptance Criteria
- Repeated similar recordings produce similar scores.
- Timeline trend changes smoothly, not randomly.
- Labels align with score direction.

### Agent Prompt Chunk
"Implement `ml/scoring.py` with baseline-aware trend scoring and stable labels for timeline demos."

---

## Phase 4: Frontend UX (Pitch-Ready)

### Objective
Deliver a clean, high-confidence demo flow with strong visual clarity.

### Screens
- Landing page with product statement and disclaimer.
- Record page with three tasks:
- Sustained vowel
- Read passage
- Free speech
- Results page with:
- Score card
- Confidence indicator
- Session comparison vs baseline
- Timeline page with trend chart and badge states.

### Tasks
- Build Next.js app in `frontend`.
- Implement upload and scoring calls.
- Add seeded-demo toggle to auto-load sample users.
- Add loading, retry, and error states.
- Add polished visual system (intentional typography, motion, and color tokens).

### Acceptance Criteria
- Full flow works from browser without manual API calls.
- Demo mode loads instantly and always shows meaningful trend.
- Mobile and desktop layouts are both usable.

### Agent Prompt Chunk
"Build Next.js frontend for record-score-timeline flow with seeded demo mode and polished presentation."

---

## Phase 5: Seeded Data + Demo Reliability

### Objective
Make demo robust even with bad live audio or unstable network.

### Tasks
- Create 3 seeded users in `data/seeded`:
- `stable_user`
- `improving_user`
- `worsening_user`
- Precompute 5-8 sessions each with realistic score trajectories.
- Add `GET /api/v1/demo/seeded-users` and `GET /api/v1/demo/load/{user}`.
- Add fallback button in UI: "Use preloaded demo session."

### Acceptance Criteria
- Demo can be completed without live recording.
- Each seeded user tells a clear story through trend lines.

### Agent Prompt Chunk
"Create seeded demo users and endpoints so presentation works even without live audio."

---

## Phase 6: QA, Pitch Assets, and Guardrails

### Objective
Finalize for hackathon presentation quality.

### Tasks
- Add smoke tests for critical endpoints.
- Add one end-to-end frontend test for happy path.
- Add a clear disclaimer in UI and README:
- "Prototype for digital biomarker feasibility, not diagnostic use."
- Write `docs/DEMO_SCRIPT.md`:
- 3-minute script
- click path
- backup flow
- key one-liners for judges

### Acceptance Criteria
- Demo runbook exists and is rehearsable.
- App handles common errors gracefully.
- All claims in UI are non-diagnostic.

### Agent Prompt Chunk
"Add smoke tests, disclaimer text, and a 3-minute demo script with fallback flow."

---

## Phase 7: Stretch (Only If Time Remains)

### Objective
Add high-impact upgrades without expanding core risk.

### Options
- Add medication timing selector and show ON/OFF contextual annotation.
- Add explainability card: top contributing voice signals for current score.
- Export one-page PDF summary for a user timeline.
- Add basic auth for private demo links.

### Acceptance Criteria
- No regressions in core demo flow.
- Each stretch feature is independently toggleable.

---

## Work Sequencing Rules for AI Agents

- Only execute one phase at a time.
- Do not start a new phase until acceptance criteria of current phase are met.
- Keep PRs/snapshots small: ideally 1 phase per change set.
- Preserve API response fields once Phase 0 is locked.
- If blocked, implement a local mock and continue.

## Definition of Done (Hackathon MVP)

- End-to-end demo works in under 2 minutes.
- Live recording works at least once on stage machine.
- Seeded fallback always works.
- UI looks intentional and polished.
- Story is clear: baseline -> latest score -> trend interpretation.

