Slide 5 — How It Works (Pipeline)


Browser mic  →  WebM audio  →  ffmpeg (WAV)  →  Noise reduction
     →  Feature extraction (parselmouth + librosa)
     →  16 acoustic biomarkers (jitter, shimmer, HNR, NHR, F0...)
     →  XGBoost classification  →  PD probability (0–1)
     →  XGBoost regression      →  UPDRS severity estimate
     →  PD Voice Index (0–100 composite score)
     →  SHAP explanations  →  "which biomarkers drove this result"
Stored per session, per patient. Trend computed across sessions. Doctor sees deterioration before it becomes a clinical emergency.