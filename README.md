# TriFetch Hiring Assignment

Clinical review console that lets Trifetch clinicians inspect long-form Holter ECG traces, re-run the AI classification on any 6–30 s window, and capture medical-grade rationale.

## Project Structure

```
backend/    FastAPI service + ECG classifier agent and preprocessing
frontend/   React + Vite single-page app for ECG exploration
data/       Sample AF / VTACH episodes consumed by the API
```

## Prerequisites

- **Python 3.10+** (developed with 3.11).
- **Node.js 20+** and npm 10+ for the Vite frontend.
- macOS/Linux or WSL recommended; Windows works with PowerShell equivalents.

## Backend (FastAPI) — Setup & Run

1. `cd trifetch-assignment/backend`
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. API docs are available at `http://localhost:8000/docs`.

### Backend Notes

- Episodes are loaded from the checked-in `data/` directory (AF/VTACH samples).
- `app.ecg_classifier_agent` performs preprocessing, signal-quality gating, feature extraction, and medical reasoning before responding.
- Re-classification endpoint: `POST /episodes/{id}/classify` with `{ "start_seconds": ..., "duration_seconds": ... }`.

## Frontend (Vite + React) — Setup & Run

1. `cd trifetch-assignment/frontend`
2. Install dependencies: `npm install`
3. Launch the dev server: `npm run dev`
4. Open `http://localhost:5173` and ensure the backend (`localhost:8000`) is running.

### Frontend Notes

- `EventDetail.tsx` renders a hospital-style ECG grid on `<canvas>` and keeps the graph/timeline in sync.
- Classification controls debounce requests so clinicians can scrub the strip without overloading the backend.
- Styling lives in `src/App.css`; button/brand tokens mimic the MoMe reference UI.

## Technical Choices & Reasoning

- **FastAPI backend** for its async-friendly IO and automatic OpenAPI docs, making it easy for reviewers to hit endpoints directly.
- **Custom ECG classifier agent** replaces heuristics with a multi-stage pipeline (quality checks → feature extraction → event scoring → doctor-friendly reasoning). This keeps the AI explanation aligned with cardiology terms (heart rate, rhythm regularity, QRS width, pauses) instead of device jargon.
- **React + Vite frontend** for instant HMR and TypeScript safety; the ECG renderer uses raw canvas for pixel-level control (paper grid, lead labels, event markers) rather than slower SVG/DOM approaches.
- **Data-driven layout**: timeline + viewport ensures the blue “clinical window” stays consistent between the waveform, buttons, and backend payloads, reducing clinical review errors.

## Future Model / Product Improvements (Optional Ideas)

1. **Confidence-aware UX** – expose the classifier’s confidence to gate “auto-approve” suggestions and highlight low-confidence beats automatically.
2. **Beat annotation layer** – overlay detected R-peaks / arrhythmic beats on the canvas to make AI reasoning visually inspectable.
3. **Continuous ingest** – convert the data loader to stream from MongoDB collections (user already has MongoDB) so reviewers always see the latest Holter uploads.
4. **Model versioning** – log classifier version + feature dump per review window to audit future retraining cycles.
