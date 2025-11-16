from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
import os
import json
from datetime import datetime, timezone

from app.data_loader import list_all_episodes, load_episode
from app.preprocess import preprocess_ecg
from app.event_detector import detect_event_start
from app.model import predict_event_type
from app.ecg_classifier_agent import classify_ecg



app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "TriFetch backend running!"}

# ----------------------------
# LIST ALL EPISODES ENDPOINT
# ----------------------------
@app.get("/episodes")
def get_episodes():
    return list_all_episodes()

# ----------------------------
# EVENTS TABLE ENDPOINT
# ----------------------------
@app.get("/events")
def get_events(page: int = 1, page_size: int = 10):
    """
    Returns paginated rows for the Events table UI.
    We only read lightweight JSON metadata for speed.
    
    Query params:
    - page: Page number (1-indexed)
    - page_size: Number of items per page
    """
    rows = []
    episodes = list_all_episodes()

    for ep in episodes:
        # Read the event_*.json file inside the episode folder
        try:
            json_file = next(f for f in os.listdir(ep["path"]) if f.endswith(".json"))
            meta_path = os.path.join(ep["path"], json_file)
            meta = json.load(open(meta_path))
        except StopIteration:
            continue

        # Parse/format fields
        patient_name = meta.get("Patient_IR_ID", ep["id"])
        original_label = meta.get("Event_Name", ep["event_type"])
        
        # Use IsRejected from metadata if available, otherwise use folder name
        is_rejected_meta = meta.get("IsRejected", "0")
        is_rejected = is_rejected_meta == "1" if isinstance(is_rejected_meta, str) else bool(is_rejected_meta)
        # Folder name takes precedence for approved/rejected status
        approved_status = ep["approved"] and not is_rejected

        # Event time formatting (e.g., "Nov 04, 2025, 03:52:19 PM PST")
        event_time_raw = meta.get("EventOccuredTime")
        event_time_dt = None
        formatted_time = None
        if event_time_raw:
            try:
                event_time_dt = datetime.strptime(event_time_raw, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    event_time_dt = datetime.strptime(event_time_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    event_time_dt = None
        if event_time_dt:
            formatted_time = event_time_dt.strftime("%b %d, %Y, %I:%M:%S %p PST")

        # Time in queue (days)
        time_in_queue_days = None
        if event_time_dt:
            now = datetime.now(timezone.utc)
            # Assume metadata time is naive local; treat as UTC for diff
            delta_days = (now - event_time_dt.replace(tzinfo=timezone.utc)).days
            time_in_queue_days = max(delta_days, 0)

        # Predicted label (use folder label as it's the ground truth classification)
        predicted_label = ep["event_type"]
        
        # EventIndex if available (sample index where event occurred)
        event_index = meta.get("EventIndex")

        rows.append({
            "id": ep["id"],
            "patientName": patient_name,
            "device": "Demo9911",
            "event": predicted_label,
            "original": original_label,
            "eventTime": formatted_time or event_time_raw,
            "timeInQueue": f"{time_in_queue_days} days" if time_in_queue_days is not None else None,
            "technician": "System Admin",
            "approved": approved_status,
            "isRejected": is_rejected,
            "eventIndex": event_index,
        })

    # Sort by event time desc when possible
    def sort_key(r):
        try:
            return datetime.strptime(r["eventTime"], "%b %d, %Y, %I:%M:%S %p PST")
        except Exception:
            return datetime.min

    rows.sort(key=sort_key, reverse=True)
    
    # Pagination
    total_count = len(rows)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))  # Ensure page is within valid range
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_rows = rows[start_idx:end_idx]
    
    return {
        "rows": paginated_rows,
        "total_count": total_count,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size,
    }

# ----------------------------
# GET SINGLE EPISODE (clean ECG)
# ----------------------------
@app.get("/episodes/{episode_id}")
def get_episode(episode_id: str):
    episodes = list_all_episodes()

    episode = next((e for e in episodes if e["id"] == episode_id), None)
    if episode is None:
        return {"error": "episode not found"}

    metadata, ecg = load_episode(episode["path"])

    clean_ecg = preprocess_ecg(ecg)

    # ML classification
    predicted_label = predict_event_type(ecg)

    # Detect start time - use EventIndex from metadata if available for more accuracy
    event_index_from_meta = metadata.get("EventIndex")
    if event_index_from_meta is not None:
        # Use metadata EventIndex if available (more accurate)
        fs = 200
        event_info = {
            "start_index": int(event_index_from_meta),
            "start_time": float(event_index_from_meta) / fs,
        }
    else:
        # Fall back to detection algorithm
        event_info = detect_event_start(clean_ecg)

    # ---------- AI Classification using ECG Classifier Agent ----------
    # Use the sophisticated ECG classifier agent for comprehensive analysis
    classification_result = classify_ecg(clean_ecg)
    
    ai = {
        "classification": classification_result["classification"],
        "decision": classification_result["decision"],
        "reasoning": classification_result["reasoning"],
        "confidence": classification_result.get("confidence", 0.0),
    }

    # ---------- Full ECG data for timeline navigation ----------
    fs = 200
    total_samples = clean_ecg.shape[0]
    total_duration_seconds = total_samples / fs
    
    # Down-sample full ECG for timeline overview (keep ~2000 points for smooth navigation)
    timeline_points = 2000
    step_full = max(1, total_samples // timeline_points)
    full_ecg_downsampled = {
        "fs": fs,
        "total_samples": int(total_samples),
        "total_duration_seconds": float(total_duration_seconds),
        "ch1": clean_ecg[::step_full, 0].tolist(),
        "ch2": clean_ecg[::step_full, 1].tolist(),
        "downsample_step": int(step_full),
    }
    
    # Event start index in downsampled coordinates
    event_start_index_downsampled = int(event_info["start_index"] // step_full)

    return {
        "metadata": metadata,
        "predicted_label": predicted_label,
        "event_start_time": event_info["start_time"],
        "event_start_index": event_info["start_index"],
        "shape": clean_ecg.shape,
        "ai": ai,
        "full_ecg": full_ecg_downsampled,
        "event_start_index_downsampled": event_start_index_downsampled,
        "is_rejected": metadata.get("IsRejected") == "1" if isinstance(metadata.get("IsRejected"), str) else bool(metadata.get("IsRejected", False)),
        "event_index_from_meta": metadata.get("EventIndex"),
    }

# ----------------------------
# CLASSIFY SELECTED WINDOW
# ----------------------------
@app.post("/episodes/{episode_id}/classify")
def classify_window(
    episode_id: str,
    payload: dict = Body(...),
):
    """
    Classify a specific time window chosen by the user (blue frame).
    Payload:
      {
        "start_seconds": float,
        "duration_seconds": float
      }
    """
    start_seconds = float(payload.get("start_seconds", 0.0))
    duration_seconds = float(payload.get("duration_seconds", 6.0))
    end_seconds = start_seconds + duration_seconds

    episodes = list_all_episodes()
    episode = next((e for e in episodes if e["id"] == episode_id), None)
    if episode is None:
        return {"error": "episode not found"}

    metadata, ecg = load_episode(episode["path"])
    fs = 200
    start_idx = max(0, int(start_seconds * fs))
    end_idx = min(ecg.shape[0], int(end_seconds * fs))
    if end_idx <= start_idx:
        end_idx = min(ecg.shape[0], start_idx + int(1.0 * fs))  # ensure >=1s

    window = ecg[start_idx:end_idx]
    clean = preprocess_ecg(window)

    # Use ECG Classifier Agent for comprehensive classification
    classification_result = classify_ecg(clean, window_seconds=duration_seconds)

    return {
        "classification": classification_result["classification"],
        "decision": classification_result["decision"],
        "reasoning": classification_result["reasoning"],
        "confidence": classification_result.get("confidence", 0.0),
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
    }
