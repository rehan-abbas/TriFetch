import numpy as np

# Keep sampling rate consistent with preprocessing
FS = 200  # Hz

def _moving_rms(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute moving RMS efficiently using cumulative sums.
    window_size is in samples.
    """
    if window_size <= 1:
        return np.abs(signal)
    pad = window_size - 1
    padded = np.pad(signal, (pad, 0), mode="constant")
    csum = np.cumsum(padded * padded, dtype=float)
    # RMS over window_size samples
    sq_mean = (csum[window_size:] - csum[:-window_size]) / window_size
    rms = np.sqrt(np.maximum(sq_mean, 0.0))
    return rms

def detect_event_start(clean_ecg: np.ndarray) -> dict:
    """
    Detects the start of an ECG event.
    clean_ecg: shape (N, 2). Returns dict with 'start_index' and 'start_time'.
    Strategy:
      - Combine channels via absolute max to be robust to polarity differences.
      - Compute moving RMS envelope over ~200 ms.
      - Threshold at baseline mean + 3*std from the first 5 seconds as baseline.
      - Pick first index crossing the threshold; back off 0.25s for start.
    """
    if clean_ecg.ndim != 2 or clean_ecg.shape[1] != 2:
        raise ValueError("clean_ecg must be (N, 2)")

    num_samples = clean_ecg.shape[0]
    if num_samples == 0:
        return {"start_index": 0, "start_time": 0.0}

    # Combine channels: robust envelope seed
    combined = np.maximum(np.abs(clean_ecg[:, 0]), np.abs(clean_ecg[:, 1]))

    # Envelope via moving RMS over ~200 ms
    window_ms = 200
    window_size = max(1, int(FS * window_ms / 1000))
    envelope = _moving_rms(combined, window_size)

    # Baseline from first 5s (or all if shorter)
    baseline_len = min(num_samples, 5 * FS)
    baseline = envelope[:baseline_len]
    base_mean = float(np.mean(baseline)) if baseline_len > 0 else float(np.mean(envelope))
    base_std = float(np.std(baseline)) if baseline_len > 0 else float(np.std(envelope))
    threshold = base_mean + 3.0 * base_std

    # First threshold crossing
    crossing_indices = np.where(envelope > threshold)[0]
    if crossing_indices.size == 0:
        # Fallback: use global max as proxy
        peak_idx = int(np.argmax(envelope))
        start_idx = max(0, peak_idx - int(0.25 * FS))
    else:
        cross_idx = int(crossing_indices[0])
        start_idx = max(0, cross_idx - int(0.25 * FS))

    start_time = start_idx / float(FS)
    return {"start_index": start_idx, "start_time": start_time}


