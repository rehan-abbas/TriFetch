import numpy as np
from scipy.signal import welch

FS = 200

def compute_basic_features(ecg_signal):
    """
    Extracts simple, robust features from ECG.
    This is not medically perfect but works well for classification.
    """

    ch = ecg_signal[:, 0]  # use channel 1 for simplicity

    features = []

    # --------- Time-domain features ---------
    features.append(np.mean(ch))
    features.append(np.std(ch))
    features.append(np.max(ch) - np.min(ch))  # amplitude range
    features.append(np.median(ch))
    features.append(np.percentile(ch, 25))
    features.append(np.percentile(ch, 75))

    # --------- Energy ---------
    features.append(np.sum(ch ** 2))

    # --------- Frequency-domain (using Welch) ---------
    f, pxx = welch(ch, FS, nperseg=1024)
    features.append(np.sum(pxx[(f >= 0.5) & (f <= 4)]))    # Low freq energy
    features.append(np.sum(pxx[(f >= 4) & (f <= 12)]))     # Mid freq energy
    features.append(np.sum(pxx[(f >= 12) & (f <= 40)]))    # High freq energy

    return np.array(features, dtype=float)
