import numpy as np
from scipy.signal import butter, filtfilt

FS = 200  # sampling rate

def bandpass_filter(signal, low=0.5, high=40, fs=FS, order=3):
    """
    Bandpass filter using Butterworth.
    Removes baseline wander (<0.5Hz) and high frequency noise (>40Hz).
    """
    nyquist = fs / 2
    low /= nyquist
    high /= nyquist

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detrend_signal(signal):
    """Removes DC offset (baseline drift)."""
    return signal - np.mean(signal)

def normalize_signal(signal):
    """Normalizes signal to range [-1, 1]."""
    max_val = max(abs(np.max(signal)), abs(np.min(signal)))
    if max_val == 0:
        return signal
    return signal / max_val

def preprocess_ecg(ecg_array):
    """
    Input: (18000, 2) ECG
    Output: Cleaned ECG (18000, 2)
    """

    processed = np.zeros_like(ecg_array)

    # Process each channel independently
    for ch in range(ecg_array.shape[1]):
        signal = ecg_array[:, ch]

        # Step 1: Detrend
        signal = detrend_signal(signal)

        # Step 2: Bandpass filter
        signal = bandpass_filter(signal)

        # Step 3: Normalize
        signal = normalize_signal(signal)

        processed[:, ch] = signal

    return processed
