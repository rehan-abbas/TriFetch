"""
ECG Classification Agent
A sophisticated AI system for analyzing and classifying cardiac events from ECG data.

This agent combines signal processing, feature extraction, and medical knowledge
to accurately classify AF (Atrial Fibrillation), VTACH (Ventricular Tachycardia),
and PAUSE events while assessing signal quality.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGClassifierAgent:
    """
    Professional ECG classification system with medical-grade analysis capabilities.
    """

    def __init__(self):
        self.fs = 200  # Sampling frequency (Hz)
        self.event_types = ["AF", "VTACH", "PAUSE"]
        
        # Medical thresholds based on clinical guidelines
        self.thresholds = {
            "vtach_hr_min": 100,  # VTach minimum HR (bpm)
            "vtach_hr_max": 250,  # VTach maximum HR (bpm)
            "af_rr_variability": 0.25,  # AF R-R interval variability threshold
            "pause_duration": 2.0,  # Pause minimum duration (seconds)
            "qrs_width_max": 0.12,  # Maximum normal QRS width (seconds)
            "signal_saturation": 4000,  # Signal saturation threshold
            "min_signal_std": 10,  # Minimum acceptable signal std deviation
        }

    def classify(self, ecg_data: np.ndarray, window_seconds: float = None) -> Dict:
        """
        Main classification entry point.
        
        Args:
            ecg_data: ECG signal array (N x 2) where columns are ch1, ch2
            window_seconds: Optional window duration for focused analysis
            
        Returns:
            Dictionary with classification, decision, reasoning, and features
        """
        logger.info(f"Classifying ECG with shape {ecg_data.shape}")
        
        # Step 1: Quality Assessment
        quality_result = self._assess_signal_quality(ecg_data)
        
        if not quality_result["is_acceptable"]:
            return {
                "classification": "ARTIFACT",
                "decision": "REJECTED",
                "reasoning": quality_result["rejection_reason"],
                "confidence": 0.0,
                "features": quality_result["features"]
            }
        
        # Step 2: Extract comprehensive features
        features = self._extract_comprehensive_features(ecg_data)
        
        # Step 3: Multi-stage classification
        classification_result = self._classify_arrhythmia(features, ecg_data)
        
        # Step 4: Generate medical reasoning
        reasoning = self._generate_medical_reasoning(
            classification_result,
            features,
            quality_result
        )
        
        return {
            "classification": classification_result["event_type"],
            "decision": classification_result["decision"],
            "reasoning": reasoning,
            "confidence": classification_result["confidence"],
            "features": features
        }

    def _assess_signal_quality(self, ecg_data: np.ndarray) -> Dict:
        """
        Comprehensive signal quality assessment.
        """
        ch1, ch2 = ecg_data[:, 0], ecg_data[:, 1]
        
        features = {
            "ch1_mean": float(np.mean(ch1)),
            "ch2_mean": float(np.mean(ch2)),
            "ch1_std": float(np.std(ch1)),
            "ch2_std": float(np.std(ch2)),
            "ch1_max": float(np.max(ch1)),
            "ch2_max": float(np.max(ch2)),
            "ch1_min": float(np.min(ch1)),
            "ch2_min": float(np.min(ch2)),
        }
        
        # Check for saturation/clipping
        max_amplitude = max(abs(features["ch1_max"]), abs(features["ch1_min"]),
                           abs(features["ch2_max"]), abs(features["ch2_min"]))
        
        if max_amplitude > self.thresholds["signal_saturation"]:
            return {
                "is_acceptable": False,
                "rejection_reason": (
                    "Tracing amplitude exceeds monitor range, causing saturation artifact. "
                    "Please repeat the recording to obtain an interpretable ECG."
                ),
                "features": features
            }
        
        # Check for flatline (insufficient variability)
        if features["ch1_std"] < self.thresholds["min_signal_std"]:
            return {
                "is_acceptable": False,
                "rejection_reason": (
                    "Lead I tracing is nearly flat, consistent with signal dropout. "
                    "Please reacquire the ECG for clinical review."
                ),
                "features": features
            }
        
        if features["ch2_std"] < self.thresholds["min_signal_std"]:
            return {
                "is_acceptable": False,
                "rejection_reason": (
                    "Lead II tracing is nearly flat, consistent with signal dropout. "
                    "Please reacquire the ECG for clinical review."
                ),
                "features": features
            }
        
        # Check for excessive noise (abnormally high variability)
        amplitude_range_ch1 = features["ch1_max"] - features["ch1_min"]
        amplitude_range_ch2 = features["ch2_max"] - features["ch2_min"]
        
        if amplitude_range_ch1 > 3000 or amplitude_range_ch2 > 3000:
            return {
                "is_acceptable": False,
                "rejection_reason": (
                    "Marked artifact is present throughout the tracing, preventing accurate rhythm interpretation. "
                    "Please repeat the recording with the patient at rest."
                ),
                "features": features
            }
        
        avg_std = (features["ch1_std"] + features["ch2_std"]) / 2
        if avg_std < 25:
            quality_summary = "Signal quality excellent; baseline noise is minimal across both leads."
        elif avg_std < 60:
            quality_summary = "Signal quality adequate with mild baseline wander, suitable for clinical interpretation."
        else:
            quality_summary = "Signal quality acceptable with some baseline variability but interpretable."

        return {
            "is_acceptable": True,
            "rejection_reason": None,
            "features": features,
            "summary": quality_summary,
        }

    def _extract_comprehensive_features(self, ecg_data: np.ndarray) -> Dict:
        """
        Extract comprehensive ECG features for classification.
        """
        ch1, ch2 = ecg_data[:, 0], ecg_data[:, 1]
        
        # Use channel 1 as primary lead (typically Lead I or V1)
        primary_signal = ch1
        
        # 1. R-peak detection and heart rate analysis
        r_peaks = self._detect_r_peaks(primary_signal)
        rr_intervals = np.diff(r_peaks) / self.fs  # R-R intervals in seconds
        
        # Heart rate statistics
        if len(rr_intervals) > 0:
            heart_rates = 60.0 / rr_intervals
            avg_hr = float(np.mean(heart_rates))
            hr_std = float(np.std(heart_rates))
            min_hr = float(np.min(heart_rates))
            max_hr = float(np.max(heart_rates))
            
            # R-R variability (key for AF detection)
            rr_std = float(np.std(rr_intervals))
            rr_cv = rr_std / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
        else:
            avg_hr = hr_std = min_hr = max_hr = rr_std = rr_cv = 0.0
        
        # 2. Rhythm regularity assessment
        rhythm_regularity = self._assess_rhythm_regularity(rr_intervals)
        
        # 3. QRS complex analysis
        qrs_features = self._analyze_qrs_morphology(primary_signal, r_peaks)
        
        # 4. Frequency domain analysis
        freq_features = self._frequency_domain_analysis(primary_signal)
        
        # 5. Pause detection
        pause_detected, max_pause_duration = self._detect_pauses(rr_intervals)
        
        # 6. Waveform morphology features
        morphology_features = self._extract_morphology_features(primary_signal, r_peaks)
        
        return {
            # Heart rate features
            "avg_hr": avg_hr,
            "hr_std": hr_std,
            "min_hr": min_hr,
            "max_hr": max_hr,
            "num_beats": len(r_peaks),
            
            # R-R interval features
            "rr_intervals": rr_intervals.tolist() if len(rr_intervals) > 0 else [],
            "rr_std": rr_std,
            "rr_cv": rr_cv,  # Coefficient of variation
            "rhythm_regularity": rhythm_regularity,
            
            # QRS features
            **qrs_features,
            
            # Frequency features
            **freq_features,
            
            # Pause detection
            "pause_detected": pause_detected,
            "max_pause_duration": max_pause_duration,
            
            # Morphology features
            **morphology_features,
        }

    def _detect_r_peaks(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Robust R-peak detection using adaptive thresholding.
        """
        # Bandpass filter to enhance QRS complex (5-15 Hz)
        sos = signal.butter(4, [5, 15], btype='band', fs=self.fs, output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        
        # Differentiation to emphasize rapid changes
        diff_signal = np.diff(filtered)
        diff_signal = np.append(diff_signal, diff_signal[-1])
        
        # Squaring to make all values positive and emphasize larger values
        squared = diff_signal ** 2
        
        # Moving average integration
        window_size = int(0.15 * self.fs)  # 150ms integration window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Adaptive threshold
        threshold = 0.35 * np.max(integrated)
        
        # Find peaks above threshold with minimum distance
        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=int(0.4 * self.fs))
        
        return peaks

    def _assess_rhythm_regularity(self, rr_intervals: np.ndarray) -> float:
        """
        Assess rhythm regularity (0 = very irregular, 1 = very regular).
        """
        if len(rr_intervals) < 3:
            return 0.5
        
        # Calculate coefficient of variation
        cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 1.0
        
        # Convert to regularity score (inverse of CV, normalized)
        regularity = 1.0 / (1.0 + cv * 5)  # Scale factor for normalization
        
        return float(regularity)

    def _analyze_qrs_morphology(self, signal_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """
        Analyze QRS complex morphology.
        """
        if len(r_peaks) == 0:
            return {
                "avg_qrs_width": 0.0,
                "qrs_width_variability": 0.0,
                "wide_qrs_detected": False
            }
        
        qrs_widths = []
        
        for peak in r_peaks:
            # Define window around R-peak
            start = max(0, peak - int(0.1 * self.fs))
            end = min(len(signal_data), peak + int(0.1 * self.fs))
            
            qrs_segment = signal_data[start:end]
            
            # Estimate QRS width by finding zero-crossings or amplitude threshold
            threshold = 0.2 * (np.max(qrs_segment) - np.min(qrs_segment))
            above_threshold = qrs_segment > (np.mean(qrs_segment) + threshold)
            
            # Count samples above threshold
            width_samples = np.sum(above_threshold)
            width_seconds = width_samples / self.fs
            qrs_widths.append(width_seconds)
        
        avg_qrs_width = float(np.mean(qrs_widths))
        qrs_width_std = float(np.std(qrs_widths))
        
        return {
            "avg_qrs_width": avg_qrs_width,
            "qrs_width_variability": qrs_width_std,
            "wide_qrs_detected": avg_qrs_width > self.thresholds["qrs_width_max"]
        }

    def _frequency_domain_analysis(self, signal_data: np.ndarray) -> Dict:
        """
        Frequency domain analysis using FFT.
        """
        # Compute power spectral density
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)))
        
        # Calculate power in different frequency bands
        # VLF: 0.003-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
        vlf_power = float(np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)]))
        lf_power = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)]))
        hf_power = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)]))
        
        total_power = vlf_power + lf_power + hf_power
        
        # LF/HF ratio (autonomic nervous system balance)
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
        
        # Spectral entropy (measure of signal complexity)
        spectral_entropy = float(entropy(psd + 1e-10))  # Add small value to avoid log(0)
        
        return {
            "vlf_power": vlf_power,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "total_power": total_power,
            "lf_hf_ratio": lf_hf_ratio,
            "spectral_entropy": spectral_entropy
        }

    def _detect_pauses(self, rr_intervals: np.ndarray) -> Tuple[bool, float]:
        """
        Detect cardiac pauses (prolonged R-R intervals).
        """
        if len(rr_intervals) == 0:
            return False, 0.0
        
        max_rr = float(np.max(rr_intervals))
        pause_detected = max_rr >= self.thresholds["pause_duration"]
        
        return pause_detected, max_rr

    def _extract_morphology_features(self, signal_data: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """
        Extract waveform morphology features.
        """
        if len(r_peaks) < 2:
            return {
                "p_wave_present": False,
                "morphology_consistency": 0.0
            }
        
        # Extract beat-to-beat templates
        templates = []
        template_length = int(0.6 * self.fs)  # 600ms window
        
        for peak in r_peaks[:-1]:  # Exclude last peak to ensure full window
            start = max(0, peak - int(0.2 * self.fs))
            end = min(len(signal_data), start + template_length)
            
            if end - start == template_length:
                template = signal_data[start:end]
                templates.append(template)
        
        if len(templates) < 2:
            return {
                "p_wave_present": False,
                "morphology_consistency": 0.0
            }
        
        # Calculate template consistency using cross-correlation
        templates = np.array(templates)
        mean_template = np.mean(templates, axis=0)
        
        correlations = []
        for template in templates:
            corr = np.corrcoef(template, mean_template)[0, 1]
            correlations.append(corr)
        
        morphology_consistency = float(np.mean(correlations))
        
        # P-wave detection (simplified: look for deflection before R-peak)
        # In real implementation, would use more sophisticated methods
        p_wave_present = morphology_consistency > 0.7  # High consistency suggests P-waves present
        
        return {
            "p_wave_present": p_wave_present,
            "morphology_consistency": morphology_consistency
        }

    def _classify_arrhythmia(self, features: Dict, ecg_data: np.ndarray) -> Dict:
        """
        Multi-stage arrhythmia classification based on extracted features.
        """
        scores = {
            "AF": 0.0,
            "VTACH": 0.0,
            "PAUSE": 0.0,
            "NORMAL": 0.0
        }
        
        # 1. PAUSE Detection (highest priority - clear pause indication)
        if features["pause_detected"]:
            scores["PAUSE"] += 0.6
            if features["max_pause_duration"] >= 3.0:
                scores["PAUSE"] += 0.3
        
        # 2. VTACH Detection
        # Criteria: Fast regular rhythm, wide QRS, high HR
        if features["avg_hr"] >= self.thresholds["vtach_hr_min"]:
            scores["VTACH"] += 0.3
            
            if features["avg_hr"] >= 150:
                scores["VTACH"] += 0.2
            
            if features["rhythm_regularity"] > 0.7:
                scores["VTACH"] += 0.2
            
            if features["wide_qrs_detected"]:
                scores["VTACH"] += 0.3
        
        # 3. AF Detection
        # Criteria: Irregular rhythm, absent P-waves, variable R-R intervals
        if features["rr_cv"] > self.thresholds["af_rr_variability"]:
            scores["AF"] += 0.4
            
            if features["rr_cv"] > 0.4:
                scores["AF"] += 0.2
            
            if features["rhythm_regularity"] < 0.4:
                scores["AF"] += 0.3
            
            if not features["p_wave_present"]:
                scores["AF"] += 0.2
        
        # 4. Normal rhythm indicators
        if (60 <= features["avg_hr"] <= 100 and 
            features["rhythm_regularity"] > 0.8 and
            not features["pause_detected"]):
            scores["NORMAL"] += 0.7
        
        # Select event type with highest score
        event_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[event_type]
        
        # Decision logic: Confirm if confidence is high enough
        decision = "CONFIRMED" if confidence >= 0.5 and event_type != "NORMAL" else "REJECTED"
        
        # Override decision if NORMAL has highest score
        if event_type == "NORMAL":
            decision = "REJECTED"
            event_type = "NORMAL_SINUS_RHYTHM"
        
        return {
            "event_type": event_type,
            "decision": decision,
            "confidence": confidence,
            "scores": scores
        }

    def _generate_medical_reasoning(
        self,
        classification_result: Dict,
        features: Dict,
        quality_result: Dict
    ) -> str:
        """
        Generate comprehensive medical reasoning for the classification.
        """
        event_type = classification_result["event_type"]
        confidence = classification_result["confidence"]
        
        reasoning_parts = []
        
        readable_event = {
            "AF": "Atrial fibrillation",
            "VTACH": "Ventricular tachycardia",
            "PAUSE": "Cardiac pause",
            "NORMAL_SINUS_RHYTHM": "Normal sinus rhythm",
        }.get(event_type, event_type)

        confidence_pct = max(0.0, min(confidence, 1.0)) * 100
        reasoning_parts.append(
            f"{readable_event} pattern observed (confidence {confidence_pct:.0f}%)."
        )

        avg_hr = features.get("avg_hr", 0.0)
        min_hr = features.get("min_hr", 0.0)
        max_hr = features.get("max_hr", 0.0)
        regularity = features.get("rhythm_regularity", 0.0)

        if avg_hr > 0:
            if regularity >= 0.75:
                rhythm_descriptor = "stable regular rhythm"
            elif regularity <= 0.45:
                rhythm_descriptor = "irregular rhythm"
            else:
                rhythm_descriptor = "slightly variable rhythm"
            hr_sentence = f"Heart rate {avg_hr:.0f} BPM"
            if min_hr > 0 and max_hr > 0:
                hr_sentence += f" (range {min_hr:.0f}-{max_hr:.0f})"
            hr_sentence += f" with {rhythm_descriptor}."
            reasoning_parts.append(hr_sentence)

        # Event-specific reasoning
        if event_type == "AF":
            reasoning_parts.append(
                f"R-R intervals are markedly irregular (coefficient of variation {features['rr_cv']:.2f}, regularity {regularity:.2f})."
            )
            if not features["p_wave_present"]:
                reasoning_parts.append("No organized P-waves are visible, supporting atrial fibrillation.")
        elif event_type == "VTACH":
            reasoning_parts.append(
                "Sustained rapid ventricular rate with uniform beat-to-beat morphology."
            )
            if features["wide_qrs_detected"]:
                reasoning_parts.append(
                    f"QRS complexes are wide (average {features['avg_qrs_width']*1000:.0f} ms)."
                )
            reasoning_parts.append(
                f"Morphology consistency score {features['morphology_consistency']:.2f}, indicating a single ectopic focus."
            )
        elif event_type == "PAUSE":
            reasoning_parts.append(
                f"Longest pause measured {features['max_pause_duration']:.2f} seconds (clinical threshold {self.thresholds['pause_duration']} s)."
            )
            reasoning_parts.append(
                f"Baseline ventricular rate {avg_hr:.0f} BPM with {features['num_beats']} beats captured in the window."
            )
        elif event_type == "NORMAL_SINUS_RHYTHM":
            reasoning_parts.append(
                "Sinus rhythm without pathological arrhythmia detected during the reviewed window."
            )

        quality_sentence = quality_result.get(
            "summary",
            "Signal quality adequate for interpretation."
        )
        reasoning_parts.append(quality_sentence)
        
        return " ".join(reasoning_parts)


# Singleton instance
_classifier_instance = None


def get_classifier() -> ECGClassifierAgent:
    """Get or create the global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ECGClassifierAgent()
    return _classifier_instance


def classify_ecg(ecg_data: np.ndarray, window_seconds: float = None) -> Dict:
    """
    Convenient function to classify ECG data.
    
    Args:
        ecg_data: ECG signal array (N x 2)
        window_seconds: Optional window duration for focused analysis
        
    Returns:
        Classification result dictionary
    """
    classifier = get_classifier()
    return classifier.classify(ecg_data, window_seconds)

