import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import "../App.css";

type EpisodeDetail = {
  metadata: Record<string, any>;
  predicted_label: string;
  event_start_time: number;
  event_start_index: number;
  shape: [number, number];
  ai: {
    classification: string;
    decision: "CONFIRMED" | "REJECTED";
    reasoning: string;
  };
  full_ecg: {
    fs: number;
    total_samples: number;
    total_duration_seconds: number;
    ch1: number[];
    ch2: number[];
    downsample_step: number;
  };
  event_start_index_downsampled: number;
};

export default function EventDetail() {
  const { id } = useParams();
  const [data, setData] = useState<EpisodeDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const mainCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const timelineCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Timeline navigation state
  const [viewStartSeconds, setViewStartSeconds] = useState(0);
  const [viewDurationSeconds, setViewDurationSeconds] = useState(6); // 6 second window
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartViewStart, setDragStartViewStart] = useState(0);
  const classifyTimeout = useRef<number | null>(null);
  // Use light ECG paper theme to match reference UI
  const useDarkGraphTheme = false;

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/episodes/${id}`);
        const json = (await res.json()) as EpisodeDetail;
        setData(json);
        // Initialize view to show event start
        if (json.full_ecg) {
          const eventTime = json.event_start_time || 0;
          setViewStartSeconds(Math.max(0, eventTime - 3));
        }
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [id]);

  const decisionColor = useMemo(() => {
    if (!data) return "#9ca3af";
    return data.ai.decision === "CONFIRMED" ? "#1fbe7f" : "#ef4444";
  }, [data]);

  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${String(secs).padStart(2, "0")}:${String(ms).padStart(
      2,
      "0"
    )}`;
  };

  // Format event time from metadata
  const formatEventTime = (meta: Record<string, any>) => {
    const timeStr = meta.EventOccuredTime;
    if (!timeStr) return "";
    try {
      // Handle format like "2025-11-08 14:09:19.884"
      let dt: Date;
      if (timeStr.includes("T")) {
        dt = new Date(timeStr);
      } else {
        // Replace space with T and add Z if no timezone
        const normalized = timeStr.replace(" ", "T");
        dt = new Date(normalized);
      }
      if (isNaN(dt.getTime())) {
        return timeStr;
      }
      return dt.toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
        timeZoneName: "short",
      });
    } catch {
      return timeStr;
    }
  };

  // Calculate max heart rate from ECG (simplified: count peaks)
  const calculateMaxHR = (
    ch1: number[],
    fs: number,
    downsampleStep: number = 1
  ) => {
    if (!ch1 || ch1.length < 2) return 0;
    const effectiveFs = fs / Math.max(1, downsampleStep || 1);
    if (!isFinite(effectiveFs) || effectiveFs <= 0) return 0;
    // Simple peak detection
    let peaks = 0;
    for (let i = 1; i < ch1.length - 1; i++) {
      if (ch1[i] > ch1[i - 1] && ch1[i] > ch1[i + 1] && ch1[i] > 0.3) {
        peaks++;
      }
    }
    const duration = ch1.length / effectiveFs;
    if (duration <= 0) return 0;
    const calculatedHr = (peaks / duration) * 60;
    // Clamp to physiologically plausible range to avoid outliers
    return Math.round(Math.min(Math.max(calculatedHr, 30), 220));
  };

  // Draw main ECG graph - Hospital-grade rendering
  useEffect(() => {
    if (!data || !mainCanvasRef.current) return;
    const canvas = mainCanvasRef.current;
    const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!ctx) return;

    // Enable high-quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const W = canvas.width;
    const H = canvas.height;
    // Single lead occupies full height to match reference UI
    const leadHeight = H;
    ctx.clearRect(0, 0, W, H);

    const { full_ecg } = data;
    if (!full_ecg) return;

    const fs = full_ecg.fs;
    const downsampleStep = full_ecg.downsample_step;
    const actualFs = fs / downsampleStep;

    // Calculate view window in samples
    const viewStartSamples = viewStartSeconds * fs;
    const viewEndSamples = (viewStartSeconds + viewDurationSeconds) * fs;
    const viewStartIdx = Math.floor(viewStartSamples / downsampleStep);
    const viewEndIdx = Math.floor(viewEndSamples / downsampleStep);

    // Extract view window
    const ch1View = full_ecg.ch1.slice(viewStartIdx, viewEndIdx);

    if (ch1View.length === 0) return;

    // Hospital-grade ECG grid with optional dark theme
    // Standard ECG paper: 1mm small squares, 5mm large squares
    const smallSquare = 10; // 1mm squares in pixels
    const largeSquare = 50; // 5mm squares in pixels

    // Background
    ctx.fillStyle = useDarkGraphTheme ? "#0b1220" : "#FFF8F8";
    ctx.fillRect(0, 0, W, H);

    // Draw minor grid lines (1mm)
    ctx.strokeStyle = useDarkGraphTheme ? "#1f2a44" : "#FFE0E0";
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= W; x += smallSquare) {
      if (x % largeSquare !== 0) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
    }
    for (let y = 0; y <= H; y += smallSquare) {
      if (y % largeSquare !== 0) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
    }

    // Draw major grid lines (5mm)
    ctx.strokeStyle = useDarkGraphTheme ? "#2b3b61" : "#FF9999";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += largeSquare) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y <= H; y += largeSquare) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Draw leads with hospital-grade quality
    const drawLead = (signal: number[], yOffset: number, label: string) => {
      if (signal.length === 0) return;

      // Calculate signal statistics for proper scaling
      const signalArray = new Float32Array(signal);
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;

      for (let i = 0; i < signalArray.length; i++) {
        const val = signalArray[i];
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
      }

      const amp = max - min || 1;
      const mean = sum / signalArray.length;
      const centerY = yOffset + leadHeight / 2;

      // Standard ECG: 10mm = 1mV vertical scale
      // Use 75% of lead height for waveform display
      const scaleY = (leadHeight * 0.75) / amp;

      // Draw baseline (isoelectric line) - dashed
      ctx.strokeStyle = useDarkGraphTheme ? "#334155" : "#E0E0E0";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(W, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw lead label - professional styling
      ctx.fillStyle = useDarkGraphTheme ? "#93c5fd" : "#2C3E50";
      ctx.font = 'bold 14px "Arial", sans-serif';
      ctx.textBaseline = "middle";
      ctx.fillText(label, 12, centerY);

      // Draw waveform - thick, high contrast
      ctx.strokeStyle = useDarkGraphTheme ? "#e5e7eb" : "#000000";
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 0;

      // Smooth waveform rendering - ensure proper x,y plotting
      ctx.beginPath();
      let firstPoint = true;
      for (let i = 0; i < signalArray.length; i++) {
        // Calculate x position based on time (sample index * time per sample)
        const timePosition = i / actualFs; // Time in seconds for this sample
        const px = (timePosition / viewDurationSeconds) * W;

        // Calculate y position based on signal value
        const signalValue = signalArray[i];
        const py = centerY - (signalValue - mean) * scaleY;

        // Ensure x stays within canvas bounds
        if (px >= 0 && px <= W) {
          if (firstPoint) {
            ctx.moveTo(px, py);
            firstPoint = false;
          } else {
            ctx.lineTo(px, py);
          }
        }
      }
      ctx.stroke();

      // Draw time markers every second
      ctx.strokeStyle = useDarkGraphTheme ? "#3b4257" : "#CCCCCC";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([1, 3]);
      for (let t = 0; t <= viewDurationSeconds; t += 1) {
        const x = (t / viewDurationSeconds) * W;
        ctx.beginPath();
        ctx.moveTo(x, yOffset);
        ctx.lineTo(x, yOffset + leadHeight);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    };

    // Draw only one lead (V1) across the full strip
    drawLead(ch1View, 0, "V1");

    // Draw blue event marker and selection frame
    const eventTime = data.event_start_time || 0;
    if (
      eventTime >= viewStartSeconds &&
      eventTime <= viewStartSeconds + viewDurationSeconds
    ) {
      const eventX = ((eventTime - viewStartSeconds) / viewDurationSeconds) * W;

      // Draw semi-transparent blue background frame
      ctx.fillStyle = useDarkGraphTheme
        ? "rgba(22, 159, 230, 0.22)"
        : "rgba(22, 159, 230, 0.12)";
      const frameWidth = Math.min(120, W * 0.15);
      ctx.fillRect(Math.max(0, eventX - frameWidth / 2), 0, frameWidth, H);

      // Draw blue vertical marker line with shadow
      ctx.strokeStyle = "#169fe6";
      ctx.lineWidth = 3;
      ctx.shadowBlur = useDarkGraphTheme ? 6 : 4;
      ctx.shadowColor = "rgba(22, 159, 230, 0.6)";
      ctx.beginPath();
      ctx.moveTo(eventX, 0);
      ctx.lineTo(eventX, H);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw event label above frame
      ctx.fillStyle = "#169fe6";
      ctx.font = 'bold 16px "Arial", sans-serif';
      ctx.textBaseline = "top";
      ctx.textAlign = "center";
      const eventLabel = data.predicted_label || "EVENT";
      ctx.fillText(eventLabel, eventX, 8);

      // Draw duration label below frame
      ctx.font = '12px "Arial", sans-serif';
      ctx.textBaseline = "bottom";
      const durationText = `${viewDurationSeconds.toFixed(2)} secs`;
      ctx.fillText(durationText, eventX, H - 8);
      ctx.textAlign = "left";
    }
  }, [data, viewStartSeconds, viewDurationSeconds]);

  // Draw timeline overview
  useEffect(() => {
    if (!data || !timelineCanvasRef.current) return;
    const canvas = timelineCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const { full_ecg } = data;
    if (!full_ecg) return;

    const totalDuration = full_ecg.total_duration_seconds;
    const ch1 = full_ecg.ch1;

    // Draw compressed ECG - single line
    const min1 = Math.min(...ch1);
    const max1 = Math.max(...ch1);
    const amp1 = max1 - min1 || 1;

    const centerY = H / 2;
    const scale = (H * 0.6) / amp1;

    ctx.strokeStyle = useDarkGraphTheme ? "#e5e7eb" : "#111827";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ch1.forEach((v, i) => {
      const px = (i / (ch1.length - 1)) * W;
      const py = centerY - (v - (min1 + max1) / 2) * scale;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Draw blue selection box
    const selectionLeft = (viewStartSeconds / totalDuration) * W;
    const selectionWidth = (viewDurationSeconds / totalDuration) * W;
    ctx.strokeStyle = "#169fe6";
    ctx.fillStyle = useDarkGraphTheme
      ? "rgba(22, 159, 230, 0.22)"
      : "rgba(22, 159, 230, 0.1)";
    ctx.lineWidth = 2;
    ctx.fillRect(selectionLeft, 0, selectionWidth, H);
    ctx.strokeRect(selectionLeft, 0, selectionWidth, H);

    // Draw time labels
    ctx.fillStyle = useDarkGraphTheme ? "#9ca3af" : "#6b7280";
    ctx.font = "10px sans-serif";
    ctx.fillText("0:00:00", 4, H - 4);
    const endTime = formatTime(totalDuration);
    const endWidth = ctx.measureText(endTime).width;
    ctx.fillText(endTime, W - endWidth - 4, H - 4);
  }, [data, viewStartSeconds, viewDurationSeconds]);

  // Handle timeline drag
  const handleTimelineMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return;
    const canvas = timelineCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const totalDuration = data.full_ecg.total_duration_seconds;
    const clickedSeconds = (x / canvas.width) * totalDuration;

    setIsDragging(true);
    setDragStartX(x);
    setDragStartViewStart(viewStartSeconds);

    // Snap to clicked position
    setViewStartSeconds(
      Math.max(
        0,
        Math.min(
          clickedSeconds - viewDurationSeconds / 2,
          totalDuration - viewDurationSeconds
        )
      )
    );
  };

  const handleTimelineMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !data) return;
    const canvas = timelineCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const deltaX = x - dragStartX;
    const totalDuration = data.full_ecg.total_duration_seconds;
    const deltaSeconds = (deltaX / canvas.width) * totalDuration;

    const newStart = dragStartViewStart + deltaSeconds;
    setViewStartSeconds(
      Math.max(0, Math.min(newStart, totalDuration - viewDurationSeconds))
    );
  };

  const handleTimelineMouseUp = () => {
    setIsDragging(false);
    // Trigger classify after drag ends
    triggerClassify();
  };

  // Navigation functions
  const navigateTimeline = (direction: "first" | "prev" | "next" | "last") => {
    if (!data) return;
    const totalDuration = data.full_ecg.total_duration_seconds;

    switch (direction) {
      case "first":
        setViewStartSeconds(0);
        break;
      case "prev":
        setViewStartSeconds(
          Math.max(0, viewStartSeconds - viewDurationSeconds)
        );
        break;
      case "next":
        setViewStartSeconds(
          Math.min(
            totalDuration - viewDurationSeconds,
            viewStartSeconds + viewDurationSeconds
          )
        );
        break;
      case "last":
        setViewStartSeconds(Math.max(0, totalDuration - viewDurationSeconds));
        break;
    }
    // Trigger classification
    triggerClassify();
  };

  // Call backend to classify the currently selected window (blue frame)
  const triggerClassify = () => {
    if (!data) return;
    // Debounce to avoid spamming while dragging
    if (classifyTimeout.current) window.clearTimeout(classifyTimeout.current);
    classifyTimeout.current = window.setTimeout(async () => {
      const res = await fetch(`http://localhost:8000/episodes/${id}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_seconds: viewStartSeconds,
          duration_seconds: viewDurationSeconds,
        }),
      });
      const json = await res.json();
      setData((prev) =>
        prev
          ? {
              ...prev,
              ai: {
                classification: json.classification,
                decision: json.decision,
                reasoning: json.reasoning,
              },
            }
          : prev
      );
    }, 200);
  };

  if (loading) return <div className="events-container">Loading…</div>;
  if (!data) return <div className="events-container">Not found</div>;

  const meta = data.metadata || {};
  const maxHR = data.full_ecg
    ? calculateMaxHR(
        data.full_ecg.ch1,
        data.full_ecg.fs,
        data.full_ecg.downsample_step
      )
    : 0;

  const pageStyles = {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #f9fafb 0%, #e6ecf5 100%)",
    padding: "40px 0 72px",
    boxSizing: "border-box" as const,
  };

  const layoutStyles = {
    width: "min(1280px, 95vw)",
    margin: "0 auto",
    display: "flex",
    flexDirection: "column" as const,
    gap: 24,
  };

  const heroPanelStyles = {
    background: "#ffffff",
    borderRadius: 28,
    padding: "32px clamp(24px, 4vw, 48px)",
    border: "1px solid rgba(15, 23, 42, 0.06)",
    boxShadow: "0 30px 70px rgba(15, 23, 42, 0.08)",
  };

  const infoPanelStyles = {
    background: "rgba(255,255,255,0.95)",
    borderRadius: 24,
    padding: "32px clamp(24px, 4vw, 36px)",
    border: "1px solid rgba(148, 163, 184, 0.25)",
    boxShadow: "0 18px 40px rgba(15, 23, 42, 0.06)",
    backdropFilter: "blur(6px)",
  };

  const sectionTitleStyles = {
    fontSize: 12,
    color: "#6b7280",
    letterSpacing: 0.5,
    marginBottom: 8,
  };

  return (
    <div style={pageStyles}>
      <header
        style={{
          width: layoutStyles.width,
          margin: "0 auto 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 clamp(24px, 4vw, 32px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="brand">
            <div className="dot" />
            <span className="title">MoMe</span>
            <span className="subtitle">MONITOR ME</span>
          </div>
        </div>
        <div style={{ color: "#0f172a", fontWeight: 600, fontSize: 14 }}>
          Admin
        </div>
      </header>

      <main style={layoutStyles}>
        <section style={heroPanelStyles}>
          <Link
            to="/"
            style={{
              display: "inline-flex",
              alignItems: "center",
              color: "#169fe6",
              textDecoration: "none",
              fontSize: 14,
              fontWeight: 500,
              marginBottom: 16,
            }}
          >
            ← Back to Queue
          </Link>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: 24,
              marginBottom: 24,
            }}
          >
            <div>
              <div style={sectionTitleStyles}>EVENT</div>
              <div style={{ fontSize: 20, fontWeight: 700 }}>
                {meta.Event_Name || data.predicted_label}
              </div>
              <div style={{ marginTop: 8, color: "#4b5563", fontSize: 14 }}>
                MAX HR: {maxHR} BPM
              </div>
              <div style={{ color: "#4b5563", fontSize: 14 }}>
                {formatEventTime(meta)}
              </div>
            </div>
            <div>
              <div style={sectionTitleStyles}>PATIENT</div>
              <div style={{ fontSize: 16, fontWeight: 600 }}>
                {meta.Patient_IR_ID || id}
              </div>
              <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>
                Episode ID: {id}
              </div>
            </div>
            <div>
              <div style={sectionTitleStyles}>WINDOW</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[6, 10, 30].map((duration) => (
                  <button
                    key={duration}
                    className={`btn-scale ${
                      viewDurationSeconds === duration ? "active" : ""
                    }`}
                    onClick={() => {
                      setViewDurationSeconds(duration);
                      triggerClassify();
                    }}
                  >
                    {duration} sec
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div
            style={{
              display: "flex",
              gap: 24,
              marginBottom: 16,
              alignItems: "flex-start",
            }}
          >
            <div style={{ flex: 1, position: "relative" }}>
              <canvas
                ref={mainCanvasRef}
                width={1200}
                height={400}
                style={{
                  width: "100%",
                  border: "1px solid #e5e7eb",
                  borderRadius: 12,
                  display: "block",
                  background: "#fff",
                }}
              />
            </div>

            <div style={{ flex: "0 0 400px" }}>
              <div style={infoPanelStyles}>
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionTitleStyles}>AI CLASSIFICATION</div>

                  <div style={{ marginBottom: 20 }}>
                    <div style={{ ...sectionTitleStyles, marginBottom: 8 }}>
                      DECISION
                    </div>
                    <div
                      style={{
                        color: decisionColor,
                        fontWeight: 700,
                        fontSize: 18,
                      }}
                    >
                      {data.ai.decision}
                    </div>
                  </div>

                  <div>
                    <div style={{ ...sectionTitleStyles, marginBottom: 8 }}>
                      REASONING
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 16,
                        fontSize: 14,
                        lineHeight: 1.8,
                        color: "#374151",
                      }}
                    >
                      {(() => {
                        if (!data.ai.reasoning)
                          return [
                            <div key="left"></div>,
                            <div key="right"></div>,
                          ];

                        // Parse reasoning into sentences
                        const sentences = data.ai.reasoning
                          .split(/\.\s+/)
                          .filter((s) => s.trim().length > 0)
                          .map((s) => s.trim());

                        // Categorize sentences by ECG hospital terms
                        const leftColumn: string[] = [];
                        const rightColumn: string[] = [];

                        sentences.forEach((sentence) => {
                          const lower = sentence.toLowerCase();

                          // Left column: Heart rate, rhythm, beats, R-R intervals, regularity
                          if (
                            lower.includes("heart rate") ||
                            lower.includes("bpm") ||
                            (lower.includes("rhythm") &&
                              !lower.includes("detected")) ||
                            lower.includes("r-r interval") ||
                            lower.includes("beats detected") ||
                            lower.includes("regularity") ||
                            lower.includes("confidence") ||
                            lower.includes("range:")
                          ) {
                            leftColumn.push(sentence);
                          }
                          // Right column: QRS, P-waves, morphology, signal quality, pause, diagnoses
                          else if (
                            lower.includes("qrs") ||
                            lower.includes("p-wave") ||
                            lower.includes("morphology") ||
                            lower.includes("signal quality") ||
                            lower.includes("pause") ||
                            lower.includes("atrial fibrillation") ||
                            lower.includes("ventricular tachycardia") ||
                            lower.includes("cardiac pause") ||
                            lower.includes("detected") ||
                            lower.includes("consistent")
                          ) {
                            rightColumn.push(sentence);
                          }
                          // Default: alternate between columns
                          else {
                            if (leftColumn.length <= rightColumn.length) {
                              leftColumn.push(sentence);
                            } else {
                              rightColumn.push(sentence);
                            }
                          }
                        });

                        return [
                          <div key="left">
                            {leftColumn.map((sentence, idx) => (
                              <div
                                key={idx}
                                style={{
                                  marginBottom:
                                    idx < leftColumn.length - 1 ? 8 : 0,
                                }}
                              >
                                {sentence}
                                {idx < leftColumn.length - 1 && "."}
                              </div>
                            ))}
                          </div>,
                          <div key="right">
                            {rightColumn.map((sentence, idx) => (
                              <div
                                key={idx}
                                style={{
                                  marginBottom:
                                    idx < rightColumn.length - 1 ? 8 : 0,
                                }}
                              >
                                {sentence}
                                {idx < rightColumn.length - 1 && "."}
                              </div>
                            ))}
                          </div>,
                        ];
                      })()}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <canvas
              ref={timelineCanvasRef}
              width={1200}
              height={80}
              style={{
                width: "100%",
                border: "1px solid #e5e7eb",
                borderRadius: 12,
                cursor: isDragging ? "grabbing" : "grab",
                display: "block",
                background: "#fff",
              }}
              onMouseDown={handleTimelineMouseDown}
              onMouseMove={handleTimelineMouseMove}
              onMouseUp={handleTimelineMouseUp}
              onMouseLeave={handleTimelineMouseUp}
            />
          </div>

          <div
            style={{
              display: "flex",
              gap: 6,
              justifyContent: "center",
              paddingTop: 8,
            }}
          >
            <button
              className="btn-nav"
              onClick={() => navigateTimeline("first")}
              title="First window"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <path d="M6 2L2 8 6 14z" />
                <path d="M12 2L8 8l4 6z" />
              </svg>
            </button>
            <button
              className="btn-nav"
              onClick={() => navigateTimeline("prev")}
              title="Previous window"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <path d="M12 2v12L4 8l8-6z" />
              </svg>
            </button>
            <button
              className="btn-nav"
              onClick={() => navigateTimeline("next")}
              title="Next window"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <path d="M4 2v12l8-6-8-6z" />
              </svg>
            </button>
            <button
              className="btn-nav"
              onClick={() => navigateTimeline("last")}
              title="Last window"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
              >
                <path d="M2 2v12l6-6z" />
                <path d="M8 2v12l6-6z" />
              </svg>
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
