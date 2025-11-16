import { useEffect, useState } from "react";
import "./App.css";
import { useNavigate } from "react-router-dom";

type EventRow = {
  id: string;
  patientName: string;
  device: string;
  event: "AF" | "VTACH" | "PAUSE" | string;
  original: string;
  eventTime: string;
  timeInQueue: string | null;
  technician: string;
  approved: boolean;
  isRejected?: boolean;
  eventIndex?: number;
};

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

function Pill({ label }: { label: string }) {
  const colorClass =
    label === "AF" || label === "AFIB"
      ? "pill green"
      : label === "VTACH"
      ? "pill orange"
      : label === "PAUSE"
      ? "pill red"
      : "pill gray";
  return <span className={colorClass}>{label}</span>;
}

function App() {
  const [rows, setRows] = useState<EventRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0]);
  const navigate = useNavigate();

  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    const fetchRows = async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `http://localhost:8000/events?page=${page}&page_size=${pageSize}`
        );
        const data = (await res.json()) as {
          rows: EventRow[];
          total_count: number;
          total_pages: number;
          page: number;
          page_size: number;
        };
        setRows(data.rows);
        setTotalCount(data.total_count);
        setTotalPages(data.total_pages);
        if (data.page !== page) {
          setPage(data.page);
        }
      } finally {
        setLoading(false);
      }
    };
    fetchRows();
    const interval = setInterval(fetchRows, 60000);
    return () => clearInterval(interval);
  }, [page, pageSize]);

  const pageRows = rows;

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [totalPages, page]);

  return (
    <div className="events-container">
      <header className="topbar">
        <div className="brand">
          <div className="logo-icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="#169fe6" />
              <path d="M16 8L20 14H12L16 8Z" fill="white" />
              <path d="M16 24L12 18H20L16 24Z" fill="white" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="title">MoMe</span>
            <span className="subtitle">MONITOR ME</span>
          </div>
        </div>
        <div className="header-info">
          <span className="event-count">Total Events: {totalCount}</span>
        </div>
      </header>

      <div className="content-card">
        <div className="table-container">
          <table className="events-table">
            <thead>
              <tr>
                <th>Patient Name</th>
                <th>Device</th>
                <th>Event</th>
                <th>Original</th>
                <th>Event Time (Practice)</th>
                <th>Time in Queue</th>
                <th>Technician</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} className="loading">
                    Loading…
                  </td>
                </tr>
              ) : pageRows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="loading">
                    No data available
                  </td>
                </tr>
              ) : (
                pageRows.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => navigate(`/event/${r.id}`)}
                    className="table-row"
                  >
                    <td>
                      <span className="link">{r.patientName}</span>
                    </td>
                    <td>{r.device}</td>
                    <td>
                      <Pill label={r.event} />
                    </td>
                    <td>{r.original}</td>
                    <td>{r.eventTime}</td>
                    <td>{r.timeInQueue ?? "-"}</td>
                    <td>{r.technician}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="footer">
          <div className="pagination">
            <button
              className="pagination-nav"
              disabled={page === 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              ‹
            </button>
            {(() => {
              const makeRange = (start: number, end: number) => {
                const arr: number[] = [];
                for (let i = start; i <= end; i++) arr.push(i);
                return arr;
              };
              const buttons: (number | string)[] = [];
              const maxVisiblePages = 14;

              if (totalPages <= maxVisiblePages) {
                buttons.push(...makeRange(1, totalPages));
              } else {
                if (page <= 7) {
                  buttons.push(...makeRange(1, 7), "…", totalPages);
                } else if (page >= totalPages - 6) {
                  buttons.push(
                    1,
                    "…",
                    ...makeRange(totalPages - 6, totalPages)
                  );
                } else {
                  buttons.push(
                    1,
                    "…",
                    page - 1,
                    page,
                    page + 1,
                    "…",
                    totalPages
                  );
                }
              }

              return buttons.map((b, idx) =>
                typeof b === "number" ? (
                  <button
                    key={`p-${b}-${idx}`}
                    className={`pagination-number ${
                      b === page ? "active" : ""
                    }`}
                    onClick={() => setPage(b)}
                  >
                    {b}
                  </button>
                ) : (
                  <span key={`e-${idx}`} className="pagination-ellipsis">
                    …
                  </span>
                )
              );
            })()}
            <button
              className="pagination-nav"
              disabled={page === totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              ›
            </button>
          </div>
          <div className="page-size">
            <span className="page-size-label">Show:</span>
            {PAGE_SIZE_OPTIONS.map((opt) => (
              <button
                key={opt}
                className={`page-size-btn ${opt === pageSize ? "active" : ""}`}
                onClick={() => {
                  setPageSize(opt);
                  setPage(1);
                }}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
