// frontend/src/App.jsx
/**
 * Root component. Owns all top-level state:
 *   - activeJobId: the job currently being displayed
 *   - jobHistory:  list of all jobs submitted this session (+ localStorage)
 *
 * Layout:
 *   ┌─────────────────────────────────────┬─────────────┐
 *   │  Header                             │             │
 *   ├──────────────────┬──────────────────│  Sidebar    │
 *   │  Submit form     │  Progress /      │  (history)  │
 *   │                  │  Results panel   │             │
 *   └──────────────────┴──────────────────┴─────────────┘
 */
import { useCallback, useEffect, useState } from "react";
import { useJobPolling } from "./hooks/useJobPolling.js";
import { isRunning }      from "./utils/formatting.js";
import SubmitForm         from "./components/SubmitForm.jsx";
import ProgressTracker    from "./components/ProgressTracker.jsx";
import VerdictCard        from "./components/VerdictCard.jsx";
import ConfidenceGauge    from "./components/ConfidenceGauge.jsx";
import TemporalChart      from "./components/TemporalChart.jsx";
import HeatmapViewer      from "./components/HeatmapViewer.jsx";
import JobHistory         from "./components/JobHistory.jsx";
import StatusBadge        from "./components/StatusBadge.jsx";

const STORAGE_KEY = "deepfake_job_history";
const MAX_HISTORY = 20;

function loadHistory() {
  try   { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"); }
  catch { return []; }
}

function saveHistory(jobs) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.slice(0, MAX_HISTORY))); }
  catch {}
}

export default function App() {
  const [activeJobId, setActiveJobId] = useState(null);
  const [history,     setHistory]     = useState(loadHistory);

  // Poll the active job — hook manages interval lifecycle
  const { job, result, error } = useJobPolling(activeJobId);

  // When a job reaches DONE, update the history entry with the verdict
  // so the sidebar can show it without a separate fetch.
  useEffect(() => {
    if (!job || !job.job_id) return;

    setHistory((prev) => {
      const updated   = [...prev];
      const existsIdx = updated.findIndex((h) => h.job_id === job.job_id);

      const entry = {
        job_id:    job.job_id,
        status:    job.status,
        source:    job.source,
        created_at: job.created_at,
        verdict:   result?.report?.verdict ?? null,
      };

      if (existsIdx >= 0) {
        updated[existsIdx] = entry;
      } else {
        updated.unshift(entry);
      }

      saveHistory(updated);
      return updated;
    });
  }, [job, result]);

  const handleJobCreated = useCallback((newJob) => {
    setActiveJobId(newJob.job_id);
    setHistory((prev) => {
      const entry = {
        job_id:     newJob.job_id,
        status:     newJob.status,
        source:     newJob.source,
        created_at: newJob.created_at,
        verdict:    null,
      };
      const updated = [entry, ...prev.filter((h) => h.job_id !== newJob.job_id)];
      saveHistory(updated);
      return updated;
    });
  }, []);

  const report    = result?.report ?? null;
  const running   = job ? isRunning(job.status) : false;

  return (
    <div className="min-h-screen bg-surface-900 font-sans text-slate-200">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 px-6 py-4
                         flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🔍</span>
          <div>
            <h1 className="font-display text-2xl text-slate-100 leading-none">
              Deepfake Detector
            </h1>
            <p className="text-xs font-body text-slate-600 mt-0.5">
              Production analysis pipeline
            </p>
          </div>
        </div>

        {/* Current job status in header */}
        {job && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-body text-slate-600 hidden sm:block">
              {job.job_id.slice(0, 8)}…
            </span>
            <StatusBadge status={job.status} running={running} />
          </div>
        )}
      </header>

      {/* ── Main layout ─────────────────────────────────────────────── */}
      <div className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1
                      lg:grid-cols-[1fr_2fr_240px] gap-4 items-start">

        {/* Left column: submit form + progress */}
        <div className="space-y-4">
          <SubmitForm onJobCreated={handleJobCreated} />

          {job && running && (
            <ProgressTracker status={job.status} />
          )}

          {/* Network/poll error */}
          {error && (
            <div className="card border-red-800/40 bg-red-900/10">
              <p className="text-xs font-body text-red-400">
                Polling error: {error}
              </p>
            </div>
          )}
        </div>

        {/* Centre column: results */}
        <div className="space-y-4">
          {report ? (
            <>
              <VerdictCard report={report} />

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <ConfidenceGauge probFake={report.weighted_prob_fake ?? 0} />

                {/* Quick stats panel */}
                <div className="card">
                  <p className="section-label">Temporal summary</p>
                  <div className="space-y-2.5">
                    <Row label="Temporal verdict"
                         value={report.temporal?.temporal_verdict} />
                    <Row label="Suspicious frames"
                         value={`${((report.temporal?.suspicious_frame_ratio ?? 0) * 100).toFixed(1)}%`} />
                    <Row label="Run-length score"
                         value={report.temporal?.run_length_score?.toFixed(2)} />
                    <Row label="Peak frame"
                         value={`#${report.temporal?.peak_frame_idx} (${((report.temporal?.peak_score ?? 0) * 100).toFixed(1)}%)`} />
                    <Row label="Analysis time"
                         value={`${report.elapsed_seconds?.toFixed(1)}s`} />
                    <Row label="Generated"
                         value={report.generated_at
                           ? new Date(report.generated_at).toLocaleTimeString()
                           : "—"
                         } />
                  </div>
                </div>
              </div>

              <TemporalChart temporal={report.temporal} />
              <HeatmapViewer heatmaps={report.heatmaps ?? []} jobId={activeJobId} />
            </>
          ) : (
            /* Empty state when no job is selected */
            !job && (
              <div className="card flex flex-col items-center justify-center
                              min-h-64 text-center border-dashed">
                <span className="text-4xl mb-4 opacity-30">🎬</span>
                <p className="text-sm font-body text-slate-600">
                  Submit a URL or upload a video to begin analysis.
                </p>
              </div>
            )
          )}

          {/* Show progress tracker in centre column on large screens too */}
          {job && running && (
            <div className="xl:hidden">
              <ProgressTracker status={job.status} />
            </div>
          )}
        </div>

        {/* Right column: job history */}
        <div className="hidden lg:block">
          <JobHistory
            jobs={history}
            activeJobId={activeJobId}
            onSelect={(h) => setActiveJobId(h.job_id)}
          />
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between items-baseline gap-2">
      <span className="text-xs font-body text-slate-500 flex-shrink-0">{label}</span>
      <span className="text-xs font-body text-slate-300 text-right truncate">{value ?? "—"}</span>
    </div>
  );
}