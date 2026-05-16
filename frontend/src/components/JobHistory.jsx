// frontend/src/components/JobHistory.jsx
import { Clock } from "lucide-react";
import { fmtTime, shortSource, verdictMeta } from "../utils/formatting.js";
import StatusBadge from "./StatusBadge.jsx";

/**
 * Sidebar list of recent jobs stored in component state (passed down from App).
 * Each row is clickable to reload that job's results into the main panel.
 * We persist jobs in localStorage so history survives a page refresh.
 */
export default function JobHistory({ jobs = [], onSelect, activeJobId }) {
  if (!jobs.length) {
    return (
      <div className="card h-full">
        <p className="section-label">Recent analyses</p>
        <p className="text-xs font-body text-slate-700 mt-2">
          No analyses yet — submit a video to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="card h-full">
      <p className="section-label">Recent analyses</p>
      <ul className="space-y-2">
        {jobs.map((job) => {
          const isActive = job.job_id === activeJobId;
          const meta     = job.verdict ? verdictMeta(job.verdict) : null;

          return (
            <li key={job.job_id}>
              <button
                onClick={() => onSelect(job)}
                className={`w-full text-left rounded-lg p-3 border transition-all duration-150
                            ${isActive
                              ? "border-amber-500/40 bg-amber-500/5"
                              : "border-slate-700/50 hover:border-slate-600 hover:bg-surface-700/40"
                            }`}
              >
                {/* Source name */}
                <p className="text-xs font-body text-slate-300 truncate mb-1.5">
                  {shortSource(job.source ?? job.job_id)}
                </p>

                <div className="flex items-center justify-between gap-2">
                  <StatusBadge status={job.status} />

                  {/* Verdict badge (shown once job is done) */}
                  {job.verdict && (
                    <span className={`text-xs font-body font-semibold ${meta?.colour}`}>
                      {job.verdict}
                    </span>
                  )}
                </div>

                {/* Timestamp */}
                <div className="flex items-center gap-1 mt-1.5">
                  <Clock size={10} className="text-slate-700" />
                  <span className="text-xs font-body text-slate-700">
                    {fmtTime(job.created_at)}
                  </span>
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}