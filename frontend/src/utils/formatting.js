// frontend/src/utils/formatting.js
/**
 * Pure formatting helpers — no React, no side effects.
 * Every function takes raw data and returns a display string or object.
 */

/** Format a 0–1 probability as a percentage string: 0.857 → "85.7%" */
export function fmtPercent(value, decimals = 1) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a Unix timestamp as a short local time string: 1700000000 → "14:23:45" */
export function fmtTime(unixSeconds) {
  if (!unixSeconds) return "—";
  return new Date(unixSeconds * 1000).toLocaleTimeString([], {
    hour:   "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Format elapsed seconds with appropriate units: 90 → "1m 30s", 5 → "5s" */
export function fmtDuration(seconds) {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

/**
 * Map a job status string to its display label, colour classes, and
 * a short description shown in the progress tracker.
 */
export function statusMeta(status) {
  const map = {
    PENDING:           { label: "Queued",              colour: "text-slate-400",  bg: "bg-slate-700",  dot: "bg-slate-400"  },
    DOWNLOADING:       { label: "Downloading",         colour: "text-blue-400",   bg: "bg-blue-900",   dot: "bg-blue-400"   },
    EXTRACTING_FRAMES: { label: "Extracting Frames",   colour: "text-violet-400", bg: "bg-violet-900", dot: "bg-violet-400" },
    DETECTING_FACES:   { label: "Detecting Faces",     colour: "text-sky-400",    bg: "bg-sky-900",    dot: "bg-sky-400"    },
    RUNNING_INFERENCE: { label: "Running Inference",   colour: "text-amber-400",  bg: "bg-amber-900",  dot: "bg-amber-400"  },
    ANALYZING:         { label: "Analyzing",           colour: "text-orange-400", bg: "bg-orange-900", dot: "bg-orange-400" },
    DONE:              { label: "Complete",            colour: "text-emerald-400",bg: "bg-emerald-900",dot: "bg-emerald-400"},
    FAILED:            { label: "Failed",              colour: "text-red-400",    bg: "bg-red-900",    dot: "bg-red-400"    },
  };
  return map[status] ?? map.PENDING;
}

/** Returns true for statuses where the pipeline is still running. */
export function isRunning(status) {
  return !["DONE", "FAILED"].includes(status);
}

/**
 * Map a verdict string to its display colour and icon name.
 * Icons are lucide-react names — imported by the component that uses this.
 */
export function verdictMeta(verdict) {
  if (verdict === "DEEPFAKE") return { colour: "text-amber-400",  bg: "bg-amber-500/10",  border: "border-amber-500/30", icon: "AlertTriangle" };
  if (verdict === "REAL")     return { colour: "text-emerald-400",bg: "bg-emerald-500/10",border: "border-emerald-500/30",icon: "ShieldCheck"   };
  return                             { colour: "text-slate-400",  bg: "bg-slate-700/20",  border: "border-slate-600",    icon: "HelpCircle"    };
}

/** Shorten a URL or file path to a readable label for the history list. */
export function shortSource(source) {
  if (!source) return "Unknown source";
  try {
    const url = new URL(source);
    return url.hostname + (url.pathname.length > 1 ? url.pathname.slice(0, 30) : "");
  } catch (_) {
    // Local file path — return just the filename
    return source.split(/[\\/]/).pop();
  }
}