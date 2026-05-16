// frontend/src/components/TemporalChart.jsx
import {
  CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

/**
 * Recharts line chart showing per-frame P(fake) over time.
 *
 * Two lines are drawn:
 *   - Raw scores (faint, showing original model output per frame)
 *   - Smoothed scores (bright, Gaussian-smoothed — what Day 4 computed)
 *
 * A dashed red reference line at y=0.5 marks the decision boundary.
 * Reference lines mark the start of each suspicious window in amber.
 */
export default function TemporalChart({ temporal }) {
  if (!temporal?.frame_indices?.length) {
    return (
      <div className="card">
        <p className="section-label">Frame-level scores</p>
        <p className="text-sm font-body text-slate-600">No temporal data available.</p>
      </div>
    );
  }

  // Build the chart data array from the two parallel arrays in the report
  const data = temporal.frame_indices.map((frameIdx, i) => ({
    frame:    frameIdx,
    raw:      +(temporal.raw_scores[i] ?? 0).toFixed(4),
    smoothed: +(temporal.smoothed_scores[i] ?? 0).toFixed(4),
  }));

  // Custom tooltip shown on hover — displays frame index and both scores
  function CustomTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-surface-700 border border-slate-600 rounded-lg
                      px-3 py-2 text-xs font-body shadow-xl">
        <p className="text-slate-400 mb-1">Frame #{label}</p>
        {payload.map((p) => (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {(p.value * 100).toFixed(1)}%
          </p>
        ))}
      </div>
    );
  }

  return (
    <div className="card animate-fade-in">
      <p className="section-label">Frame-level scores over time</p>

      {/* Legend */}
      <div className="flex gap-4 mb-4">
        <LegendItem colour="#f59e0b" label="Smoothed score" />
        <LegendItem colour="#475569" label="Raw score" dashed />
        <LegendItem colour="#ef4444" label="Decision boundary (50%)" dashed />
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1c1f26" />

          <XAxis
            dataKey="frame"
            tick={{ fill: "#64748b", fontSize: 10, fontFamily: "IBM Plex Mono" }}
            label={{ value: "Frame index", position: "insideBottom",
                     offset: -2, fill: "#475569", fontSize: 10 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#64748b", fontSize: 10, fontFamily: "IBM Plex Mono" }}
            width={42}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Decision boundary */}
          <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="4 4"
                         strokeOpacity={0.6} />

          {/* Suspicious window start lines */}
          {(temporal.suspicious_windows ?? []).map((win, i) => (
            <ReferenceLine key={i} x={win.start_frame}
                           stroke="#f59e0b" strokeDasharray="2 4"
                           strokeOpacity={0.5} />
          ))}

          {/* Raw score — muted, behind the smoothed line */}
          <Line type="monotone" dataKey="raw" name="Raw"
                stroke="#475569" strokeWidth={1} dot={false}
                strokeDasharray="3 3" />

          {/* Smoothed score — prominent, amber */}
          <Line type="monotone" dataKey="smoothed" name="Smoothed"
                stroke="#f59e0b" strokeWidth={2.5} dot={false}
                activeDot={{ r: 4, fill: "#f59e0b" }} />
        </LineChart>
      </ResponsiveContainer>

      {/* Suspicious windows summary */}
      {temporal.suspicious_windows?.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700">
          <p className="text-xs font-body text-slate-500 mb-2">Suspicious windows detected:</p>
          <div className="flex flex-wrap gap-2">
            {temporal.suspicious_windows.map((win, i) => (
              <span key={i}
                    className="text-xs font-body text-amber-400 bg-amber-500/10
                               border border-amber-500/20 rounded-md px-2 py-1">
                Frames {win.start_frame}–{win.end_frame} &nbsp;
                (mean {(win.mean_prob_fake * 100).toFixed(1)}%)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LegendItem({ colour, label, dashed }) {
  return (
    <div className="flex items-center gap-1.5">
      <svg width="20" height="10">
        <line x1="0" y1="5" x2="20" y2="5"
              stroke={colour} strokeWidth="2"
              strokeDasharray={dashed ? "4 3" : "none"} />
      </svg>
      <span className="text-xs font-body text-slate-500">{label}</span>
    </div>
  );
}