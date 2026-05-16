// frontend/src/components/VerdictCard.jsx
import { AlertTriangle, HelpCircle, ShieldCheck } from "lucide-react";
import { fmtPercent, verdictMeta } from "../utils/formatting.js";

const ICONS = { AlertTriangle, ShieldCheck, HelpCircle };

/**
 * The hero component of the results page.
 * Large, unmissable verdict display with supporting statistics.
 * Font size is intentionally dramatic — this is the answer the user came for.
 */
export default function VerdictCard({ report }) {
  const verdict = report?.verdict ?? "INCONCLUSIVE";
  const meta    = verdictMeta(verdict);
  const Icon    = ICONS[meta.icon];

  const ci       = report?.confidence_interval ?? {};
  const temporal = report?.temporal ?? {};

  return (
    <div className={`card border ${meta.border} ${meta.bg} animate-slide-up`}>
      {/* Verdict headline */}
      <div className="flex items-center gap-4 mb-6">
        <div className={`p-3 rounded-xl ${meta.bg} border ${meta.border}`}>
          <Icon size={28} className={meta.colour} />
        </div>
        <div>
          <p className="section-label mb-0">Verdict</p>
          <h2 className={`font-display text-6xl leading-none ${meta.colour}`}>
            {verdict}
          </h2>
        </div>
      </div>

      {/* Key statistics grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Weighted P(fake)"
          value={fmtPercent(report?.weighted_prob_fake)}
          highlight={meta.colour}
        />
        <Stat
          label="95% CI"
          value={
            ci.ci_lower_95 != null
              ? `${fmtPercent(ci.ci_lower_95)} – ${fmtPercent(ci.ci_upper_95)}`
              : "—"
          }
        />
        <Stat
          label="Faces scored"
          value={report?.total_faces_scored ?? "—"}
        />
        <Stat
          label="Temporal pattern"
          value={temporal.temporal_verdict ?? "—"}
        />
      </div>

      {/* Bootstrap interpretation */}
      {ci.interpretation && (
        <p className="mt-4 text-xs font-body text-slate-500 leading-relaxed
                      border-t border-slate-700 pt-4">
          {ci.interpretation}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }) {
  return (
    <div className="bg-surface-900/60 rounded-lg p-3">
      <p className="text-xs font-body text-slate-500 mb-1">{label}</p>
      <p className={`text-sm font-body font-medium ${highlight ?? "text-slate-200"}`}>
        {value}
      </p>
    </div>
  );
}