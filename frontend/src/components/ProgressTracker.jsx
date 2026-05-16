// frontend/src/components/ProgressTracker.jsx
import { CheckCircle, Circle, Loader } from "lucide-react";
import { statusMeta } from "../utils/formatting.js";

// The pipeline stages in the order they execute.
// Each object maps to one row in the tracker.
const STAGES = [
  { key: "DOWNLOADING",       label: "Download & validate video" },
  { key: "EXTRACTING_FRAMES", label: "Extract key frames"        },
  { key: "DETECTING_FACES",   label: "Detect & align faces"      },
  { key: "RUNNING_INFERENCE", label: "Run deepfake model"        },
  { key: "ANALYZING",         label: "Temporal analysis & CAM"   },
  { key: "DONE",              label: "Analysis complete"         },
];

function stageState(stageKey, currentStatus) {
  const currentIdx = STAGES.findIndex((s) => s.key === currentStatus);
  const stageIdx   = STAGES.findIndex((s) => s.key === stageKey);

  if (currentStatus === "FAILED") {
    return stageIdx < currentIdx ? "done" : "idle";
  }
  if (stageIdx < currentIdx)  return "done";
  if (stageIdx === currentIdx) return "active";
  return "idle";
}

export default function ProgressTracker({ status }) {
  return (
    <div className="card animate-fade-in">
      <p className="section-label">Pipeline progress</p>
      <ol className="space-y-3">
        {STAGES.map((stage) => {
          const state = stageState(stage.key, status);
          const meta  = statusMeta(stage.key);

          return (
            <li key={stage.key}
                className="flex items-center gap-3">
              {/* Status icon */}
              <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
                {state === "done"   && <CheckCircle size={18} className="text-emerald-400" />}
                {state === "active" && <Loader      size={18} className={`${meta.colour} animate-spin`} />}
                {state === "idle"   && <Circle      size={18} className="text-slate-700" />}
              </span>

              {/* Stage label */}
              <span className={`text-sm font-body transition-colors duration-300
                ${state === "done"   ? "text-slate-400 line-through decoration-slate-600" : ""}
                ${state === "active" ? `${meta.colour} font-medium` : ""}
                ${state === "idle"   ? "text-slate-700" : ""}
              `}>
                {stage.label}
              </span>

              {/* Active indicator */}
              {state === "active" && (
                <span className={`ml-auto text-xs font-body ${meta.colour} opacity-75`}>
                  in progress…
                </span>
              )}
            </li>
          );
        })}
      </ol>

      {/* FAILED state banner */}
      {status === "FAILED" && (
        <div className="mt-4 p-3 bg-red-900/20 border border-red-800/40
                        rounded-lg text-sm text-red-400 font-body">
          Pipeline failed. Check the error message below.
        </div>
      )}
    </div>
  );
}