// frontend/src/components/StatusBadge.jsx
import { statusMeta } from "../utils/formatting.js";

/**
 * A small pill showing the current job status with a pulsing dot.
 * The dot pulses while the job is running and is static when terminal.
 */
export default function StatusBadge({ status, running = false }) {
  const meta = statusMeta(status);
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1
                      rounded-full text-xs font-body font-medium
                      ${meta.colour} ${meta.bg} bg-opacity-20`}>
      <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}
                        ${running ? "animate-pulse" : ""}`} />
      {meta.label}
    </span>
  );
}