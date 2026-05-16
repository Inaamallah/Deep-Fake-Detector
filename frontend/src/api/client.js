// frontend/src/api/client.js
/**
 * API client for the FastAPI deepfake detection backend.
 *
 * All network calls live here. This single-responsibility module means:
 *  - The URL and auth header are defined exactly once.
 *  - Every component imports named functions, never raw fetch() calls.
 *  - If the backend URL changes, you change it in one place.
 *
 * The Vite dev proxy (vite.config.js) forwards /api/* to localhost:8000,
 * so we never hard-code the host — just the path prefix.
 *
 * The API key is read from the VITE_API_KEY environment variable.
 * Create frontend/.env.local with: VITE_API_KEY=dev-key-change-in-production
 * NEVER commit .env.local — it is in .gitignore.
 */

const API_KEY = import.meta.env.VITE_API_KEY ?? "dev-key-change-in-production";
const BASE    = "/api/v1";

// Shared fetch wrapper: attaches auth header and parses JSON.
// Throws a descriptive Error on non-2xx responses so components
// can catch a single error type rather than checking status codes everywhere.
async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      "X-API-Key":    API_KEY,
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch (_) {
      // Response body was not JSON — keep the status code message.
    }
    throw new Error(detail);
  }

  return res.json();
}

// ── Job submission ──────────────────────────────────────────────────────────

/**
 * Submit a URL for analysis.
 * Returns a JobResponse with job_id and status "PENDING".
 */
export async function submitURL(url) {
  return apiFetch("/jobs/url", {
    method: "POST",
    body:   JSON.stringify({ url }),
  });
}

/**
 * Upload a local video file for analysis.
 * Uses multipart/form-data — do NOT set Content-Type manually here;
 * the browser sets the boundary automatically when you pass FormData.
 */
export async function submitFile(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/jobs/upload`, {
    method:  "POST",
    headers: { "X-API-Key": API_KEY },  // no Content-Type — browser sets it
    body:    form,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try   { detail = (await res.json()).detail ?? detail; }
    catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

// ── Job polling ─────────────────────────────────────────────────────────────

/**
 * Fetch the current status of a job.
 * Returns a JobResponse.
 */
export async function getJobStatus(jobId) {
  return apiFetch(`/jobs/${jobId}`);
}

/**
 * Fetch the final analysis report for a completed job.
 * Returns a ResultResponse — report field is null until status is DONE.
 */
export async function getResult(jobId) {
  return apiFetch(`/jobs/${jobId}/result`);
}

/**
 * Build the URL for a heatmap image.
 * Used as the `src` attribute of an <img> tag — no fetch needed.
 */
export function heatmapImageURL(jobId, filename) {
  const name = filename.split("/").pop();  // strip any directory prefix
  return `${BASE}/jobs/${jobId}/heatmaps/${name}`;
}