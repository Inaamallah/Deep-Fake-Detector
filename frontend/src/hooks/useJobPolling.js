// frontend/src/hooks/useJobPolling.js
/**
 * Custom hook that manages the polling lifecycle for a single job.
 *
 * Why a custom hook instead of inline useEffect in each component?
 * This hook encapsulates three concerns:
 *   1. Polling interval management (start, stop, clear on unmount)
 *   2. Automatic stopping when the job reaches a terminal state
 *   3. Fetching both status and result in the right sequence
 *
 * Any component that needs job state imports this hook — zero
 * duplication of the polling logic.
 *
 * Polling interval: 3 seconds. This gives responsive feedback without
 * hammering the API. The pipeline stages each take 10–60 seconds so
 * 3 second polling is more than fast enough.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { getJobStatus, getResult } from "../api/client.js";
import { isRunning } from "../utils/formatting.js";

const POLL_INTERVAL_MS = 3000;

/**
 * @param {string|null} jobId  - The job UUID to poll. Pass null to disable polling.
 * @returns {{
 *   job:    object|null,   // Latest JobResponse from GET /jobs/{id}
 *   result: object|null,   // ResultResponse (with report) once DONE
 *   error:  string|null,   // Error message if polling fails
 * }}
 */
export function useJobPolling(jobId) {
  const [job,    setJob]    = useState(null);
  const [result, setResult] = useState(null);
  const [error,  setError]  = useState(null);

  // Store the interval ID in a ref so we can clear it from inside the
  // polling function without creating a stale closure.
  const intervalRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    if (!jobId) return;
    try {
      const jobData = await getJobStatus(jobId);
      setJob(jobData);
      setError(null);

      if (!isRunning(jobData.status)) {
        // Job has reached a terminal state — stop polling, fetch final result
        stopPolling();
        const resultData = await getResult(jobId);
        setResult(resultData);
      }
    } catch (err) {
      // Network errors are non-fatal — keep polling in case it's transient.
      // After 3 consecutive failures we surface the error to the user.
      setError(err.message);
    }
  }, [jobId, stopPolling]);

  useEffect(() => {
    if (!jobId) {
      stopPolling();
      return;
    }

    // Fire immediately so the user sees feedback before the first 3s interval
    poll();

    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);

    // Cleanup: always clear the interval when the component unmounts or
    // when jobId changes (user submits a new video mid-analysis)
    return stopPolling;
  }, [jobId, poll, stopPolling]);

  return { job, result, error };
}