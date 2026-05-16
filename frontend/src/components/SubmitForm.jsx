// frontend/src/components/SubmitForm.jsx
import { Link, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { submitFile, submitURL } from "../api/client.js";

const ACCEPTED_TYPES = ".mp4,.mov,.avi,.webm,.mkv,.m4v";
const MAX_MB         = 500;

/**
 * Dual-mode submission form: URL paste or drag-and-drop file upload.
 *
 * Tab state controls which input is shown.
 * On submit, calls the appropriate API function, then calls onJobCreated
 * with the returned job object so the parent can begin polling.
 *
 * Error handling: errors from the API are caught and shown inline.
 * The submit button is disabled while a submission is in-flight.
 */
export default function SubmitForm({ onJobCreated }) {
  const [tab,       setTab]       = useState("url");   // "url" | "upload"
  const [url,       setUrl]       = useState("");
  const [file,      setFile]      = useState(null);
  const [dragOver,  setDragOver]  = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const fileInputRef = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let job;
      if (tab === "url") {
        if (!url.trim()) throw new Error("Please enter a URL.");
        job = await submitURL(url.trim());
      } else {
        if (!file) throw new Error("Please select a video file.");
        job = await submitFile(file);
      }
      onJobCreated(job);
      // Reset form
      setUrl("");
      setFile(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) validateAndSetFile(dropped);
  }

  function validateAndSetFile(f) {
    const ext      = f.name.split(".").pop().toLowerCase();
    const allowed  = ["mp4","mov","avi","webm","mkv","m4v"];
    if (!allowed.includes(ext)) {
      setError(`Unsupported format ".${ext}". Allowed: ${allowed.join(", ")}.`);
      return;
    }
    const sizeMB = f.size / 1_000_000;
    if (sizeMB > MAX_MB) {
      setError(`File is ${sizeMB.toFixed(0)}MB — exceeds the ${MAX_MB}MB limit.`);
      return;
    }
    setError(null);
    setFile(f);
  }

  return (
    <div className="card">
      <p className="section-label">Analyze a video</p>

      {/* Mode tabs */}
      <div className="flex gap-1 mb-5 bg-surface-900 rounded-lg p-1 w-fit">
        {[
          { id: "url",    label: "URL",    icon: <Link    size={13} /> },
          { id: "upload", label: "Upload", icon: <Upload  size={13} /> },
        ].map(({ id, label, icon }) => (
          <button
            key={id}
            onClick={() => { setTab(id); setError(null); }}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md
                        text-xs font-sans font-medium transition-all duration-150
                        ${tab === id
                          ? "bg-amber-500 text-surface-900"
                          : "text-slate-400 hover:text-slate-200"
                        }`}
          >
            {icon}{label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* URL input */}
        {tab === "url" && (
          <div>
            <label className="text-xs font-body text-slate-500 block mb-1.5">
              YouTube, Vimeo, TikTok, or direct .mp4 URL
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="input-field"
              disabled={loading}
            />
          </div>
        )}

        {/* File drag-drop zone */}
        {tab === "upload" && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true);  }}
            onDragLeave={()  => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-xl p-8 text-center
                        cursor-pointer transition-all duration-200
                        ${dragOver
                          ? "border-amber-500 bg-amber-500/5"
                          : "border-slate-700 hover:border-slate-500 hover:bg-surface-700/40"
                        }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_TYPES}
              className="hidden"
              onChange={(e) => e.target.files?.[0] && validateAndSetFile(e.target.files[0])}
            />

            <Upload size={28} className="mx-auto mb-3 text-slate-600" />

            {file ? (
              <div>
                <p className="text-sm font-body text-slate-200 font-medium">{file.name}</p>
                <p className="text-xs font-body text-slate-500 mt-1">
                  {(file.size / 1_000_000).toFixed(1)} MB
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm font-body text-slate-400">
                  Drop a video file here or click to browse
                </p>
                <p className="text-xs font-body text-slate-600 mt-1">
                  MP4, MOV, AVI, WebM, MKV — max {MAX_MB}MB
                </p>
              </div>
            )}
          </div>
        )}

        {/* Error message */}
        {error && (
          <p className="text-xs font-body text-red-400 bg-red-900/20
                        border border-red-800/40 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        {/* Submit */}
        <button type="submit" disabled={loading} className="btn-primary w-full">
          {loading ? "Submitting…" : "Analyze video"}
        </button>
      </form>
    </div>
  );
}