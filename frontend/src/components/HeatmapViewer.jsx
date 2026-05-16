// frontend/src/components/HeatmapViewer.jsx
import { useState } from "react";
import { heatmapImageURL } from "../api/client.js";
import { fmtPercent } from "../utils/formatting.js";

/**
 * Grid of the top suspicious frame heatmaps with a lightbox overlay.
 *
 * Each tile shows the overlay image (face + Grad-CAM heat) with the
 * frame's P(fake) score as a badge. Clicking a tile opens a full-size
 * lightbox showing both the raw heatmap and the overlay side by side.
 */
export default function HeatmapViewer({ heatmaps = [], jobId }) {
  const [selected, setSelected] = useState(null);

  if (!heatmaps.length) {
    return (
      <div className="card">
        <p className="section-label">Grad-CAM heatmaps</p>
        <p className="text-sm font-body text-slate-600">
          No heatmaps generated — this happens when no faces were detected
          in the top-scoring frames.
        </p>
      </div>
    );
  }

  return (
    <div className="card animate-fade-in">
      <p className="section-label">
        Grad-CAM heatmaps
        <span className="ml-2 text-slate-600 normal-case tracking-normal font-sans font-normal text-xs">
          (top {heatmaps.length} suspicious frames — click to enlarge)
        </span>
      </p>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {heatmaps.map((h, i) => {
          const overlayFile   = h.overlay_path.split(/[\\/]/).pop();
          const overlayURL    = heatmapImageURL(jobId, overlayFile);

          return (
            <button
              key={i}
              onClick={() => setSelected(h)}
              className="group relative rounded-lg overflow-hidden border
                         border-slate-700 hover:border-amber-500/50
                         transition-all duration-200 hover:scale-105
                         focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              <img
                src={overlayURL}
                alt={`Frame ${h.frame_idx} heatmap overlay`}
                className="w-full aspect-square object-cover"
                loading="lazy"
              />
              {/* P(fake) badge */}
              <div className="absolute bottom-0 inset-x-0
                              bg-gradient-to-t from-black/80 to-transparent
                              px-2 py-1.5">
                <p className="text-xs font-body text-amber-400 font-medium">
                  {fmtPercent(h.prob_fake)}
                </p>
                <p className="text-xs font-body text-slate-400">
                  Frame #{h.frame_idx}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Lightbox overlay */}
      {selected && (
        <Lightbox heatmap={selected} jobId={jobId} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function Lightbox({ heatmap, jobId, onClose }) {
  const heatmapFile  = heatmap.heatmap_path.split(/[\\/]/).pop();
  const overlayFile  = heatmap.overlay_path.split(/[\\/]/).pop();

  return (
    // Clicking the backdrop closes the lightbox
    <div
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm
                 flex items-center justify-center p-4 animate-fade-in"
      onClick={onClose}
    >
      <div
        className="bg-surface-800 border border-slate-700 rounded-2xl
                   p-6 max-w-2xl w-full shadow-2xl"
        onClick={(e) => e.stopPropagation()}   // prevent backdrop click
      >
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="font-sans font-semibold text-slate-200">
              Frame #{heatmap.frame_idx}
            </h3>
            <p className="text-xs font-body text-amber-400 mt-0.5">
              P(fake) = {fmtPercent(heatmap.prob_fake)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-200 transition-colors text-lg font-body"
          >
            ✕
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-body text-slate-500 mb-2">Grad-CAM heatmap</p>
            <img
              src={heatmapImageURL(jobId, heatmapFile)}
              alt="Heatmap"
              className="w-full rounded-lg border border-slate-700"
            />
          </div>
          <div>
            <p className="text-xs font-body text-slate-500 mb-2">Face + heatmap overlay</p>
            <img
              src={heatmapImageURL(jobId, overlayFile)}
              alt="Overlay"
              className="w-full rounded-lg border border-slate-700"
            />
          </div>
        </div>

        <p className="mt-4 text-xs font-body text-slate-600 leading-relaxed">
          Hot regions (red/yellow) indicate the facial areas that most activated the model.
          Typical deepfake artifacts appear around eyes, mouth edges, and jaw blending boundaries.
        </p>
      </div>
    </div>
  );
}