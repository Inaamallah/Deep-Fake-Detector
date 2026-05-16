// frontend/src/components/ConfidenceGauge.jsx
/**
 * SVG arc gauge showing weighted P(fake) as a needle on a semicircle.
 *
 * Why SVG rather than a third-party gauge library?
 * Libraries add bundle weight and often have styling conflicts with Tailwind.
 * This gauge is 60 lines of pure SVG math — more predictable and
 * more customisable than any library dependency.
 *
 * The gauge arc spans 180° (a half-circle), left = 0% fake, right = 100% fake.
 * A 0.5 threshold line is drawn at the 12 o'clock position.
 */
export default function ConfidenceGauge({ probFake = 0 }) {
  // SVG coordinate system
  const cx = 100, cy = 100, r = 70;

  // Convert a [0,1] probability to an angle in degrees.
  // 0.0 → 180° (far left), 0.5 → 0° (top), 1.0 → 0° (far right).
  // We map [0,1] → [-180°, 0°] then add 180° offset to get the right visual.
  function probToAngle(p) {
    return 180 - p * 180;
  }

  function polarToXY(angleDeg, radius) {
    const rad = (angleDeg * Math.PI) / 180;
    return {
      x: cx + radius * Math.cos(rad),
      y: cy - radius * Math.sin(rad),   // SVG y-axis is inverted
    };
  }

  function arcPath(startAngle, endAngle, radius) {
    const start    = polarToXY(startAngle, radius);
    const end      = polarToXY(endAngle,   radius);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  }

  const needleAngle = probToAngle(probFake);
  const needleTip   = polarToXY(needleAngle, r - 10);
  const needleBase1 = polarToXY(needleAngle + 90, 6);
  const needleBase2 = polarToXY(needleAngle - 90, 6);

  // Colour interpolation: green at 0, amber at 0.5, red at 1
  function gaugeColour(p) {
    if (p < 0.4) return "#10b981";    // emerald
    if (p < 0.6) return "#f59e0b";    // amber
    return "#ef4444";                  // red
  }

  const colour = gaugeColour(probFake);

  return (
    <div className="card flex flex-col items-center">
      <p className="section-label w-full">Confidence gauge</p>

      <svg viewBox="0 0 200 115" className="w-full max-w-xs">
        {/* Background arc */}
        <path d={arcPath(0, 180, r)}
              fill="none" stroke="#1c1f26" strokeWidth="14" strokeLinecap="round" />

        {/* Filled arc showing the probability */}
        <path d={arcPath(180, 180 - probFake * 180, r)}
              fill="none" stroke={colour} strokeWidth="14"
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.8s ease-out" }} />

        {/* 0.5 threshold tick mark */}
        <line x1={cx} y1={cy - r + 10} x2={cx} y2={cy - r - 4}
              stroke="#475569" strokeWidth="2" strokeLinecap="round" />
        <text x={cx} y={cy - r - 8} textAnchor="middle"
              fontSize="6" fill="#475569" fontFamily="IBM Plex Mono">
          50%
        </text>

        {/* Needle */}
        <polygon
          points={`${needleTip.x},${needleTip.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
          fill={colour}
          style={{ transition: "all 0.8s cubic-bezier(0.34,1.56,0.64,1)" }}
        />
        {/* Needle pivot */}
        <circle cx={cx} cy={cy} r={5} fill="#252832" stroke={colour} strokeWidth="2" />

        {/* Percentage label */}
        <text x={cx} y={cy + 22} textAnchor="middle"
              fontSize="16" fill={colour} fontFamily="Bebas Neue" letterSpacing="1">
          {Math.round(probFake * 100)}%
        </text>
        <text x={cx} y={cy + 32} textAnchor="middle"
              fontSize="5.5" fill="#64748b" fontFamily="IBM Plex Mono">
          P(FAKE)
        </text>

        {/* Scale labels */}
        <text x="26" y="108" textAnchor="middle" fontSize="6"
              fill="#64748b" fontFamily="IBM Plex Mono">REAL</text>
        <text x="174" y="108" textAnchor="middle" fontSize="6"
              fill="#64748b" fontFamily="IBM Plex Mono">FAKE</text>
      </svg>
    </div>
  );
}