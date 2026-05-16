// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      // Design token: every colour in the UI is derived from these variables.
      // This means you can retheme the entire application by changing one file.
      colors: {
        // Background scale — darkest to lightest
        surface: {
          900: "#0d0e10",
          800: "#14161a",
          700: "#1c1f26",
          600: "#252832",
        },
        // Amber — used for DEEPFAKE verdicts and warnings
        amber: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
        // Emerald — used for REAL verdicts and safe states
        emerald: {
          400: "#34d399",
          500: "#10b981",
        },
        // Slate — used for neutral text and borders
        slate: {
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
        },
      },
      fontFamily: {
        // Display: used for the verdict headline — loud, unmistakable
        display: ["'Bebas Neue'", "sans-serif"],
        // Body: used for all readable text — clean, technical
        body:    ["'IBM Plex Mono'", "monospace"],
        // Sans: used for UI chrome like buttons and labels
        sans:    ["'DM Sans'", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in":    "fadeIn 0.4s ease-out",
        "slide-up":   "slideUp 0.5s ease-out",
      },
      keyframes: {
        fadeIn:  { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideUp: { "0%": { opacity: "0", transform: "translateY(16px)" },
                   "100%": { opacity: "1", transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};