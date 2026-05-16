// frontend/src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";   // Tailwind directives live here

ReactDOM.createRoot(document.getElementById("root")).render(
  // StrictMode renders components twice in development to surface
  // side-effects. This is intentional and only happens in dev mode.
  <React.StrictMode>
    <App />
  </React.StrictMode>
);