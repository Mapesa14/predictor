import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

/** One bad field should cost one screen, not the whole app.
 *
 * A single market rendered as an object once white-screened everything,
 * including the fixtures that were fine. */
class Boundary extends React.Component {
  constructor(p) {
    super(p);
    this.state = { err: null };
  }
  static getDerivedStateFromError(err) {
    return { err };
  }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <div className="wrap">
        <h1>Something on this screen broke</h1>
        <p className="note">
          The rest of the app still works. Nothing here is a prediction — this
          is an error, not a forecast.
        </p>
        <p className="note"><code>{String(this.state.err && this.state.err.message)}</code></p>
        <p><a className="back" href="#/" onClick={() => location.reload()}>← Back to today's fixtures</a></p>
      </div>
    );
  }
}

createRoot(document.getElementById("root")).render(
  <Boundary><App /></Boundary>
);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}