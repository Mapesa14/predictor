/** Where the API lives, which is not the same place in every build.
 *
 * On the web the app and the service share an origin, so a relative "/api/..."
 * is right and the dev server proxies it. Inside the native shell there is no
 * such origin: Capacitor serves the bundle from https://localhost, and a
 * relative path resolves to the phone itself. Every request has to be absolute
 * and has to be HTTPS, because Android blocks cleartext by default.
 *
 * So the base is baked in at build time from VITE_API_BASE, and can be changed
 * at runtime from the Settings screen - which matters more than it sounds:
 * a phone build that points at the wrong host is otherwise a reinstall, and
 * during development you want it aimed at a laptop on the LAN.
 */
const BUILD_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");
const KEY = "predictor.apiBase";

export function apiBase() {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved) return saved.replace(/\/+$/, "");
  } catch {
    // Private mode, or a WebView with site data blocked. Fall through to the
    // build-time value rather than taking the whole app down for a preference.
  }
  return BUILD_BASE;
}

export function setApiBase(v) {
  try {
    const clean = String(v || "").trim().replace(/\/+$/, "");
    if (clean) localStorage.setItem(KEY, clean);
    else localStorage.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

/** Absolute when we have a base, relative when we don't. */
export function api(path) {
  const p = path.startsWith("/") ? path : "/" + path;
  return apiBase() + p;
}

/** True in the Capacitor shell. Drives the things a browser does not need:
 *  no service worker, a hardware back button, and safe-area padding. */
export const isNative = typeof window !== "undefined" &&
  !!(window.Capacitor && window.Capacitor.isNativePlatform &&
     window.Capacitor.isNativePlatform());

export function nativePlatform() {
  try {
    return window.Capacitor.getPlatform();
  } catch {
    return "web";
  }
}
