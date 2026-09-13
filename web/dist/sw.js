/* Bumped by the build (vite defines 1789286311110 into the app; this worker is
   copied verbatim, so the id is stamped in at build time by build-sw.mjs).
   A fixed name meant a deploy never reached anyone already using the app:
   the shell was served from cache forever. */
const CACHE = "predictor-1789286311110";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (url.pathname.startsWith("/api")) {
    // network-first: the numbers are the whole point, but a stale copy beats nothing
    e.respondWith(
      fetch(e.request)
        .then((r) => {
          const copy = r.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return r;
        })
        .catch(() =>
          caches.match(e.request).then((m) => m || Response.error())
        )
    );
  } else {
    // shell-first
    e.respondWith(
      caches.match(e.request).then(
        (m) =>
          m ||
          fetch(e.request).then((r) => {
            const copy = r.clone();
            caches.open(CACHE).then((c) => c.put(e.request, copy));
            return r;
          })
      )
    );
  }
});