const CACHE_NAME = "keiba-lab-web-v29";
const CORE = ["./", "./index.html", "./manifest.webmanifest"];
self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.includes("/data/")) {
    event.respondWith(fetch(request, { cache: "no-store" }));
    return;
  }
  // Always check the network for the app shell so phone browsers do not keep
  // serving an older index.html after a successful GitHub Pages deployment.
  if (request.mode === "navigate" || url.pathname.endsWith("/index.html") || url.pathname === new URL("./", self.location.href).pathname) {
    event.respondWith(fetch(request, { cache: "no-store" }).then(response => {
      if (response.ok) {
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put("./index.html", copy));
      }
      return response;
    }).catch(() => caches.match("./index.html")));
    return;
  }
  event.respondWith(caches.match(request).then(cached => cached || fetch(request).then(response => {
    if (response.ok) {
      const copy = response.clone();
      caches.open(CACHE_NAME).then(cache => cache.put(request, copy));
    }
    return response;
  }).catch(() => caches.match("./index.html"))));
});
