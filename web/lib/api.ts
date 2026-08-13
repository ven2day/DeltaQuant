// Shared REST/WebSocket base URL + fetch helper for the dashboard backend.
//
// The REST API and WebSocket share a host/port (both served by the same FastAPI
// app) — derive the base URL from NEXT_PUBLIC_WS_URL rather than adding a second
// env var to keep in sync.
export function apiBaseUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws";
  // wss:// -> https://, ws:// -> http://. A naive `replace(/^ws/, "http")` breaks
  // on wss:// (matches just the "ws" prefix of "wss", producing "httpss://") —
  // match the full scheme including the optional trailing "s" instead.
  return wsUrl.replace(/^ws(s?):\/\//, "http$1://").replace(/\/ws$/, "");
}

// `credentials: "include"` is required on every request: the backend's session
// cookie is set by a different origin (backend host:port != frontend host:port
// whenever they're not proxied behind the same reverse proxy), so without it the
// browser silently drops the cookie and every request 401s even right after a
// successful login.
export function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  // A browser/network edge can leave fetch pending for a long time (for example
  // during a proxy reload).  Authentication uses this helper before rendering
  // anything, so an unbounded request would strand the whole UI on "Checking
  // session…".  Callers can provide their own signal for a different policy.
  if (init.signal) {
    return fetch(`${apiBaseUrl()}${path}`, { ...init, credentials: "include" });
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10_000);
  return fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    credentials: "include",
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeout));
}

export async function checkAuthenticated(): Promise<boolean> {
  try {
    const response = await apiFetch("/api/session");
    if (!response.ok) return false;
    const data = (await response.json()) as { authenticated: boolean };
    return data.authenticated === true;
  } catch {
    // Network/backend-down errors are not an auth decision — treat as "unknown,
    // let the page render and let the WebSocket/panel-level error states speak
    // for themselves" rather than bouncing to /login when the real problem is
    // the backend being unreachable.
    return true;
  }
}

export async function logout(): Promise<void> {
  try {
    await apiFetch("/api/logout", { method: "POST" });
  } catch {
    // Best-effort — the caller redirects to /login regardless, which is the
    // meaningful part of "logging out" from the browser's point of view.
  }
}
