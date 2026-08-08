"use client";

import { AlertCircle, Lock, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import { apiFetch } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);

    try {
      const response = await apiFetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        router.replace("/");
        return;
      }

      if (response.status === 429) {
        setError("Too many failed attempts. Try again later.");
      } else {
        setError("Invalid username or password.");
      }
    } catch {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-surface shadow-card">
        <div className="flex flex-col items-center gap-2 border-b border-border px-6 py-6">
          <ShieldCheck size={28} className="text-cat-1" strokeWidth={2} />
          <div className="text-base font-semibold text-ink-primary">₹DeltaQuant</div>
          <div className="text-xs text-ink-muted">Sign in to view the live dashboard</div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4 px-6 py-6">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="username" className="text-xs font-medium text-ink-secondary">
              Username
            </label>
            <input
              id="username"
              name="username"
              type="text"
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="rounded-lg border border-border bg-page px-3 py-2 text-sm text-ink-primary outline-none focus:border-cat-1"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="password" className="text-xs font-medium text-ink-secondary">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-lg border border-border bg-page px-3 py-2 text-sm text-ink-primary outline-none focus:border-cat-1"
            />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-status-critical/40 bg-status-critical/10 px-3 py-2 text-xs text-status-critical">
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="mt-1 flex items-center justify-center gap-2 rounded-lg bg-cat-1 px-3 py-2 text-sm font-semibold text-white transition-opacity disabled:opacity-60"
          >
            <Lock size={14} strokeWidth={2} />
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </main>
  );
}
