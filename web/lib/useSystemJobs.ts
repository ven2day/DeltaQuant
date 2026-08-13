"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api";
import type { SystemJobsStatus } from "./types";

const POLL_INTERVAL_MS = 60_000;

export function useSystemJobs(): {
  jobs: SystemJobsStatus | null;
  loading: boolean;
  error: string | null;
  runJobNow: (name: string) => Promise<boolean>;
} {
  const [jobs, setJobs] = useState<SystemJobsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(() => {
    setLoading(true);
    apiFetch("/api/system/jobs")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<SystemJobsStatus>;
      })
      .then((data) => {
        setJobs(data);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load job status");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  const runJobNow = useCallback(
    async (name: string): Promise<boolean> => {
      const res = await apiFetch(`/api/system/jobs/${name}/run`, { method: "POST" });
      if (!res.ok) return false;
      const data = (await res.json()) as { started: boolean };
      fetchJobs();
      return data.started;
    },
    [fetchJobs],
  );

  return { jobs, loading, error, runJobNow };
}
