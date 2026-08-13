"use client";

import { useEffect, useState } from "react";
import { checkAuthenticated } from "./api";

/**
 * Redirects to /login when there is no valid dashboard session. Used by the
 * protected dashboard page only — /login itself must never use this (it would
 * redirect-loop).
 */
export function useAuthGuard(): { checking: boolean } {
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;

    checkAuthenticated().then((authenticated) => {
      if (cancelled) return;
      if (!authenticated) {
        // This is an authentication boundary, not an in-app navigation.  A
        // full replacement is intentionally more robust than a soft router
        // transition here: if Next's client router is stale after a rebuild or
        // deployment, the old transition can fail and leave this page showing
        // "Checking session…" forever.
        window.location.replace("/login");
        return;
      }
      setChecking(false);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  return { checking };
}
