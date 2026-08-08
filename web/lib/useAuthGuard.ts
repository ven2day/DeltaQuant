"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { checkAuthenticated } from "./api";

/**
 * Redirects to /login when there is no valid dashboard session. Used by the
 * protected dashboard page only — /login itself must never use this (it would
 * redirect-loop).
 */
export function useAuthGuard(): { checking: boolean } {
  const router = useRouter();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;

    checkAuthenticated().then((authenticated) => {
      if (cancelled) return;
      if (!authenticated) {
        router.replace("/login");
        return;
      }
      setChecking(false);
    });

    return () => {
      cancelled = true;
    };
  }, [router]);

  return { checking };
}
