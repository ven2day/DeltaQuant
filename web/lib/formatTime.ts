// Shared timestamp formatting: always the VIEWER'S own local timezone (undefined
// locale/timeZone lets the browser pick both), with an explicit offset label so it's
// never ambiguous whether a given panel is showing local time or something else.
// Signal History is the one deliberate exception — it pins Asia/Kolkata and labels it
// "IST" explicitly, since NSE trading-hours context only makes sense read in IST.

export function formatLocalDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

export function formatLocalTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  });
}
