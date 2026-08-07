// Meter: the fill carries severity/value; the unfilled track is a lighter
// step of the same ramp so state reads across the whole bar at a glance.
export function Meter({
  fraction,
  color = "var(--cat-1)",
}: {
  fraction: number;
  color?: string;
}) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full"
      style={{ backgroundColor: "rgba(255,255,255,0.08)" }}
    >
      <div
        className="h-full rounded-full transition-[width]"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  );
}
