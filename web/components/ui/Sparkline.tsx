// Minimal single-series trend line for a stat tile. Per dataviz mark specs:
// 2px line with round join/cap, a soft ~10% opacity area fill under it, no
// axes/gridlines (a sparkline reads as a shape, not a chart to interrogate).
export function Sparkline({
  values,
  width = 96,
  height = 28,
  color = "var(--cat-1)",
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  if (values.length < 2) {
    return <svg width={width} height={height} aria-hidden />;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);

  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return [x, y] as const;
  });

  const linePath = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} role="img" aria-label="Trend">
      <path d={areaPath} fill={color} opacity={0.1} stroke="none" />
      <path
        d={linePath}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
