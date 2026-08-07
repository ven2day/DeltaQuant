import type { Config } from "tailwindcss";

// Dark-mode values from the validated dataviz reference palette
// (surfaces / ink / status / categorical) — see references/palette.md.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "-apple-system", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        page: "var(--page)",
        surface: "var(--surface)",
        "surface-raised": "var(--surface-raised)",
        border: "var(--border)",
        ink: {
          primary: "var(--ink-primary)",
          secondary: "var(--ink-secondary)",
          muted: "var(--ink-muted)",
        },
        status: {
          good: "var(--status-good)",
          warning: "var(--status-warning)",
          serious: "var(--status-serious)",
          critical: "var(--status-critical)",
        },
        cat: {
          1: "var(--cat-1)",
          2: "var(--cat-2)",
          3: "var(--cat-3)",
          4: "var(--cat-4)",
          5: "var(--cat-5)",
          6: "var(--cat-6)",
          7: "var(--cat-7)",
          8: "var(--cat-8)",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.4), 0 1px 1px rgba(0,0,0,0.3)",
      },
    },
  },
  plugins: [],
};

export default config;
