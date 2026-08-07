import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const jetBrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "₹DeltaQuant — Live Dashboard",
  description: "Live paper-trading dashboard for ₹DeltaQuant",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetBrainsMono.variable}`} suppressHydrationWarning>
      <body
        className="min-h-screen bg-page font-sans text-ink-primary antialiased"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
