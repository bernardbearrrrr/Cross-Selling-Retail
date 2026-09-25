import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Shell } from "@/components/shell";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Cross-Sell Pro",
  description: "Dashboard cross-selling Apriori dan XGBoost",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="id" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full bg-slate-950">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
