"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, ScanLine, ShoppingBag, Sparkles } from "lucide-react";
import { cn } from "@/lib/cn";
import { useDataset } from "@/components/dataset-context";
import type { DatasetMode } from "@/lib/types";

const LINKS = [
  { href: "/beranda", label: "Beranda", icon: Sparkles },
  { href: "/komparasi", label: "Komparasi", icon: BarChart3 },
  { href: "/", label: "POS Kasir", icon: ScanLine },
];

export function Sidebar() {
  const pathname = usePathname();
  const { mode, setMode } = useDataset();

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-slate-800 bg-slate-900/50 px-5 py-6 backdrop-blur-xl">
      <div className="mb-8 flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-indigo-500/20 text-indigo-400 ring-1 ring-indigo-500/40">
          <ShoppingBag size={18} />
        </span>
        <div>
          <p className="text-sm font-semibold tracking-tight text-white">Cross-Sell Pro</p>
          <p className="text-xs text-slate-400">Apriori + XGBoost</p>
        </div>
      </div>

      <label className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
        Mode dataset
      </label>
      <select
        value={mode}
        onChange={(event) => setMode(event.target.value as DatasetMode)}
        className="mb-3 w-full rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2.5 text-sm text-slate-200 outline-none ring-indigo-500/40 focus:ring-2"
      >
        <option value="toko">Toko Lokal 2017</option>
        <option value="instacart">Instacart</option>
      </select>
      <span
        className={cn(
          "mb-8 inline-flex w-fit rounded-full px-2.5 py-1 text-xs font-semibold",
          mode === "toko" ? "bg-emerald-500/15 text-emerald-300" : "bg-indigo-500/15 text-indigo-300",
        )}
      >
        {mode === "toko" ? "Toko Lokal 2017" : "Instacart"}
      </span>

      <nav className="flex flex-col gap-2">
        {LINKS.map((link) => {
          const active = pathname === link.href;
          const Icon = link.icon;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "flex items-center gap-3 rounded-full border px-4 py-2.5 text-sm transition",
                active
                  ? "border-indigo-500/50 bg-indigo-500/15 text-white"
                  : "border-transparent text-slate-400 hover:border-slate-800 hover:bg-slate-800/60 hover:text-slate-100",
              )}
            >
              <Icon size={16} />
              {link.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
