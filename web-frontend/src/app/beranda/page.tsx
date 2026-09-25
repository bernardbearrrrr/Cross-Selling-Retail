"use client";

import { useEffect, useState } from "react";
import { fetchStats } from "@/lib/api";
import { useDataset } from "@/components/dataset-context";
import type { StatsResponse } from "@/lib/types";

const COPY = {
  toko: [
    ["Layer 1", "Jurnal kasir Januari–Maret 2017 disaring menjadi transaksi JUL."],
    ["Layer 2", "Apriori memakai struk Januari–Februari dengan min_support 0,001."],
    ["Layer 3", "Delapan fitur konteks struk menggantikan riwayat pelanggan."],
    ["Layer 4", "XGBoost diuji pada struk Maret dan disimpan terpisah dari model Instacart."],
    ["Layer 5", "Kasir memindai antecedent, lalu model menilai apakah consequent ditawarkan."],
  ],
  instacart: [
    ["Layer 1", "Enam CSV Instacart diperiksa sebelum penambangan aturan."],
    ["Layer 2", "50.000 pesanan prior pertama menghasilkan aturan Apriori."],
    ["Layer 3", "Fitur perilaku pelanggan diseimbangkan dengan SMOTE."],
    ["Layer 4", "Lima algoritma dibandingkan. XGBoost menjadi model integrasi."],
    ["Layer 5", "Inferensi memakai profil member dan aturan ber-lift tertinggi."],
  ],
} as const;

export default function BerandaPage() {
  const { mode, label } = useDataset();
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    fetchStats(mode).then(setStats).catch(() => setStats(null));
  }, [mode]);

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-300">Arsitektur</p>
      <h1 className="mt-2 text-3xl font-semibold text-white">Lima lapisan {label}</h1>
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {COPY[mode].map(([title, body]) => (
          <article key={title} className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
            <h2 className="text-sm font-semibold text-indigo-200">{title}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">{body}</p>
          </article>
        ))}
      </div>
      <p className="mt-6 text-sm text-slate-400">
        {stats
          ? `${stats.rule_count ?? 0} aturan siap. Model ${stats.model_ready ? "terhubung" : "belum ada"} di ${stats.model}.`
          : "Statistik API belum termuat. Jalankan backend di port 8000."}
      </p>
    </div>
  );
}
