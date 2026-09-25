"use client";

import { useEffect, useState } from "react";
import { fetchStats } from "@/lib/api";
import { useDataset } from "@/components/dataset-context";
import type { StatsResponse } from "@/lib/types";

export default function KomparasiPage() {
  const { mode, label } = useDataset();
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    fetchStats(mode)
      .then(setStats)
      .catch(() => {
        setStats(null);
        setError("Tabel komparasi belum dapat dimuat dari API.");
      });
  }, [mode]);

  const rows = stats?.comparison ?? [];
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-300">Evaluasi</p>
      <h1 className="mt-2 text-3xl font-semibold text-white">Komparasi model {label}</h1>
      {error ? <p className="mt-4 text-sm text-amber-300">{error}</p> : null}
      <div className="mt-8 overflow-hidden rounded-2xl border border-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900 text-slate-400">
            <tr>
              {["Model", "Akurasi", "Precision Macro", "Recall Macro", "F1 Macro", "AUC-ROC"].map((head) => (
                <th key={head} className="px-4 py-3 font-medium">{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.Model} className="border-t border-slate-800">
                <td className="px-4 py-3 text-white">{row.Model}</td>
                <td className="px-4 py-3">{(row.Akurasi * 100).toFixed(2)}%</td>
                <td className="px-4 py-3">{row["Precision Macro"].toFixed(3)}</td>
                <td className="px-4 py-3">{row["Recall Macro"].toFixed(3)}</td>
                <td className="px-4 py-3">{row["F1 Macro"].toFixed(3)}</td>
                <td className="px-4 py-3">{row["AUC-ROC"].toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
