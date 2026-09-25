"use client";

import { useEffect, useState } from "react";
import { Loader2, X } from "lucide-react";
import { fetchStats, recommendBasket } from "@/lib/api";
import { useDataset } from "@/components/dataset-context";
import type { BasketResponse, RecommendationCard } from "@/lib/types";

const DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"];
const PROFILES = ["Member Loyalis", "Member Kasual", "Pelanggan Baru"];

export function PosCashier() {
  const { mode, label } = useDataset();
  const [catalog, setCatalog] = useState<string[]>([]);
  const [basket, setBasket] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [profile, setProfile] = useState(PROFILES[0]);
  const [hour, setHour] = useState(11);
  const [day, setDay] = useState(2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<BasketResponse | null>(null);

  useEffect(() => {
    let active = true;
    setBasket([]);
    setResult(null);
    setError("");
    fetchStats(mode)
      .then((stats) => {
        if (!active) return;
        const list = stats.products ?? [];
        setCatalog(list);
        setDraft(list[0] ?? "");
      })
      .catch(() => {
        if (!active) return;
        setCatalog([]);
        setDraft("");
        setError("API belum terjangkau di http://127.0.0.1:8000. Jalankan uvicorn api:app --port 8000.");
      });
    return () => {
      active = false;
    };
  }, [mode]);

  function addProduct() {
    if (!draft || basket.includes(draft)) return;
    setBasket((current) => [...current, draft]);
  }

  async function onScan() {
    setLoading(true);
    setError("");
    try {
      const payload =
        mode === "toko"
          ? { mode, basket, hour_of_day: hour, day_of_week: day }
          : { mode, basket, profile };
      setResult(await recommendBasket(payload));
    } catch {
      setResult(null);
      setError("Prediksi gagal. Periksa isi struk dan pastikan API masih berjalan.");
    } finally {
      setLoading(false);
    }
  }

  const available = catalog.filter((item) => !basket.includes(item));

  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-300">Simulasi kasir</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">Struk banyak barang</h1>
      <p className="mt-2 max-w-2xl text-sm text-slate-400">
        Mode aktif: {label}. Ukuran keranjang dihitung dari jumlah produk yang dipindai.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-[0_16px_40px_rgba(0,0,0,0.28)]">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-slate-300">Struk aktif</h2>
            <p className="text-xs text-slate-400">{basket.length} barang</p>
          </div>
          <div className="mt-5 space-y-4">
            <Field label="Tambah produk">
              <div className="flex gap-2">
                <select
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none ring-indigo-500/30 focus:ring-2"
                >
                  {available.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={addProduct}
                  disabled={!draft}
                  className="rounded-xl border border-slate-700 px-3 text-sm text-slate-200 hover:bg-slate-800"
                >
                  Tambah
                </button>
              </div>
            </Field>
            <div className="flex min-h-16 flex-wrap gap-2">
              {basket.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setBasket((current) => current.filter((name) => name !== item))}
                  className="inline-flex items-center gap-1 rounded-full border border-indigo-500/40 bg-indigo-500/10 px-3 py-1 text-xs text-indigo-100"
                >
                  {item}
                  <X size={12} />
                </button>
              ))}
            </div>
            {mode === "toko" ? (
              <>
                <Field label="Jam transaksi">
                  <input className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none ring-indigo-500/30 focus:ring-2" type="number" min={0} max={23} value={hour} onChange={(event) => setHour(Number(event.target.value))} />
                </Field>
                <Field label="Hari belanja">
                  <select className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none ring-indigo-500/30 focus:ring-2" value={day} onChange={(event) => setDay(Number(event.target.value))}>
                    {DAYS.map((name, index) => (
                      <option key={name} value={index}>{name}</option>
                    ))}
                  </select>
                </Field>
              </>
            ) : (
              <Field label="Profil pelanggan">
                <select className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none ring-indigo-500/30 focus:ring-2" value={profile} onChange={(event) => setProfile(event.target.value)}>
                  {PROFILES.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </Field>
            )}
            <button
              type="button"
              onClick={onScan}
              disabled={basket.length === 0 || loading}
              className="mt-2 w-full rounded-xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Memindai..." : "Pindai Produk"}
            </button>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-[0_16px_40px_rgba(0,0,0,0.28)]">
          <h2 className="text-sm font-medium text-slate-300">Tiga rekomendasi teratas</h2>
          {error ? <p className="mt-4 text-sm text-amber-300">{error}</p> : null}
          {loading ? (
            <div className="mt-8 flex items-center gap-3 text-slate-400">
              <Loader2 className="animate-spin text-indigo-400" size={18} />
              Model sedang menilai seluruh struk...
            </div>
          ) : null}
          {!loading && result ? <BasketResults result={result} /> : null}
          {!loading && !result && !error ? (
            <p className="mt-8 text-sm text-slate-500">Tambahkan satu atau lebih produk, lalu pindai struk.</p>
          ) : null}
        </section>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs text-slate-400">{label}</span>
      {children}
    </label>
  );
}

function BasketResults({ result }: { result: BasketResponse }) {
  if (result.recommendations.length === 0) {
    return <p className="mt-6 text-sm text-slate-400">Tidak ada pasangan di luar struk ini.</p>;
  }
  return (
    <div className="mt-5 space-y-4">
      {result.recommendations.map((item, index) => (
        <RecommendationCard key={item.recommended_product} rank={index + 1} item={item} />
      ))}
    </div>
  );
}

function RecommendationCard({ rank, item }: { rank: number; item: RecommendationCard }) {
  const percent = item.probability_percent;
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
      <p className="text-xs text-slate-500">Rekomendasi {rank} · dari {item.antecedent}</p>
      <p className="mt-1 font-medium text-white">{item.recommended_product}</p>
      <p className={`mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${item.recommend ? "bg-emerald-500/15 text-emerald-200" : "bg-amber-500/15 text-amber-200"}`}>
        {item.decision}
      </p>
      <div className="mt-3">
        <div className="mb-1 flex justify-between text-xs text-slate-400">
          <span>Probabilitas</span>
          <span className="font-semibold text-slate-200">{percent.toFixed(2)}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full bg-indigo-400 transition-all duration-700" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
        </div>
      </div>
      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-slate-400">Lihat Detail Fitur Backend</summary>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {Object.entries(item.features).map(([name, value]) => (
            <div key={name} className="rounded-lg border border-slate-800 px-2 py-1.5 text-center">
              <p className="text-[10px] text-slate-500">{name}</p>
              <p className="text-xs text-white">{value.toFixed(4)}</p>
            </div>
          ))}
        </div>
      </details>
    </article>
  );
}
