"use client";

import { createContext, useContext, useMemo, useState } from "react";
import type { DatasetMode } from "@/lib/types";

type DatasetContextValue = {
  mode: DatasetMode;
  setMode: (mode: DatasetMode) => void;
  label: string;
};

const DatasetContext = createContext<DatasetContextValue | null>(null);

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<DatasetMode>("toko");
  const value = useMemo(
    () => ({
      mode,
      setMode,
      label: mode === "toko" ? "Toko Lokal 2017" : "Instacart",
    }),
    [mode],
  );
  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset() {
  const value = useContext(DatasetContext);
  if (!value) throw new Error("useDataset harus dipakai di dalam DatasetProvider.");
  return value;
}
