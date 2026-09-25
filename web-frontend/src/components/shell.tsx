"use client";

import { DatasetProvider } from "@/components/dataset-context";
import { Sidebar } from "@/components/sidebar";

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <DatasetProvider>
      <div className="flex min-h-screen bg-slate-950 text-slate-200">
        <Sidebar />
        <main className="min-w-0 flex-1 px-8 py-8">{children}</main>
      </div>
    </DatasetProvider>
  );
}
