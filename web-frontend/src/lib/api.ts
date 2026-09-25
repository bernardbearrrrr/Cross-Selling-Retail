import axios from "axios";
import type { BasketResponse, DatasetMode, StatsResponse } from "@/lib/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const client = axios.create({ baseURL: API_URL, timeout: 30000 });

export async function fetchStats(mode: DatasetMode) {
  const { data } = await client.get<StatsResponse>("/api/stats", { params: { mode } });
  return data;
}

export async function recommendBasket(payload: Record<string, unknown>) {
  const { data } = await client.post<BasketResponse>("/api/recommend_basket", payload);
  return data;
}
