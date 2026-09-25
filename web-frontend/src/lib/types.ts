export type DatasetMode = "instacart" | "toko";

export type RecommendationCard = {
  antecedent: string;
  recommended_product: string;
  probability_percent: number;
  decision: string;
  recommend: boolean;
  confidence: number;
  lift: number;
  features: Record<string, number>;
};

export type BasketResponse = {
  mode: string;
  basket: string[];
  current_basket_size: number;
  recommendations: RecommendationCard[];
};

export type ComparisonRow = {
  Model: string;
  Akurasi: number;
  "Precision Macro": number;
  "Recall Macro": number;
  "F1 Macro": number;
  "AUC-ROC": number;
};

export type StatsResponse = {
  mode: DatasetMode;
  label: string;
  model: string;
  model_ready: boolean;
  rules_ready: boolean;
  features: string[];
  rule_count?: number;
  products?: string[];
  comparison?: ComparisonRow[];
  profiles?: Record<string, Record<string, number>>;
};
