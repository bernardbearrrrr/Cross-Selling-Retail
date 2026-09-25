"""
API inferensi cross-selling.

Jalankan dari folder proyek:
    source venv/bin/activate
    uvicorn api:app --reload --port 8000
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.50

ACADEMIC_FEATURES = [
    "total_prior_orders",
    "avg_basket_size",
    "avg_days_between",
    "antecedent_rate",
    "rule_confidence",
    "rule_lift",
    "interest_confidence",
    "interest_lift",
]
REAL_FEATURES = [
    "current_basket_size",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "rule_confidence",
    "rule_lift",
    "antecedent_rate",
    "interest_lift",
]
PROFILES = {
    "Member Loyalis": {"total_prior_orders": 25.0, "avg_basket_size": 10.5, "avg_days_between": 5.0},
    "Member Kasual": {"total_prior_orders": 8.0, "avg_basket_size": 5.0, "avg_days_between": 12.0},
    "Pelanggan Baru": {"total_prior_orders": 2.0, "avg_basket_size": 3.0, "avg_days_between": 20.0},
}
MODES = {
    "instacart": {
        "label": "Dataset Akademis (Instacart)",
        "model": ROOT / "models" / "xgboost_cross_sell_model.pkl",
        "comparison": ROOT / "outputs" / "layer4_model_comparison.csv",
        "rules": ROOT / "outputs" / "apriori_rules.csv",
        "features": ACADEMIC_FEATURES,
        "product_column": "antecedents",
        "pair_column": "consequents",
    },
    "toko": {
        "label": "Dataset Riil (Toko Lokal 2017)",
        "model": ROOT / "models" / "xgboost_cross_sell_ril.pkl",
        "comparison": ROOT / "outputs_ril" / "layer4_model_comparison.csv",
        "rules": ROOT / "outputs_ril" / "apriori_rules_ril.csv",
        "features": REAL_FEATURES,
        "product_column": "antecedent_name",
        "pair_column": "consequent_name",
    },
}
MODE_ALIASES = {
    "instacart": "instacart",
    "akademis": "instacart",
    "akademik": "instacart",
    "dataset akademis (instacart)": "instacart",
    "toko": "toko",
    "riil": "toko",
    "lokal": "toko",
    "dataset riil (toko lokal 2017)": "toko",
}


class RecommendRequest(BaseModel):
    mode: str = Field(examples=["instacart", "toko"])
    product: str | None = None
    profile: str | None = None
    current_basket_size: float | None = None
    hour_of_day: float | None = None
    day_of_week: float | None = None
    features: dict[str, float] | None = None


class BasketRequest(BaseModel):
    mode: str = Field(examples=["instacart", "toko"])
    basket: list[str]
    profile: str | None = None
    hour_of_day: float | None = None
    day_of_week: float | None = None


app = FastAPI(title="Cross-Selling API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_mode(mode: str) -> dict:
    key = MODE_ALIASES.get(mode.strip().lower())
    if key is None:
        raise HTTPException(status_code=400, detail="Mode harus instacart atau toko.")
    return {"key": key, **MODES[key]}


@lru_cache(maxsize=4)
def load_model(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_table(path: Path) -> pd.DataFrame:
    try:
        return load_table(str(path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Berkas tidak ditemukan: {path.relative_to(ROOT)}")


def best_rule(rules: pd.DataFrame, column: str, product: str) -> pd.Series:
    matches = rules.loc[rules[column].astype(str).eq(product)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Produk tidak punya aturan: {product}")
    return matches.sort_values(["lift", "confidence"], ascending=False).iloc[0]


def assemble_features(spec: dict, rule: pd.Series, body: RecommendRequest) -> dict:
    antecedent_rate = float(rule["antecedent support"])
    rule_confidence = float(rule["confidence"])
    rule_lift = float(rule["lift"])
    if spec["key"] == "instacart":
        profile_name = body.profile or "Member Loyalis"
        if profile_name not in PROFILES:
            raise HTTPException(status_code=400, detail="Profil tidak dikenal.")
        profile = PROFILES[profile_name]
        return {
            "total_prior_orders": profile["total_prior_orders"],
            "avg_basket_size": profile["avg_basket_size"],
            "avg_days_between": profile["avg_days_between"],
            "antecedent_rate": antecedent_rate,
            "rule_confidence": rule_confidence,
            "rule_lift": rule_lift,
            "interest_confidence": antecedent_rate * rule_confidence,
            "interest_lift": antecedent_rate * rule_lift,
        }
    if body.current_basket_size is None or body.hour_of_day is None or body.day_of_week is None:
        raise HTTPException(
            status_code=400,
            detail="Mode toko membutuhkan current_basket_size, hour_of_day, dan day_of_week.",
        )
    day = float(body.day_of_week)
    return {
        "current_basket_size": float(body.current_basket_size),
        "hour_of_day": float(body.hour_of_day),
        "day_of_week": day,
        "is_weekend": float(day >= 5),
        "rule_confidence": rule_confidence,
        "rule_lift": rule_lift,
        "antecedent_rate": antecedent_rate,
        "interest_lift": antecedent_rate * rule_lift,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats(mode: str = "instacart"):
    spec = resolve_mode(mode)
    payload = {
        "mode": spec["key"],
        "label": spec["label"],
        "model": str(spec["model"].relative_to(ROOT)),
        "model_ready": spec["model"].exists(),
        "rules_ready": spec["rules"].exists(),
        "features": spec["features"],
    }
    if spec["rules"].exists():
        rules = require_table(spec["rules"])
        payload["rule_count"] = int(len(rules))
        payload["products"] = sorted(rules[spec["product_column"]].dropna().astype(str).unique().tolist())
    if spec["comparison"].exists():
        comparison = require_table(spec["comparison"])
        payload["comparison"] = comparison.to_dict(orient="records")
    if spec["key"] == "instacart":
        payload["profiles"] = PROFILES
    return payload


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    spec = resolve_mode(body.mode)
    if body.features:
        missing = [name for name in spec["features"] if name not in body.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Fitur kurang: {missing}")
        features = {name: float(body.features[name]) for name in spec["features"]}
        scanned = body.product
        consequent = None
        confidence = features.get("rule_confidence")
        lift = features.get("rule_lift")
    else:
        if not body.product:
            raise HTTPException(status_code=400, detail="Isi product atau features.")
        rules = require_table(spec["rules"])
        rule = best_rule(rules, spec["product_column"], body.product)
        features = assemble_features(spec, rule, body)
        scanned = body.product
        consequent = str(rule[spec["pair_column"]])
        confidence = float(rule["confidence"])
        lift = float(rule["lift"])

    if not spec["model"].exists():
        raise HTTPException(status_code=404, detail=f"Model tidak ditemukan: {spec['model'].name}")
    try:
        model = load_model(str(spec["model"]))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model tidak ditemukan: {spec['model'].name}")
    frame = pd.DataFrame([features], columns=spec["features"])
    probability = float(model.predict_proba(frame)[0, 1])
    recommend_item = probability >= THRESHOLD
    return {
        "mode": spec["key"],
        "scanned_product": scanned,
        "recommended_product": consequent,
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "decision": "Rekomendasikan ke Pelanggan" if recommend_item else "Tidak Perlu Ditawarkan",
        "recommend": recommend_item,
        "confidence": confidence,
        "lift": lift,
        "features": features,
    }


def rule_features(spec: dict, rule: pd.Series, basket_size: int, body: BasketRequest) -> dict:
    antecedent_rate = float(rule["antecedent support"])
    rule_confidence = float(rule["confidence"])
    rule_lift = float(rule["lift"])
    if spec["key"] == "instacart":
        profile_name = body.profile or "Member Loyalis"
        if profile_name not in PROFILES:
            raise HTTPException(status_code=400, detail="Profil tidak dikenal.")
        profile = PROFILES[profile_name]
        return {
            "total_prior_orders": profile["total_prior_orders"],
            "avg_basket_size": profile["avg_basket_size"],
            "avg_days_between": profile["avg_days_between"],
            "antecedent_rate": antecedent_rate,
            "rule_confidence": rule_confidence,
            "rule_lift": rule_lift,
            "interest_confidence": antecedent_rate * rule_confidence,
            "interest_lift": antecedent_rate * rule_lift,
        }
    if body.hour_of_day is None or body.day_of_week is None:
        raise HTTPException(status_code=400, detail="Mode toko membutuhkan hour_of_day dan day_of_week.")
    day = float(body.day_of_week)
    return {
        "current_basket_size": float(basket_size),
        "hour_of_day": float(body.hour_of_day),
        "day_of_week": day,
        "is_weekend": float(day >= 5),
        "rule_confidence": rule_confidence,
        "rule_lift": rule_lift,
        "antecedent_rate": antecedent_rate,
        "interest_lift": antecedent_rate * rule_lift,
    }


@app.post("/api/recommend_basket")
def recommend_basket(body: BasketRequest):
    spec = resolve_mode(body.mode)
    basket = list(dict.fromkeys(item.strip() for item in body.basket if item and item.strip()))
    if not basket:
        raise HTTPException(status_code=400, detail="Keranjang tidak boleh kosong.")
    basket_size = len(basket)
    present = set(basket)
    rules = require_table(spec["rules"])
    product_column = spec["product_column"]
    pair_column = spec["pair_column"]
    candidates = rules.loc[
        rules[product_column].astype(str).isin(present)
        & ~rules[pair_column].astype(str).isin(present)
    ]
    if candidates.empty or not spec["model"].exists():
        if not spec["model"].exists():
            raise HTTPException(status_code=404, detail=f"Model tidak ditemukan: {spec['model'].name}")
        return {
            "mode": spec["key"],
            "basket": basket,
            "current_basket_size": basket_size,
            "recommendations": [],
        }

    feature_rows = [rule_features(spec, row, basket_size, body) for _, row in candidates.iterrows()]
    frame = pd.DataFrame(feature_rows, columns=spec["features"])
    model = load_model(str(spec["model"]))
    probabilities = model.predict_proba(frame)[:, 1]
    scored = sorted(
        zip(probabilities, feature_rows, candidates.itertuples(index=False)),
        key=lambda item: item[0],
        reverse=True,
    )
    seen = set()
    recommendations = []
    columns = list(candidates.columns)
    for probability, features, record in scored:
        values = dict(zip(columns, record))
        consequent = str(values[pair_column])
        if consequent in seen:
            continue
        seen.add(consequent)
        offered = float(probability) >= THRESHOLD
        recommendations.append({
            "antecedent": str(values[product_column]),
            "recommended_product": consequent,
            "probability": float(probability),
            "probability_percent": round(float(probability) * 100, 2),
            "threshold": THRESHOLD,
            "recommend": offered,
            "decision": "Rekomendasikan ke Pelanggan" if offered else "Tidak Perlu Ditawarkan",
            "confidence": float(values["confidence"]),
            "lift": float(values["lift"]),
            "features": features,
        })
        if len(recommendations) == 3:
            break
    return {
        "mode": spec["key"],
        "basket": basket,
        "current_basket_size": basket_size,
        "recommendations": recommendations,
    }
