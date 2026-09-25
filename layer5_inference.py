"""
Layer 5 — Integration Layer

Simulasi inferensi cross-selling. Skrip memuat model XGBoost hasil Layer 4,
lalu menilai satu vektor fitur pengguna terhadap satu aturan asosiasi.

Jalankan dari folder proyek:
    python layer5_inference.py
"""

from pathlib import Path

import joblib
import pandas as pd

# Urutan kolom wajib sama dengan saat XGBClassifier.fit di Layer 4.
FEATURE_COLUMNS = [
    "total_prior_orders",
    "avg_basket_size",
    "avg_days_between",
    "antecedent_rate",
    "rule_confidence",
    "rule_lift",
    "interest_confidence",
    "interest_lift",
]

# Ambang bawaan XGBoost untuk kelas positif.
DECISION_THRESHOLD = 0.5

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_cross_sell_model.pkl"

# Baris pertama data uji Layer 4.
# Split: test_size=0.2, stratify=y, random_state=42, sebelum SMOTE.
# Indeks sumber pada matriks fitur: 197592.
# Aturan yang melekat pada baris ini: Organic Raspberries -> Bag of Organic Bananas.
# Label aktual y = 0, artinya consequent tidak dibeli pada pesanan train.
TEST_SAMPLE = {
    "total_prior_orders": 18.0,
    "avg_basket_size": 11.11111068725586,
    "avg_days_between": 5.411764621734619,
    "antecedent_rate": 0.0555555559694767,
    "rule_confidence": 0.3154238164424896,
    "rule_lift": 2.6081016063690186,
    "interest_confidence": 0.017523545771837234,
    "interest_lift": 0.144894540309906,
}
TEST_ACTUAL_LABEL = 0
TEST_RULE = "Organic Raspberries -> Bag of Organic Bananas"


def load_model(model_path=MODEL_PATH):
    """Memuat XGBClassifier yang disimpan Layer 4."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}. "
            "Jalankan layer4_model_training.ipynb terlebih dahulu."
        )
    return joblib.load(path)


def predict_cross_sell(
    total_prior_orders,
    avg_basket_size,
    avg_days_between,
    antecedent_rate,
    rule_confidence,
    rule_lift,
    interest_confidence,
    interest_lift,
    model=None,
    threshold=DECISION_THRESHOLD,
):
    """
    Menilai apakah suatu aturan cross-selling layak direkomendasikan.

    Parameter fitur mengikuti matriks Layer 3. `model` boleh dikosongkan;
    bila kosong, berkas pkl di folder skrip ini yang dimuat.
    Keputusan "Ya" diberikan bila probabilitas kelas 1 mencapai ambang.
    """
    classifier = load_model() if model is None else model
    features = pd.DataFrame(
        [
            {
                "total_prior_orders": total_prior_orders,
                "avg_basket_size": avg_basket_size,
                "avg_days_between": avg_days_between,
                "antecedent_rate": antecedent_rate,
                "rule_confidence": rule_confidence,
                "rule_lift": rule_lift,
                "interest_confidence": interest_confidence,
                "interest_lift": interest_lift,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    probability = float(classifier.predict_proba(features)[0, 1])
    recommend = probability >= threshold
    return {
        "probability": probability,
        "recommend": recommend,
        "decision": "Ya" if recommend else "Tidak",
    }


def main():
    model = load_model()
    result = predict_cross_sell(model=model, **TEST_SAMPLE)

    print("=" * 62)
    print("Simulasi Inferensi Cross-Selling")
    print("=" * 62)
    print(f"Model            : models/{MODEL_PATH.name}")
    print(f"Aturan           : {TEST_RULE}")
    print(f"Label aktual uji : {TEST_ACTUAL_LABEL} (consequent tidak dibeli)")
    print()
    print("Fitur masukan")
    for name in FEATURE_COLUMNS:
        print(f"  {name:<22}: {TEST_SAMPLE[name]:.6f}")
    print()
    print("Hasil prediksi")
    print(f"  Probabilitas kelas 1 : {result['probability']:.6f}")
    print(f"  Ambang keputusan      : {DECISION_THRESHOLD:.2f}")
    print(f"  Rekomendasi           : {result['decision']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
