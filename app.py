"""
Dashboard presentasi Tugas Akhir.

Jalankan dari folder proyek:
    source venv/bin/activate
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.50

MODE_ACADEMIC = "Dataset Akademis (Instacart)"
MODE_REAL = "Dataset Riil (Toko Lokal 2017)"

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
WEEKDAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

PROFILES = {
    "Member Loyalis": {
        "total_prior_orders": 25,
        "avg_basket_size": 10.5,
        "avg_days_between": 5.0,
    },
    "Member Kasual": {
        "total_prior_orders": 8,
        "avg_basket_size": 5.0,
        "avg_days_between": 12.0,
    },
    "Pelanggan Baru": {
        "total_prior_orders": 2,
        "avg_basket_size": 3.0,
        "avg_days_between": 20.0,
    },
}

MODES = {
    MODE_ACADEMIC: {
        "model": ROOT / "models" / "xgboost_cross_sell_model.pkl",
        "comparison": ROOT / "outputs" / "layer4_model_comparison.csv",
        "rules": ROOT / "outputs" / "apriori_rules.csv",
        "charts": [
            ROOT / "outputs" / "layer4_model_comparison.png",
            ROOT / "outputs" / "layer4_xgboost_confusion_matrix.png",
        ],
        "features": ACADEMIC_FEATURES,
        "product_column": "antecedents",
        "pair_column": "consequents",
        "flow": (
            "Alur berkas: dataset CSV → outputs/apriori_rules.csv → "
            "layer3_smoted_features.csv → models/xgboost_cross_sell_model.pkl."
        ),
        "layers": [
            ("Layer 1", "Data Input dan Schema Validation",
             "Enam berkas CSV Instacart dimuat dan diperiksa: jumlah baris, tipe data, nilai kosong, kunci, serta relasi aisle, department, order, dan product."),
            ("Layer 2", "Pre-processing dan Apriori Mining",
             "Lima puluh ribu pesanan prior pertama menjadi matriks keranjang. Apriori memakai min_support 0,01. Aturan tersimpan di outputs/apriori_rules.csv."),
            ("Layer 3", "Feature Engineering dan SMOTE",
             "Delapan fitur perilaku pelanggan digabung dengan confidence dan lift. Setelah SMOTE, matriks berimbang berukuran 519.142 baris."),
            ("Layer 4", "Model Training dan Benchmarking",
             "Lima algoritma dilatih pada data yang sama. XGBoost disimpan sebagai models/xgboost_cross_sell_model.pkl."),
            ("Layer 5", "Integration dan Real-time Inference",
             "Model menilai delapan fitur pelanggan dan memutuskan rekomendasi pada ambang 50%."),
        ],
    },
    MODE_REAL: {
        "model": ROOT / "models" / "xgboost_cross_sell_ril.pkl",
        "comparison": ROOT / "outputs_ril" / "layer4_model_comparison.csv",
        "rules": ROOT / "outputs_ril" / "apriori_rules_ril.csv",
        "charts": [
            ROOT / "outputs_ril" / "layer4_xgboost_confusion_matrix.png",
        ],
        "features": REAL_FEATURES,
        "product_column": "antecedent_name",
        "pair_column": "consequent_name",
        "flow": (
            "Alur berkas: dataset_ril Excel → outputs_ril/jul_transactions.csv → "
            "apriori_rules_ril.csv → layer3_test_features.csv → models/xgboost_cross_sell_ril.pkl."
        ),
        "layers": [
            ("Layer 1", "Prapemrosesan Jurnal Kasir",
             "Tiga berkas Excel 2017 dibaca. Hanya transaksi TP_TRN = JUL yang disimpan. ITEM menjadi identitas produk, dengan nama tampilan dari ejaan yang paling sering."),
            ("Layer 2", "Apriori Januari–Februari",
             "Aturan ditambang dari 68.269 struk jual Januari dan Februari. min_support = 0,001 karena ambang 0,005 tidak menghasilkan pasangan. Hasilnya 138 aturan di outputs_ril/apriori_rules_ril.csv."),
            ("Layer 3", "Fitur Konteks Struk",
             "Tidak ada ID pelanggan. Satu baris adalah struk yang sudah memuat antecedent. Delapan fitur mencakup ukuran keranjang, jam, hari, akhir pekan, serta skor aturan."),
            ("Layer 4", "Uji pada Maret",
             "SMOTE hanya menyeimbangkan data latih Januari–Februari. Maret tetap murni. XGBoost disimpan sebagai models/xgboost_cross_sell_ril.pkl."),
            ("Layer 5", "Inferensi Struk Aktif",
             "Model memperkirakan apakah consequent layak ditawarkan ke keranjang yang sedang dipindai, pada ambang 50%."),
        ],
    },
}


st.set_page_config(
    page_title="Sistem Cerdas Cross-Selling",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #0E1117;
        color: #E6EDF3;
    }
    [data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        background: #161B22;
        border-right: 1px solid #2E384D;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
    h1, h2, h3, p, label, span { color: #E6EDF3; }
    h1 { font-weight: 700; letter-spacing: -0.03em; }
    [data-testid="stMetric"], [data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1F2633;
        border: 1px solid #2E384D;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }
    [data-testid="stMetric"] { padding: 0.85rem 0.7rem; text-align: center; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        justify-content: center;
        width: 100%;
    }
    [data-testid="stExpander"] details { background: #1F2633; border-radius: 12px; }
    .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div,
    [data-baseweb="input"] {
        background: #0E1117 !important;
        color: #E6EDF3 !important;
        border: 1px solid #2E384D !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #1F2633;
        border: 1px solid #2E384D;
        border-radius: 999px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.35rem;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #1F2633;
        border: 1px solid #2E384D;
        border-radius: 12px;
    }
    div[data-testid="stAlert"] {
        background: #1F2633;
        border: 1px solid #2E384D;
        border-radius: 12px;
        color: #E6EDF3;
    }
    .pos-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .pair-card {
        background: #121820;
        border: 1px solid #2E384D;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 92px;
    }
    .pair-card span { color: #93A4BD; font-size: 0.78rem; }
    .pair-card strong { display: block; margin-top: 0.35rem; font-size: 1.02rem; }
    .decision-ok, .decision-hold {
        border-radius: 12px;
        padding: 0.9rem 1rem;
        font-weight: 650;
        margin: 0.4rem 0 0.8rem 0;
    }
    .decision-ok { background: rgba(16, 185, 129, 0.16); border: 1px solid #10B981; color: #A7F3D0; }
    .decision-hold { background: rgba(245, 158, 11, 0.16); border: 1px solid #F59E0B; color: #FDE68A; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path_str):
    return joblib.load(path_str)


@st.cache_data
def load_table(path_str):
    return pd.read_csv(path_str)


def render_header(mode_name):
    st.title("Sistem Cerdas Cross-Selling pada Ritel Modern")
    st.caption(
        "Integrasi Association Rule Mining (Apriori) dan "
        "Extreme Gradient Boosting (XGBoost)"
    )
    st.caption(f"Mode aktif: {mode_name}")


def show_decision(probability, positive, negative):
    percent = probability * 100
    css = "decision-ok" if probability >= THRESHOLD else "decision-hold"
    label = positive if probability >= THRESHOLD else negative
    st.markdown(
        f'<div class="{css}">{label}<br><span style="font-size:1.35rem;font-weight:700;">{percent:.2f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Ambang keputusan {THRESHOLD:.0%}")


def predict_frame(mode, features):
    model_path = mode["model"]
    if not model_path.exists():
        st.warning(
            f"Model {model_path.relative_to(ROOT)} belum tersedia. "
            "Selesaikan pelatihan Layer 4 untuk mode ini."
        )
        return None
    frame = pd.DataFrame([features], columns=mode["features"])
    model = load_model(str(model_path))
    return float(model.predict_proba(frame)[0, 1])


def page_arsitektur(mode):
    st.subheader("Beranda dan Arsitektur 5-Layer")
    st.write(
        "Setiap lapisan menerima keluaran lapisan sebelumnya. "
        "Teks di bawah mengikuti dataset yang dipilih di sidebar."
    )
    left, right = st.columns(2)
    for index, (nomor, judul, isi) in enumerate(mode["layers"]):
        target = left if index % 2 == 0 else right
        with target:
            st.info(f"**{nomor} — {judul}**\n\n{isi}")
    st.success(mode["flow"])


def page_komparasi(mode):
    st.subheader("Komparasi Performa Model")
    path = mode["comparison"]
    if not path.exists():
        st.warning(
            f"Berkas {path.relative_to(ROOT)} belum tersedia. "
            "Jalankan notebook Layer 4 untuk mode ini, lalu muat ulang halaman."
        )
        return

    comparison = load_table(str(path))
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    accuracy = comparison.set_index("Model")["Akurasi"]
    xgb_accuracy = float(accuracy.get("XGBoost", float("nan")))
    best_name = accuracy.idxmax()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Akurasi XGBoost", f"{xgb_accuracy:.2%}")
    col_b.metric("Akurasi tertinggi", best_name)
    col_c.metric("Model integrasi", "XGBoost")
    st.info(
        "XGBoost tetap menjadi model yang disimpan karena sistem integrasi "
        "dibangun di atas Extreme Gradient Boosting. "
        f"Akurasinya {xgb_accuracy:.2%}. Ambang rekomendasi adalah 50%. "
        "Tabel tetap menampilkan kelima algoritma."
    )

    images = [chart for chart in mode["charts"] if chart.exists()]
    if images:
        columns = st.columns(len(images))
        for column, chart in zip(columns, images):
            column.image(str(chart), caption=chart.name, use_column_width=True)
    else:
        st.caption("Grafik evaluasi untuk mode ini belum tersedia. Tabel metrik di atas tetap dapat dipakai.")


def page_inferensi(mode):
    st.subheader("Evaluasi Fitur Model (Manual)")
    academic = mode["features"] == ACADEMIC_FEATURES
    if academic:
        st.write(
            "Masukkan delapan fitur perilaku pelanggan Instacart. "
            "Nilai awal mengikuti satu baris uji pada layer5_inference.py."
        )
        defaults = {
            "total_prior_orders": 18.0,
            "avg_basket_size": 11.111111,
            "avg_days_between": 5.411765,
            "antecedent_rate": 0.055556,
            "rule_confidence": 0.315424,
            "rule_lift": 2.608102,
            "interest_confidence": 0.017524,
            "interest_lift": 0.144895,
        }
        labels = {
            "total_prior_orders": "Total pesanan prior",
            "avg_basket_size": "Rata-rata ukuran keranjang",
            "avg_days_between": "Rata-rata jeda hari",
            "antecedent_rate": "Antecedent rate",
            "rule_confidence": "Rule confidence",
            "rule_lift": "Rule lift",
            "interest_confidence": "Interest confidence",
            "interest_lift": "Interest lift",
        }
    else:
        st.write(
            "Masukkan delapan fitur konteks struk toko 2017: ukuran keranjang, "
            "jam, hari, akhir pekan, dan skor aturan Apriori."
        )
        defaults = {
            "current_basket_size": 8.0,
            "hour_of_day": 11.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "rule_confidence": 0.630952,
            "rule_lift": 348.322398,
            "antecedent_rate": 0.001230,
            "interest_lift": 0.428585,
        }
        labels = {
            "current_basket_size": "Ukuran keranjang saat ini",
            "hour_of_day": "Jam transaksi",
            "day_of_week": "Hari (0 = Senin)",
            "is_weekend": "Akhir pekan (1 = ya)",
            "rule_confidence": "Rule confidence",
            "rule_lift": "Rule lift",
            "antecedent_rate": "Antecedent rate historis",
            "interest_lift": "Interest lift",
        }

    values = {}
    with st.form(f"form_inferensi_{'akademis' if academic else 'riil'}"):
        left, right = st.columns(2)
        for index, name in enumerate(mode["features"]):
            target = left if index % 2 == 0 else right
            values[name] = target.number_input(
                labels[name],
                value=float(defaults[name]),
                step=0.000001,
                format="%.6f",
            )
        submitted = st.form_submit_button("Jalankan Prediksi", type="primary")

    if not submitted:
        st.info("Isi form di atas, lalu tekan Jalankan Prediksi.")
        return
    probability = predict_frame(mode, values)
    if probability is None:
        return
    show_decision(probability, "Membeli.", "Tidak Membeli.")


def load_mode_rules(mode):
    path = mode["rules"]
    if not path.exists():
        st.warning(
            f"Berkas {path.relative_to(ROOT)} belum tersedia. "
            "Jalankan notebook Layer 2 untuk mode ini."
        )
        return None
    rules = load_table(str(path))
    needed = [mode["product_column"], mode["pair_column"], "antecedent support", "confidence", "lift"]
    missing = [column for column in needed if column not in rules.columns]
    if rules.empty or missing:
        st.warning("Berkas aturan Apriori tidak memiliki kolom yang dibutuhkan dashboard.")
        return None
    return rules


def best_rule(rules, product_column, scanned):
    matches = rules.loc[rules[product_column].astype(str).eq(scanned)].copy()
    if matches.empty:
        return None
    return matches.sort_values(["lift", "confidence"], ascending=False).iloc[0]


def page_kasir(mode):
    st.subheader("Simulasi Belanja Kasir")
    academic = mode["features"] == ACADEMIC_FEATURES
    st.write(
        "Pindai satu produk. Sistem mengambil aturan dengan lift tertinggi, "
        "menyusun delapan fitur, lalu XGBoost memutuskan apakah pasangan produk ditawarkan."
    )
    rules = load_mode_rules(mode)
    if rules is None:
        return
    products = sorted(rules[mode["product_column"]].dropna().astype(str).unique())
    if not products:
        st.warning("Tidak ada produk antecedent pada berkas aturan.")
        return

    desk, receipt = st.columns(2, gap="large")
    with desk:
        st.markdown("**Struk aktif**")
        input_box = st.container(border=True)
        with input_box:
            if academic:
                profile_name = st.radio("Profil pelanggan", list(PROFILES))
                profile = PROFILES[profile_name]
                scanned = st.selectbox("Produk yang dipindai", products, key="scan_akademis")
                orders, basket, gap = st.columns(3)
                orders.metric("Pesanan prior", f"{profile['total_prior_orders']:.0f}")
                basket.metric("Ukuran keranjang", f"{profile['avg_basket_size']:.1f}")
                gap.metric("Jeda hari", f"{profile['avg_days_between']:.1f}")
                context = profile
            else:
                basket_size = st.number_input("Ukuran keranjang saat ini", min_value=1, max_value=80, value=4, step=1)
                hour = st.slider("Jam transaksi", min_value=0, max_value=23, value=11)
                day_name = st.selectbox("Hari belanja", WEEKDAYS, index=2)
                day_of_week = WEEKDAYS.index(day_name)
                scanned = st.selectbox("Produk yang dipindai", products, key="scan_riil")
                size_col, hour_col, day_col = st.columns(3)
                size_col.metric("Jumlah barang", f"{basket_size}")
                hour_col.metric("Jam", f"{hour:02d}:00")
                day_col.metric("Hari", day_name)
                context = {
                    "current_basket_size": float(basket_size),
                    "hour_of_day": float(hour),
                    "day_of_week": float(day_of_week),
                    "is_weekend": float(day_of_week >= 5),
                }

    rule = best_rule(rules, mode["product_column"], scanned)
    if rule is None:
        receipt.warning("Produk ini tidak memiliki aturan cross-selling.")
        return
    consequent = str(rule[mode["pair_column"]])
    antecedent_rate = float(rule["antecedent support"])
    rule_confidence = float(rule["confidence"])
    rule_lift = float(rule["lift"])
    if academic:
        features = {
            "total_prior_orders": float(context["total_prior_orders"]),
            "avg_basket_size": float(context["avg_basket_size"]),
            "avg_days_between": float(context["avg_days_between"]),
            "antecedent_rate": antecedent_rate,
            "rule_confidence": rule_confidence,
            "rule_lift": rule_lift,
            "interest_confidence": antecedent_rate * rule_confidence,
            "interest_lift": antecedent_rate * rule_lift,
        }
        note = (
            "Antecedent rate memakai antecedent support aturan. "
            "Interest confidence dan interest lift adalah rate dikali confidence atau lift."
        )
    else:
        features = {
            **context,
            "rule_confidence": rule_confidence,
            "rule_lift": rule_lift,
            "antecedent_rate": antecedent_rate,
            "interest_lift": antecedent_rate * rule_lift,
        }
        note = (
            "Antecedent rate adalah pangsa struk Januari–Februari yang memuat barang ini. "
            "Interest lift = antecedent rate × lift. Akhir pekan diisi otomatis dari hari yang dipilih."
        )

    with receipt:
        st.markdown("**Rekomendasi cross-selling**")
        with st.container(border=True):
            scanned_col, pair_col = st.columns(2)
            scanned_col.markdown(
                f'<div class="pair-card"><span>Produk yang dipindai</span><strong>{scanned}</strong></div>',
                unsafe_allow_html=True,
            )
            pair_col.markdown(
                f'<div class="pair-card"><span>Rekomendasi produk</span><strong>{consequent}</strong></div>',
                unsafe_allow_html=True,
            )
            confidence_col, lift_col = st.columns(2)
            confidence_col.metric("Confidence aturan", f"{rule_confidence:.2%}")
            lift_col.metric("Lift aturan", f"{rule_lift:.2f}")
            probability = predict_frame(mode, features)
            if probability is not None:
                show_decision(probability, "Rekomendasikan ke Pelanggan", "Tidak Perlu Ditawarkan")

    with st.expander("Lihat Ekstraksi 8 Fitur Otomatis Backend"):
        st.write(note)
        grid = st.columns(4)
        for index, name in enumerate(mode["features"]):
            grid[index % 4].metric(name, f"{features[name]:.4f}")


def main():
    st.sidebar.markdown("### Cross-Selling Ritel")
    st.sidebar.markdown(
        '<span class="pos-badge" style="background:#1F2633;border:1px solid #2E384D;color:#93A4BD;">Siap presentasi</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")
    mode_name = st.sidebar.selectbox("Pilih Mode Dataset:", [MODE_ACADEMIC, MODE_REAL])
    badge_color = "#4F46E5" if mode_name == MODE_ACADEMIC else "#10B981"
    st.sidebar.markdown(
        f'<span class="pos-badge" style="background:{badge_color};color:#F8FAFC;">{mode_name}</span>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigasi",
        [
            "1. Beranda & Arsitektur 5-Layer",
            "2. Komparasi Performa Model",
            "3. Evaluasi Fitur Model (Manual)",
            "4. Simulasi Belanja Kasir (POS)",
        ],
    )
    mode = MODES[mode_name]
    render_header(mode_name)
    if page.startswith("1."):
        page_arsitektur(mode)
    elif page.startswith("2."):
        page_komparasi(mode)
    elif page.startswith("3."):
        page_inferensi(mode)
    else:
        page_kasir(mode)


if __name__ == "__main__":
    main()
