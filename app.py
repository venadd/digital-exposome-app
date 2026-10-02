import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Digital Exposome — Wellbeing Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Calibrated dataset medians
DEFAULTS = {
    'IBI': 0.017,
    'HR': 0.491,
    'NO2': 0.490,
    'Noise': 0.404,
    'NH3': 0.645,
    'PM10': 0.158,
    'CO': 0.805,
    'PM25': 0.176,
    'PM1': 0.176,
    'EDA': 0.216,
    'BVP': 0.512
}

FEATURE_ORDER = ['IBI', 'HR', 'NO2', 'Noise', 'NH3', 'PM10', 'CO', 'PM25', 'PM1', 'EDA', 'BVP']
MODEL_PATH = "rf_model.pkl"
DATA_PATH = "data.csv"

LABEL_METADATA = {
    1: {
        "title": "Level 1 — Severe Distress",
        "pill_bg": "#fee2e2",
        "pill_color": "#991b1b",
        "accent": "#dc2626",
        "summary": "Kombinasi polutan tinggi dan kebisingan memicu stres fisiologis berat.",
        "action": "Batasi aktivitas luar ruang. Masuki ruangan dengan filtrasi HEPA dan lakukan relaksasi pernapasan."
    },
    2: {
        "title": "Level 2 — Mild Distress",
        "pill_bg": "#ffedd5",
        "pill_color": "#9a3412",
        "accent": "#ea580c",
        "summary": "Stresor lingkungan mulai memengaruhi kenyamanan mental dan fisiologis.",
        "action": "Kurangi intensitas aktivitas luar ruangan dan perbanyak hidrasi tubuh."
    },
    3: {
        "title": "Level 3 — Normal Homeostasis",
        "pill_bg": "#fef9c3",
        "pill_color": "#854d0e",
        "accent": "#ca8a04",
        "summary": "Respon fisiologis stabil dalam batas ambang toleransi adaptasi tubuh normal.",
        "action": "Kondisi stabil. Pantau tren perubahan kualitas udara sekitar secara berkala."
    },
    4: {
        "title": "Level 4 — Positive Comfort",
        "pill_bg": "#dcfce7",
        "pill_color": "#166534",
        "accent": "#16a34a",
        "summary": "Kualitas lingkungan sehat selaras dengan ritme fisiologis tenang dan rileks.",
        "action": "Lingkungan sangat mendukung untuk aktivitas harian produktif maupun olahraga ringan."
    },
    5: {
        "title": "Level 5 — Optimal Wellbeing",
        "pill_bg": "#ccfbf1",
        "pill_color": "#115e59",
        "accent": "#0d9488",
        "summary": "Kondisi exposome prima: udara bersih, kebisingan minim, dan biometrik optimal.",
        "action": "Kondisi ideal untuk rekreasi, pemulihan performa, dan relaksasi menyeluruh."
    }
}

PRESETS = {
    "Pilih Preset Skenario...": None,
    "Taman Kota Segar (Optimal)": {
        "IBI": 0.28, "HR": 0.38, "NO2": 0.08, "Noise": 0.12, "NH3": 0.06,
        "PM10": 0.05, "CO": 0.09, "PM25": 0.06, "PM1": 0.04, "EDA": 0.12, "BVP": 0.58
    },
    "Jalan Raya Padat (Polusi & Bising)": {
        "IBI": 0.10, "HR": 0.74, "NO2": 0.82, "Noise": 0.88, "NH3": 0.69,
        "PM10": 0.72, "CO": 0.78, "PM25": 0.75, "PM1": 0.72, "EDA": 0.64, "BVP": 0.32
    },
    "Area Industri Berat (Stres Tinggi)": {
        "IBI": 0.05, "HR": 0.88, "NO2": 0.94, "Noise": 0.92, "NH3": 0.90,
        "PM10": 0.90, "CO": 0.88, "PM25": 0.93, "PM1": 0.90, "EDA": 0.82, "BVP": 0.20
    },
    "Ruang Kantor Terkendali": {
        "IBI": 0.18, "HR": 0.50, "NO2": 0.28, "Noise": 0.25, "NH3": 0.22,
        "PM10": 0.14, "CO": 0.18, "PM25": 0.16, "PM1": 0.13, "EDA": 0.22, "BVP": 0.50
    }
}

# ==============================================================================
# CLEAN SAAS STYLING (ZERO EMOJI, SLEEK & COMPACT)
# ==============================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Base */
    .stApp {
        background-color: #f7f9f7 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0f172a !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.8rem !important;
    }
    
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 1360px !important;
    }

    /* Sidebar Clean */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2ece4 !important;
    }

    /* Menu Navigasi Modern */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 5px !important;
        padding-top: 2px !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        background: #f8fafc !important;
        border: 1px solid #eef2f6 !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #ecfdf5 !important;
        border-color: #a7f3d0 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 0 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: #10b981 !important;
        border-color: #059669 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.22) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Card Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2ece4 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
        padding: 14px 16px !important;
        margin-bottom: 10px !important;
    }

    /* Label Slider */
    div[data-testid="stSlider"] label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Header Simpel */
    .simple-header {
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5ece6;
    }
    .simple-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .simple-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* KPI Strip Minimalis */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2ece4;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .kpi-lbl {
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .kpi-num {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }

    /* Outcome Card */
    .diag-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .diag-num {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }

    /* Tombol Bersih, Presisi & Sejajar */
    div.stButton > button {
        height: 40px !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }

    /* Rapatkan jarak antar widget */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# STATE MANAGEMENT & MODEL LOADER
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_rf_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_raw_dataset():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None

model = load_rf_pipeline()
df_data = load_raw_dataset()

for feat, val in DEFAULTS.items():
    if feat not in st.session_state:
        st.session_state[feat] = val

def apply_preset(p_name):
    if p_name and p_name in PRESETS and PRESETS[p_name] is not None:
        for k, v in PRESETS[p_name].items():
            st.session_state[k] = float(v)

def reset_all_defaults():
    for feat, val in DEFAULTS.items():
        st.session_state[feat] = float(val)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 12px 0; border-bottom: 1px solid #eef2f6; margin-bottom: 12px;">
        <div style="width: 32px; height: 32px; border-radius: 8px; background: #10b981; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.85rem; font-weight: 800; letter-spacing: -0.05em;">
            DE
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1rem; color: #0f172a; line-height: 1.15; letter-spacing: -0.02em;">ExposomeHub</div>
            <div style="font-size: 0.7rem; color: #10b981; font-weight: 600;">Wellbeing Classifier</div>
        </div>
    </div>
    <div style="font-size: 0.68rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;">
        Menu Navigasi
    </div>
    """, unsafe_allow_html=True)

    nav_item = st.radio(
        "Menu Navigasi",
        ["Telemetri & Prediksi", "Feature Importance", "Data Observasi 42k", "Ringkasan Riset"],
        index=0,
        label_visibility="collapsed"
    )

    st.write("")
    st.markdown("""
    <div style="font-size: 0.68rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;">
        Skenario Kalibrasi
    </div>
    """, unsafe_allow_html=True)

    selected_preset = st.selectbox(
        "Preset:",
        list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    if selected_preset != "Pilih Preset Skenario...":
        apply_preset(selected_preset)

    st.write("")
    st.markdown("""
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px; margin-top: 10px;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #166534; margin-bottom: 2px;">Random Forest Pipeline</div>
        <div style="font-size: 0.72rem; color: #15803d; line-height: 1.4; margin-bottom: 6px;">Akurasi validasi 96.64% berbasis fusi 11 sensor biometrik & kualitas udara.</div>
        <div style="background: #ffffff; color: #059669; font-weight: 700; font-size: 0.68rem; padding: 3px 8px; border-radius: 6px; border: 1px solid #a7f3d0; text-align: center;">
            MinMaxScaler + 100 Trees
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# HEADER SIMPEL (BERSIH, TANPA EMOJI & TANPA KOTAK BESAR)
# ==============================================================================
st.markdown("""
<div class="simple-header">
    <h1 class="simple-title">Pemantauan & Klasifikasi Kesejahteraan</h1>
    <div class="simple-sub">Memprediksi tingkat kenyamanan dan stres seseorang (skala 1–5) dengan menggabungkan respons fisik tubuh (detak jantung, variabilitas nadi, aktivitas kulit) dan kondisi lingkungan sekitar (polusi udara serta kebisingan).</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# METRIK KPI MINIMALIS
# ==============================================================================
st.markdown("""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-lbl">Akurasi Model RF</div>
        <div class="kpi-num">96.64%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">Total Baris Sampel</div>
        <div class="kpi-num">42.436</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">Sensor Multimodal</div>
        <div class="kpi-num">11 Fitur</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">Skala Kesejahteraan</div>
        <div class="kpi-num">5 Tingkat</div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"Berkas model '{MODEL_PATH}' tidak ditemukan.")
    st.stop()

# ==============================================================================
# HALAMAN 1: TELEMETRI & PREDIKSI
# ==============================================================================
if "Telemetri" in nav_item:
    col_input, col_result = st.columns([1.15, 0.85], gap="medium")

    with col_input:
        # Card 1: Fisiologis Tubuh
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>1. Sensor Fisiologis Tubuh (Wearable Empatica E4)</div>", unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                st.slider("Heart Rate (HR)", 0.0, 1.0, step=0.001, key="HR",
                          help="Denyut jantung ternormalisasi (bpm).")
                st.slider("Inter-Beat Interval (IBI)", 0.0, 1.0, step=0.001, key="IBI",
                          help="Variabilitas antardetak jantung (HRV).")
            with f2:
                st.slider("Electrodermal Activity (EDA)", 0.0, 1.0, step=0.001, key="EDA",
                          help="Konduktansi kulit terhadap respons stres simpatis.")
                st.slider("Blood Volume Pulse (BVP)", 0.0, 1.0, step=0.001, key="BVP",
                          help="Amplitudo denyut volume darah vaskular.")

        # Card 2: Lingkungan & Kualitas Udara
        with st.container(border=True):
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>2. Sensor Lingkungan & Kualitas Udara</div>", unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1:
                st.slider("Noise Level (Kebisingan dB)", 0.0, 1.0, step=0.001, key="Noise")
                st.slider("PM 1.0 (Partikel Ultrafine)", 0.0, 1.0, step=0.001, key="PM1")
                st.slider("PM 2.5 (Partikel Halus)", 0.0, 1.0, step=0.001, key="PM25")
                st.slider("PM 10 (Partikel Debu Kasar)", 0.0, 1.0, step=0.001, key="PM10")
            with e2:
                st.slider("Nitrogen Dioksida (NO2)", 0.0, 1.0, step=0.001, key="NO2")
                st.slider("Karbon Monoksida (CO)", 0.0, 1.0, step=0.001, key="CO")
                st.slider("Amonia (NH3)", 0.0, 1.0, step=0.001, key="NH3")

        # Tombol Aksi Sejajar & Presisi (Tanpa Emoji)
        b1, b2 = st.columns([1.3, 0.7], gap="small")
        with b1:
            st.button("Prediksi Status", type="primary", use_container_width=True)
        with b2:
            st.button("Reset Nilai", type="secondary", on_click=reset_all_defaults, use_container_width=True)

    with col_result:
        input_data = pd.DataFrame([{
            'IBI': st.session_state['IBI'],
            'HR': st.session_state['HR'],
            'NO2': st.session_state['NO2'],
            'Noise': st.session_state['Noise'],
            'NH3': st.session_state['NH3'],
            'PM10': st.session_state['PM10'],
            'CO': st.session_state['CO'],
            'PM25': st.session_state['PM25'],
            'PM1': st.session_state['PM1'],
            'EDA': st.session_state['EDA'],
            'BVP': st.session_state['BVP']
        }])[FEATURE_ORDER]

        pred_class = int(model.predict(input_data)[0])
        pred_probs = model.predict_proba(input_data)[0]
        classes = model.classes_

        meta = LABEL_METADATA.get(pred_class, LABEL_METADATA[3])
        top_prob = float(np.max(pred_probs))

        with st.container(border=True):
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Hasil Diagnosa Model</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="diag-box">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #047857;">Tingkat Kesejahteraan</div>
                        <div class="diag-num" style="color: {meta['accent']};">
                            {pred_class} <span style="font-size: 1.15rem; font-weight: 600; color: #64748b;">/ 5</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {meta['pill_bg']}; color: {meta['pill_color']}; font-weight: 700; font-size: 0.72rem; padding: 4px 10px; border-radius: 9999px;">
                            Keyakinan: {top_prob*100:.1f}%
                        </span>
                    </div>
                </div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0f172a; margin-top: 4px;">
                    {meta['title']}
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.45; margin-top: 3px;">
                    {meta['summary']}
                </div>
                <div style="background: #ffffff; border-radius: 6px; padding: 8px 10px; border: 1px solid #d1fae5; border-left: 3px solid {meta['accent']}; font-size: 0.76rem; color: #1e293b; margin-top: 8px; line-height: 1.4;">
                    <b style="color: #065f46;">Rekomendasi:</b> {meta['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #334155; margin-bottom: 2px;'>Distribusi Probabilitas Kelas</div>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(5.5, 1.9))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')

            colors = ['#10b981' if c == pred_class else '#e2e8f0' for c in classes]
            y_pos = np.arange(len(classes))
            bars = ax.barh(y_pos, pred_probs, color=colors, height=0.55, edgecolor='none')

            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"Level {c}" for c in classes], fontfamily='sans-serif', fontsize=8, color='#334155', fontweight='600')
            ax.set_xlim(0, 1.05)
            ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', colors='#64748b', labelsize=7.5)

            for bar in bars:
                w = bar.get_width()
                if w > 0.03:
                    ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%",
                            va='center', ha='left', color='#0f172a', fontsize=7.5, fontweight='bold')

            for spine in ax.spines.values():
                spine.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

# ==============================================================================
# HALAMAN 2: FEATURE IMPORTANCE
# ==============================================================================
elif "Feature Importance" in nav_item:
    with st.container(border=True):
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 2px;'>Peringkat Kepentingan Fitur Sensor</div>", unsafe_allow_html=True)
        st.caption("Kontribusi relatif masing-masing sensor fisiologis dan polutan udara terhadap keputusan klasifikasi Random Forest.")

        clf_step = model.named_steps['clf']
        weights = clf_step.feature_importances_
        feat_df = pd.DataFrame({
            "Sensor": FEATURE_ORDER,
            "Kategori": [
                "Fisiologis" if f in ['HR', 'IBI', 'EDA', 'BVP'] else "Lingkungan"
                for f in FEATURE_ORDER
            ],
            "Importance": weights
        }).sort_values(by="Importance", ascending=True)

        col_c, col_t = st.columns([1.2, 0.8], gap="medium")

        with col_c:
            fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.0))
            fig_imp.patch.set_facecolor('#ffffff')
            ax_imp.set_facecolor('#ffffff')

            bar_colors = ['#10b981' if cat == 'Fisiologis' else '#6ee7b7' for cat in feat_df['Kategori']]
            y_slots = np.arange(len(feat_df))
            ax_imp.barh(y_slots, feat_df['Importance'], color=bar_colors, height=0.6)

            ax_imp.set_yticks(y_slots)
            ax_imp.set_yticklabels(feat_df['Sensor'], fontfamily='sans-serif', fontsize=8.5, color='#1e293b', fontweight='600')
            ax_imp.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
            ax_imp.set_axisbelow(True)
            ax_imp.tick_params(axis='x', colors='#64748b', labelsize=8)

            for spine in ax_imp.spines.values():
                spine.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig_imp)
            st.caption("Hijau Tua: Sensor Fisiologis Tubuh | Hijau Muda: Sensor Lingkungan")

        with col_t:
            st.write("##### Tabel Kontribusi:")
            st.dataframe(
                feat_df.sort_values(by="Importance", ascending=False).style.format({"Importance": "{:.4f}"}),
                use_container_width=True,
                height=320
            )

# ==============================================================================
# HALAMAN 3: EKSPLORASI DATASET
# ==============================================================================
elif "Data Observasi" in nav_item:
    with st.container(border=True):
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 2px;'>Eksplorasi Korpus Data Digital Exposome (42.436 Baris)</div>", unsafe_allow_html=True)
        
        if df_data is not None:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.write("##### Frekuensi Sebaran Target (Label 1-5)")
                counts = df_data['Label'].value_counts().sort_index()
                fig_h, ax_h = plt.subplots(figsize=(5.5, 3.0))
                fig_h.patch.set_facecolor('#ffffff')
                ax_h.set_facecolor('#ffffff')

                palette = ['#fca5a5', '#fdba74', '#fde047', '#86efac', '#10b981']
                ax_h.bar(counts.index.astype(str), counts.values, color=palette, width=0.55)
                ax_h.yaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
                ax_h.set_xlabel("Label Kesejahteraan", color='#64748b', fontsize=8)
                ax_h.set_ylabel("Jumlah Sampel", color='#64748b', fontsize=8)
                ax_h.tick_params(colors='#64748b', labelsize=8)

                for spine in ax_h.spines.values():
                    spine.set_visible(False)

                st.pyplot(fig_h)

            with c2:
                st.write("##### Matriks Korelasi Antarsensor")
                fig_c, ax_c = plt.subplots(figsize=(5.5, 3.2))
                fig_c.patch.set_facecolor('#ffffff')
                ax_c.set_facecolor('#ffffff')

                corr_mat = df_data.corr()
                cax = ax_c.imshow(corr_mat, cmap="Greens", aspect='auto', vmin=-0.2, vmax=1.0)
                fig_c.colorbar(cax, ax=ax_c, shrink=0.8)
                ax_c.set_xticks(range(len(corr_mat.columns)))
                ax_c.set_yticks(range(len(corr_mat.columns)))
                ax_c.set_xticklabels(corr_mat.columns, rotation=90, color='#1e293b', fontsize=7.5)
                ax_c.set_yticklabels(corr_mat.columns, color='#1e293b', fontsize=7.5)

                plt.tight_layout()
                st.pyplot(fig_c)

            st.write("##### 50 Baris Pertama:")
            st.dataframe(df_data.head(50), use_container_width=True)

# ==============================================================================
# HALAMAN 4: RINGKASAN RISET
# ==============================================================================
else:
    with st.container(border=True):
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>Metodologi & Teori Digital Exposome</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Definisi:** Konsep Exposome mengukur akumulasi seluruh rangsangan lingkungan fisik (kualitas udara, aerosol, kebisingan) yang berinteraksi langsung dengan sistem fisiologis tubuh.
        - **Sumber Dataset:** *DigitalExposome: A Dataset for Wellbeing Classification using Environmental Air Quality and Human Physiological data* (Data in Brief, 2025).
        - **Partisipan:** 40 subjek pada lintasan rute perkotaan dinamis.
        - **Model Machine Learning:** Scikit-Learn Pipeline `MinMaxScaler` + `RandomForestClassifier` (100 Trees) dengan akurasi pengujian **96.64%**.
        """)
