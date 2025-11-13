import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer

# ===============================
# CONFIGURACIÓN GENERAL + DISEÑO
# ===============================
st.set_page_config(
    page_title="Predicción de Preeclampsia — IA",
    page_icon="🩺",
    layout="wide",
)

# ===== DISEÑO FUTURISTA / TECNOLÓGICO (CSS) =====
st.markdown("""
<style>

    /* Background futurista */
    .stApp {
        background: linear-gradient(135deg, #0a0f24 0%, #1a2a4a 50%, #0e1830 100%);
        color: #e6e6e6;
    }

    h1, h2, h3 {
        color: #9bc9ff !important;
        font-weight: 700;
    }

    /* Tarjetas estilo Glass */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        margin-bottom: 20px;
    }

    /* Botón futurista */
    div.stButton > button {
        background: linear-gradient(135deg, #3c8bff, #6bc1ff);
        border-radius: 10px;
        color: white;
        border: none;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 26px;
        transition: 0.25s;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #6bc1ff, #3c8bff);
        transform: scale(1.05);
    }

    /* Neón para riesgo */
    .neon-high {
        color: #ff6b6b;
        text-shadow: 0 0 8px rgba(255, 107, 107, 0.8);
        font-size: 42px;
        font-weight: 900;
        text-align: center;
    }
    .neon-low {
        color: #6bffb0;
        text-shadow: 0 0 8px rgba(107, 255, 176, 0.8);
        font-size: 42px;
        font-weight: 900;
        text-align: center;
    }

    /* Chips modernos */
    .chip {
        background: rgba(255, 255, 255, 0.12);
        padding: 10px 16px;
        border-radius: 12px;
        margin: 5px;
        display: inline-block;
        font-size: 14px;
        font-weight: 500;
        box-shadow: inset 0 0 8px rgba(255,255,255,0.25);
    }

</style>
""", unsafe_allow_html=True)

# ===============================
# CARGA DE ARTEFACTOS
# ===============================

@st.cache_resource
def load_artifacts():
    ART_DIR = os.path.join("artefactos", "v1")

    with open(os.path.join(ART_DIR, "input_schema.json")) as f:
        input_schema = json.load(f)

    with open(os.path.join(ART_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    with open(os.path.join(ART_DIR, "decision_policy.json")) as f:
        policy = json.load(f)

    rev_label = {v: k for k, v in label_map.items()}

    model_name = policy["winner"]
    threshold = float(policy["threshold"])

    pipeline = joblib.load(os.path.join(ART_DIR, f"pipeline_{model_name}.joblib"))

    return pipeline, input_schema, label_map, rev_label, threshold, policy


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, POLICY = load_artifacts()
FEATURES = list(INPUT_SCHEMA.keys())


# ===============================
# FUNCIONES DE PREDICCIÓN
# ===============================
def preprocess_input(record_dict):
    df = pd.DataFrame([record_dict])
    for col, dtype in INPUT_SCHEMA.items():
        if dtype.startswith("float") or dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")
    return df[FEATURES]

def predict(record_dict):
    df = preprocess_input(record_dict)
    proba = PIPE.predict_proba(df)[0][1]
    pred = int(proba >= THRESHOLD)
    return proba, pred, df


# ===============================
# TABS
# ===============================
tab_pred, tab_interp, tab_info = st.tabs([
    "🩺 Predicción",
    "🔍 Interpretabilidad IA",
    "📘 Acerca del Modelo"
])

# ========================================================
# 🩺 TAB 1 — PREDICCIÓN TECNOLÓGICA
# ========================================================
with tab_pred:
    st.title("🩺 Predicción de Riesgo de Preeclampsia — IA")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad (años)", 10, 60, 30)
        imc = st.number_input("IMC", 10.0, 60.0, 25.0)
        sist = st.number_input("Presión Sistólica", 80, 200, 120)
        diast = st.number_input("Presión Diastólica", 50, 130, 80)

    with col2:
        hipert = st.selectbox("Hipertensión", ["NO", "SI"])
        diab = st.selectbox("Diabetes", ["NO", "SI"])
        creat = st.number_input("Creatinina", 0.1, 5.0, 0.8)
        fam = st.selectbox("Antecedente Familiar Hipertensión", ["NO", "SI"])
        repr = st.selectbox("Reproducción Asistida", ["NO", "SI"])

    if st.button("🔍 Calcular Riesgo", use_container_width=True):
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": sist,
            "p_a_diastolica": diast,
            "hipertension": hipert,
            "diabetes": diab,
            "creatinina": creat,
            "ant_fam_hiper": fam,
            "tec_repro_asistida": repr
        }

        proba, pred, df_input = predict(payload)
        pct = round(proba * 100, 2)
        label = REV_LABEL[pred]

        if pred == 1:
            st.markdown(f"<p class='neon-high'>{label} — {pct}%</p>", unsafe_allow_html=True)
            st.warning("La IA detecta factores asociados a un riesgo ELEVADO. Requiere seguimiento clínico.")
        else:
            st.markdown(f"<p class='neon-low'>{label} — {pct}%</p>", unsafe_allow_html=True)
            st.success("Riesgo bajo según la IA. Mantener control prenatal rutinario.")

        st.subheader("📄 Datos ingresados")
        st.dataframe(pd.DataFrame([payload]), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================
# 🔍 TAB 2 — INTERPRETABILIDAD (GLOBAL + LIME)
# ========================================================
with tab_interp:
    st.title("🔍 Interpretabilidad Avanzada del Modelo")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Importancia Global de Variables")

    try:
        sample_df = pd.DataFrame([{k: 0 for k in FEATURES}])
        perm = permutation_importance(PIPE, sample_df, PIPE.predict(sample_df), n_repeats=8)

        importances = pd.DataFrame({
            "Variable": FEATURES,
            "Importancia": perm["importances_mean"]
        }).sort_values("Importancia", ascending=False)

        for idx, row in importances.iterrows():
            st.markdown(f"<span class='chip'>{row['Variable']}: {row['Importancia']:.4f}</span>", unsafe_allow_html=True)

        st.bar_chart(importances.set_index("Variable"))

    except Exception as e:
        st.warning(f"No se pudo calcular importancia global: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ========= LOCAL (LIME) =========
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🤖 Interpretación Individual con LIME")

    if "df_input" not in locals():
        st.info("⚠ Primero realiza una predicción.")
    else:
        explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(FEATURES))),
            feature_names=FEATURES,
            class_names=["SIN RIESGO", "RIESGO"],
            mode="classification"
        )

        exp = explainer.explain_instance(df_input.iloc[0].values, PIPE.predict_proba, num_features=6)

        st.write("### 🔬 Factores que influyeron:")
        st.write(exp.as_list())

        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

        if pred == 1:
            st.markdown("<p class='neon-high'>RIESGO ELEVADO</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='neon-low'>RIESGO BAJO</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================
# 📘 TAB 3 — INFORMACIÓN
# ========================================================
with tab_info:
    st.title("📘 Información del Modelo")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.write(f"**Modelo ganador:** {POLICY['winner']}")
    st.write(f"**Umbral de decisión:** {POLICY['threshold']}")

    st.subheader("📊 Métricas")
    st.json(POLICY["test_metrics"])

    st.subheader("📁 Variables usadas")
    st.write(FEATURES)

    st.info("⚠ Esta herramienta es apoyo clínico basado en IA, no reemplaza el criterio médico.")

    st.markdown("</div>", unsafe_allow_html=True)
