import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ===============================
# CONFIGURACIÓN GENERAL + DISEÑO
# ===============================
st.set_page_config(
    page_title="Predicción de Preeclampsia — IA",
    page_icon="🩺",
    layout="wide",
)

# ===== DISEÑO PROFESIONAL CLÍNICO-TECNOLÓGICO =====
st.markdown("""
<style>

    /* Fondo claro moderno */
    .stApp {
        background-color: #f4f7fb;
        color: #2c3e50;
    }

    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700;
    }

    /* Tarjetas estilo dashboard */
    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.09);
        border: 1px solid #e0e6ed;
        margin-bottom: 20px;
    }

    /* Botón moderno */
    div.stButton > button {
        background-color: #2563eb;
        border-radius: 10px;
        color: white;
        border: none;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
    }

    /* Resultado riesgo */
    .risk-high {
        color: #e11d48;
        font-size: 36px;
        text-align:center;
        font-weight: 800;
    }
    .risk-low {
        color: #059669;
        font-size: 36px;
        text-align:center;
        font-weight: 800;
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

# ========================================================
# TABS
# ========================================================
tab_pred, tab_interp, tab_info = st.tabs([
    "🩺 Predicción",
    "🧠 Interpretabilidad del Modelo",
    "📘 Acerca del Modelo"
])

# ========================================================
# 🩺 TAB 1 — PREDICCIÓN
# ========================================================
with tab_pred:
    st.title("🩺 Predicción de Riesgo de Preeclampsia")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

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
        fam = st.selectbox("Antecedente Familiar de Hipertensión", ["NO", "SI"])
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
            st.markdown(f"<p class='risk-high'>{label} — {pct}%</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='risk-low'>{label} — {pct}%</p>", unsafe_allow_html=True)

        st.subheader("📄 Datos ingresados")
        st.dataframe(pd.DataFrame([payload]), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# 🧠 TAB 2 — INTERPRETABILIDAD
# ========================================================
with tab_interp:
    st.title("🧠 Interpretabilidad del Modelo (Explicación IA)")
    st.write("Explora cómo la IA toma decisiones.")

    # 1. IMPORTANCIA GLOBAL (PERMUTATION IMPORTANCE)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Importancia Global de Variables")

    try:
        sample_df = pd.DataFrame([{k: 0 for k in FEATURES}])
        perm = permutation_importance(
            PIPE, sample_df, PIPE.predict(sample_df), n_repeats=10
        )

        importances = pd.DataFrame({
            "Variable": FEATURES,
            "Importancia": perm["importances_mean"]
        }).sort_values("Importancia", ascending=False)

        st.bar_chart(importances.set_index("Variable"))

    except Exception as e:
        st.warning(f"No se pudo calcular importancia global: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. EXPLICACIÓN LOCAL SIMULADA (CAMBIO DE UNA VARIABLE)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Interpretación Individual (Simulación IA)")

    if "df_input" not in locals():
        st.info("⚠ Realiza una predicción primero.")
    else:
        st.write("La IA simula cambios en una variable a la vez para ver su impacto:")

        impacts = {}

        for var in FEATURES:
            row = df_input.copy()

            numeric = pd.api.types.is_numeric_dtype(row[var])

            if numeric:
                row[var] = row[var] * 1.20
            else:
                row[var] = "SI" if row[var].iloc[0] == "NO" else "NO"

            new_proba = PIPE.predict_proba(row)[0][1]
            impacts[var] = new_proba - proba

        impacts_df = pd.DataFrame({
            "Variable": impacts.keys(),
            "Impacto": impacts.values()
        }).sort_values("Impacto", ascending=False)

        st.write(impacts_df)

        st.info("Valores positivos aumentan el riesgo. Valores negativos lo reducen.")

    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# 📘 TAB 3 — INFORMACIÓN
# ========================================================
with tab_info:
    st.title("📘 Información del Modelo")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write(f"**Modelo ganador:** {POLICY['winner']}")
    st.write(f"**Umbral de decisión:** {POLICY['threshold']}")

    st.subheader("📊 Métricas")
    st.json(POLICY["test_metrics"])

    st.subheader("📁 Variables usadas")
    st.write(FEATURES)

    st.info("⚠ Esta herramienta es apoyo clínico basado en IA, no reemplaza la evaluación médica.")

    st.markdown("</div>", unsafe_allow_html=True)
