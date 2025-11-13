import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# CONFIGURACIÓN DE LA APP
# ============================
st.set_page_config(
    page_title="Riesgo de Preeclampsia — ML",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background-color: #f5f6fa;
    }

    /* Encabezados */
    h1, h2, h3 {
        color: #2c3e50;
    }

    /* Tarjetas de resultado */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e1e1e1;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.06);
        text-align: center;
    }

    /* Texto destacado */
    .prob-text {
        font-size: 28px;
        font-weight: 700;
        color: #8e44ad;
    }

    /* Botón principal */
    div.stButton > button {
        background-color: #8e44ad;
        color: white;
        border-radius: 10px;
        padding: 8px 20px;
        font-size: 16px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #732d91;
    }

</style>
""", unsafe_allow_html=True)

# ============================
# CARGA DE ARTEFACTOS
# ============================
@st.cache_resource
def load_artifacts():
    ART_DIR = os.path.join("artefactos", "v1")

    with open(os.path.join(ART_DIR, "input_schema.json"), "r", encoding="utf-8") as f:
        input_schema = json.load(f)

    with open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)

    with open(os.path.join(ART_DIR, "decision_policy.json"), "r", encoding="utf-8") as f:
        policy = json.load(f)

    rev_label = {v: k for k, v in label_map.items()}

    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    pipe = joblib.load(os.path.join(ART_DIR, f"pipeline_{winner}.joblib"))

    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY = load_artifacts()

# ============================
# FUNCIÓN DE PREDICCIÓN
# ============================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, t in INPUT_SCHEMA.items():
        if c not in df.columns:
            df[c] = np.nan
        if str(t).lower().startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string")
    return df[FEATURES]

def predict_single(payload):
    df = _coerce_and_align(pd.DataFrame([payload]))
    prob = PIPE.predict_proba(df)[0][1]
    pred = int(prob >= THRESHOLD)
    return prob, pred

# ============================
# SIDEBAR
# ============================
st.sidebar.title("⚙️ Configuración")
st.sidebar.info("Ajusta parámetros del modelo o revisa información del sistema.")

st.sidebar.markdown(f"""
**Modelo ganador:** `{POLICY['winner']}`
**Umbral (threshold):** `{THRESHOLD:.2f}`  
""")

with st.sidebar.expander("📊 Métricas del Modelo"):
    st.write(POLICY["test_metrics"])

# ============================
# UI PRINCIPAL
# ============================
st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.write("Introduzca los datos clínicos de la paciente y el modelo evaluará su riesgo estimado.")

st.markdown("---")

# ============================
# FORMULARIO DE ENTRADA
# ============================
st.subheader("📋 Formulario de evaluación")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad (años)", min_value=10, max_value=60, value=30)
    imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
    p_sis = st.number_input("Presión Sistólica", 80, 200, 120)
    p_dia = st.number_input("Presión Diastólica", 50, 130, 80)

with col2:
    hipertension = st.selectbox("Hipertensión", ["NO", "SI"])
    diabetes = st.selectbox("Diabetes", ["NO", "SI"])
    ant_fam = st.selectbox("Antecedente Familiar Hipertensión", ["NO", "SI"])
    repro_asist = st.selectbox("Técnica de Reproducción Asistida", ["NO", "SI"])
    creatinina = st.number_input("Creatinina", 0.1, 5.0, 0.8)

if st.button("🔍 Calcular riesgo", use_container_width=True):
    payload = {
        "edad": edad,
        "imc": imc,
        "p_a_sistolica": p_sis,
        "p_a_diastolica": p_dia,
        "hipertension": hipertension,
        "diabetes": diabetes,
        "creatinina": creatinina,
        "ant_fam_hiper": ant_fam,
        "tec_repro_asistida": repro_asist,
    }

    prob, pred = predict_single(payload)
    prob_pct = prob * 100
    label = REV_LABEL[pred]

    st.markdown("---")
    st.subheader("📌 Resultado")

    # Tarjeta de resultado visual mejorada
    st.markdown(f"""
    <div class="result-card">
        <h3>Clasificación: <b>{label}</b></h3>
        <p class="prob-text">{prob_pct:.2f}%</p>
        <p><i>Probabilidad estimada de riesgo</i></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧾 Datos ingresados")
    st.dataframe(pd.DataFrame([payload]))
