import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

# =========================================================
# CONFIGURACIÓN GENERAL + DISEÑO MODERNO CLÍNICO
# =========================================================

st.set_page_config(
    page_title="Predicción de Preeclampsia — IA Clínica",
    page_icon="🩺",
    layout="wide",
)

# ======= ESTILO CSS PROFESIONAL =======
st.markdown("""
<style>

    /* Fondo suave */
    .stApp {
        background-color: #f5f7fa;
    }

    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 800;
    }

    /* Tarjeta moderna */
    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e0e6ed;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 25px;
    }

    /* Slider color */
    .stSlider > div > div > div > div {
        background: #2563eb !important;
    }

    /* Resultado riesgo */
    .risk-high {
        color: #dc2626;
        font-weight: 800;
        font-size: 32px;
        text-align: center;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 800;
        font-size: 32px;
        text-align: center;
    }

    /* Tablas elegantes */
    .styled-table thead th {
        background-color: #e9eef5;
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# CARGA DE ARTEFACTOS
# =========================================================
@st.cache_resource
def load_artifacts():
    ART = os.path.join("artefactos", "v1")

    with open(os.path.join(ART, "input_schema.json")) as f:
        input_schema = json.load(f)

    with open(os.path.join(ART, "label_map.json")) as f:
        label_map = json.load(f)

    with open(os.path.join(ART, "decision_policy.json")) as f:
        policy = json.load(f)

    rev_label = {v: k for k, v in label_map.items()}

    model_name = policy["winner"]
    threshold = float(policy["threshold"])

    pipeline = joblib.load(os.path.join(ART, f"pipeline_{model_name}.joblib"))

    return pipeline, input_schema, label_map, rev_label, threshold, policy


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, POLICY = load_artifacts()
FEATURES = list(INPUT_SCHEMA.keys())


# =========================================================
# FUNCIONES DE PREDICCIÓN
# =========================================================
def preprocess_input(record):
    df = pd.DataFrame([record])
    for col, dtype in INPUT_SCHEMA.items():
        if dtype.startswith("float") or dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")
    return df[FEATURES]


def predict(record):
    df = preprocess_input(record)
    proba = PIPE.predict_proba(df)[0][1]
    pred = int(proba >= THRESHOLD)
    return proba, pred, df


# =========================================================
# INTERFAZ POR PESTAÑAS
# =========================================================
tab_pred, tab_model = st.tabs(["🩺 Predicción", "📘 Acerca del Modelo"])


# =========================================================
# 🟦 1. PESTAÑA DE PREDICCIÓN
# =========================================================
with tab_pred:

    col_left, col_right = st.columns([1.2, 2])

    # ---- IZQUIERDA: CONFIGURACIÓN ----
    with col_left:
        st.subheader("⚙️ Configuración")

        THRESHOLD = st.slider(
            "Umbral de Clasificación",
            0.0, 1.0, THRESHOLD, 0.01,
            help="El modelo clasificará como 'RIESGO' si la probabilidad es mayor a este valor."
        )

    # ---- DERECHA: FORMULARIO ----
    with col_right:
        st.title("Sistema de Predicción de Riesgo de Preeclampsia")

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            edad = st.number_input("Edad (años)", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0)
            sist = st.number_input("Presión Sistólica", 80, 200, 120)
            diast = st.number_input("Presión Diastólica", 50, 130, 80)

        with c2:
            hipert = st.selectbox("Hipertensión", ["NO", "SI"])
            diab = st.selectbox("Diabetes", ["NO", "SI"])
            creat = st.number_input("Creatinina", 0.1, 5.0, 0.8)
            fam = st.selectbox("Antecedente Familiar de Hipertensión", ["NO", "SI"])
            repr = st.selectbox("Reproducción Asistida", ["NO", "SI"])

        st.markdown("</div>", unsafe_allow_html=True)

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

            st.subheader("📋 Datos Ingresados")
            st.dataframe(pd.DataFrame([payload]), use_container_width=True)


# =========================================================
# 🟦 2. PESTAÑA ACERCA DEL MODELO
# =========================================================
with tab_model:
    st.title("📘 Acerca del Modelo")

    # ---- INFORMACIÓN DEL PIPELINE ----
    st.subheader("🧩 Información del Pipeline")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    df_pipeline = pd.DataFrame({
        "Parámetro": ["Modelo Ganador", "Umbral de Decisión", "Clase Positiva", "Índice Clase Positiva", "Total Features"],
        "Valor": [
            POLICY["winner"],
            POLICY["threshold"],
            list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(1)],
            1,
            len(FEATURES)
        ]
    })

    st.table(df_pipeline.style.set_table_attributes("class='styled-table'"))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- PASOS DEL PIPELINE ----
    st.subheader("🔧 Pasos del Pipeline")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    pipeline_steps = [{"Paso": name, "Tipo": str(step.__class__.__name__)}
                      for name, step in PIPE.named_steps.items()]

    st.table(pd.DataFrame(pipeline_steps))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- MÉTRICAS ----
    st.subheader("📊 Métricas del Modelo")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.table(pd.DataFrame(POLICY["test_metrics"].items(), columns=["Métrica", "Valor"]))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- VARIABLES ----
    st.subheader("📁 Variables del Modelo")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.table(pd.DataFrame({"Variable": FEATURES}))

    st.markdown("</div>", unsafe_allow_html=True)
