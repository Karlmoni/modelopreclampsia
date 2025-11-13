import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
st.set_page_config(
    page_title="Predicción de Preeclampsia",
    page_icon="🩺",
    layout="wide",
)

# ESTILOS CSS PROFESIONALES
st.markdown(
    """
    <style>
        body {
            background-color: #f4f6f9;
        }
        .card {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
            border: 1px solid #e5e5e5;
        }
        .result {
            font-size: 36px;
            font-weight: 800;
            color: #1e3799;
            text-align: center;
        }
        .risk-high { color: #e55039 !important; }
        .risk-low { color: #1e90ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# SHAP EXPLAINER (KERNEL)
# ===============================
@st.cache_resource
def load_shap_explainer():
    background = pd.DataFrame(
        [np.zeros(len(FEATURES))], 
        columns=FEATURES
    )
    return shap.KernelExplainer(PIPE.predict_proba, background)


explainer = load_shap_explainer()

def explain_instance(instance_df):
    shap_values = explainer.shap_values(instance_df)
    return shap_values


# ===============================
# FUNCIÓN DE PREDICCIÓN
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
# UI - TABS
# ===============================
tab_pred, tab_interp, tab_info = st.tabs([
    "🩺 Predicción",
    "🔍 Interpretabilidad SHAP",
    "📘 Acerca del Modelo"
])

# ========================================================
# 🩺 TAB 1 — PREDICCIÓN
# ========================================================
with tab_pred:
    st.title("🩺 Predicción de Riesgo de Preeclampsia")
    st.write("Complete los datos clínicos para obtener una evaluación basada en un modelo de Machine Learning.")

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
        label = REV_LABEL[pred]
        pct = round(proba * 100, 2)

        color_class = "risk-high" if pred == 1 else "risk-low"

        st.markdown(
            f"""
            <div class="card">
                <h3 class="result {color_class}">{label}</h3>
                <p class="result {color_class}">{pct}%</p>
                <p style="text-align:center;">
                Probabilidad estimada de riesgo.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Explicación clínica simple
        st.subheader("📝 Interpretación Clínica (Automática)")
        if pred == 1:
            st.warning(
                """
                El modelo identifica un riesgo ELEVADO de preeclampsia.
                Se recomienda vigilancia estricta, control de PA y evaluación profesional.
                """
            )
        else:
            st.success(
                """
                El modelo estima un riesgo BAJO.  
                Aun así, se recomienda control rutinario y seguimiento clínico normal.
                """
            )

        st.subheader("📄 Datos ingresados")
        st.dataframe(pd.DataFrame([payload]), use_container_width=True)

# ========================================================
# 🔍 TAB 2 — INTERPRETABILIDAD SHAP
# ========================================================
with tab_interp:
    st.title("🔍 Interpretabilidad del Modelo (SHAP)")
    st.write("Visualice cómo cada variable influye en la predicción.")

    if st.button("Generar Interpretación SHAP"):
        shap_df = df_input.copy()
        shap_values = explain_instance(shap_df)

        st.subheader("📊 Waterfall (Explicación Individual)")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                shap_values[1][0],
                feature_names=FEATURES,
                data=shap_df.iloc[0].values
            )
        )
        st.pyplot(fig)

        st.subheader("📊 Summary Plot (Importancia Global)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[1], shap_df, plot_type="bar", show=False)
        st.pyplot(fig2)

# ========================================================
# 📘 TAB 3 — ACERCA DEL MODELO
# ========================================================
with tab_info:
    st.title("📘 Información del Modelo")
    st.write(f"**Modelo ganador:** {POLICY['winner']}")
    st.write(f"**Umbral de decisión:** {POLICY['threshold']}")

    st.subheader("📊 Métricas en Test")
    st.json(POLICY["test_metrics"])

    st.subheader("📁 Variables utilizadas")
    st.write(FEATURES)

    st.info("""
    """)
