import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# CONFIGURACIÓN GENERAL (ENCABEZADO FIJO)
# =====================================================
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.write(
    """
Esta aplicación usa un modelo de *Machine Learning* entrenado para estimar 
el **riesgo de preeclampsia** en gestantes.

> ⚠️ **Aviso importante:** esta herramienta es solo de apoyo académico y no reemplaza 
> el criterio clínico ni la evaluación médica profesional.
"""
)

# =====================================================
# CARGA DE ARTEFACTOS
# =====================================================
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():
    input_schema_path = os.path.join(ART_DIR, "input_schema.json")
    label_map_path    = os.path.join(ART_DIR, "label_map.json")
    policy_path       = os.path.join(ART_DIR, "decision_policy.json")

    with open(input_schema_path, "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner_name = policy["winner"]
    threshold   = float(policy.get("threshold", 0.5))

    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner_name}.joblib")
    pipe = joblib.load(pipe_path)

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY = load_artifacts()


# =====================================================
# BARRA LATERAL — INFORMACIÓN DEL MODELO
# =====================================================
st.sidebar.header("ℹ️ Información del modelo")
st.sidebar.markdown(f"""
**Modelo ganador:** `{POLICY['winner']}`  
**Umbral de decisión:** `{THRESHOLD:.2f}`  

**Métricas en test:**
- F1 = `{POLICY['test_metrics']['f1']:.3f}`
- Precisión = `{POLICY['test_metrics']['precision']:.3f}`
- Recall = `{POLICY['test_metrics']['recall']:.3f}`
- ROC-AUC = `{POLICY['test_metrics']['roc_auc']:.3f}`
- PR-AUC = `{POLICY['test_metrics']['pr_auc']:.3f}`
""")


# =====================================================
# PESTAÑAS — LETRA IGUAL EN AMBAS
# =====================================================
tab_pred, tab_model = st.tabs(
    ["🩺 Predicción", "📘 Diseño del Modelo"]
)

# ======================================================================
# TAB 1 — PREDICCIÓN (TAL COMO TU ORIGINAL, SOLO CORREGIDO SI/NO)
# ======================================================================
with tab_pred:

    st.subheader("📋 Ingrese los datos clínicos de la paciente")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad (años)", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0, 0.1)
            p_sis = st.number_input("Presión arterial sistólica", 70, 250, 120)
            p_dia = st.number_input("Presión arterial diastólica", 40, 150, 80)

        with col2:
            hipertension = st.selectbox("Antecedente de hipertensión", ["NO", "SI"])
            diabetes = st.selectbox("Antecedente de diabetes", ["NO", "SI"])
            ant_fam_hiper = st.selectbox("Antecedentes familiares de hipertensión", ["NO", "SI"])
            tec_repro_asistida = st.selectbox("Técnica de reproducción asistida", ["NO", "SI"])

            creatinina = st.number_input(
                "Creatinina (mg/dL)",
                min_value=0.1,
                max_value=5.0,
                value=0.8,
                step=0.1,
            )

        submitted = st.form_submit_button("Calcular riesgo")

    # -----------------------
    # PREDICCIÓN DEL MODELO
    # -----------------------
    if submitted:

        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": p_sis,
            "p_a_diastolica": p_dia,
            "hipertension": hipertension,
            "diabetes": diabetes,
            "creatinina": creatinina,
            "ant_fam_hiper": ant_fam_hiper,
            "tec_repro_asistida": tec_repro_asistida,
        }

        df = pd.DataFrame([payload])

        # Salida del modelo
        proba = PIPE.predict_proba(df)[0][1]
        pred = int(proba >= THRESHOLD)
        label = REV_LABEL[pred]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        if label == "RIESGO":
            st.error(f"**Clasificación:** {label}\n\nProbabilidad: **{proba*100:.2f}%**")
        else:
            st.success(f"**Clasificación:** {label}\n\nProbabilidad: **{proba*100:.2f}%**")

        st.markdown("#### Datos ingresados")
        st.dataframe(df)

        st.info("Interpretar siempre junto con evaluación clínica.")


# ======================================================================
# TAB 2 — DISEÑO DEL MODELO (NUEVA SECCIÓN)
# ======================================================================
with tab_model:

    st.header("📘 Diseño del Modelo")

    # -----------------------------
    # CONFIGURACIÓN DEL PIPELINE
    # -----------------------------
    st.subheader("🧩 Información del Pipeline")

    pos_label = [k for k, v in LABEL_MAP.items() if v == 1][0]

    cfg_df = pd.DataFrame({
        "Parámetro": [
            "Modelo ganador",
            "Umbral de decisión",
            "Clase positiva",
            "Código clase positiva",
            "Total de features"
        ],
        "Valor": [
            POLICY["winner"],
            f"{THRESHOLD:.3f}",
            pos_label,
            LABEL_MAP[pos_label],
            len(FEATURES)
        ]
    })

    st.table(cfg_df)

    # -----------------------------
    # PASOS DEL PIPELINE
    # -----------------------------
    st.subheader("🔧 Pasos del Pipeline")

    steps = [{"Paso": name, "Tipo": type(step).__name__}
             for name, step in PIPE.named_steps.items()]

    st.table(pd.DataFrame(steps))

    # -----------------------------
    # MÉTRICAS DEL MODELO
    # -----------------------------
    st.subheader("📊 Métricas del Modelo")

    metrics_df = pd.DataFrame(POLICY["test_metrics"].items(), columns=["Métrica", "Valor"])
    st.table(metrics_df)

    # -----------------------------
    # VARIABLES DE ENTRADA
    # -----------------------------
    st.subheader("📁 Variables de Entrada")

    vars_df = pd.DataFrame({"Variable": FEATURES})
    st.table(vars_df)
