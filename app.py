import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACIÓN GENERAL Y ENCABEZADO FIJO
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

"""
)

# =====================================================
# Cargar artefactos
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
# SIDEBAR INFO DEL MODELO
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
# PESTAÑAS
# =====================================================
T1, T2, T3 = st.tabs(
    ["🩺 Predicción", "📘 Diseño del Modelo", "📊 Importancia de Variables"]
)

# =====================================================
# TAB 1 — PREDICCIÓN
# =====================================================
with T1:

    st.subheader("📋 Ingrese los datos clínicos de la paciente")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad (años)", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0, 0.1)
            p_sis = st.number_input("Presión arterial sistólica (mmHg)", 70, 250, 120)
            p_dia = st.number_input("Presión arterial diastólica (mmHg)", 40, 150, 80)

        with col2:
            hipertension = st.selectbox("Antecedente de hipertensión", ["NO", "SI"])
            diabetes = st.selectbox("Antecedente de diabetes", ["NO", "SI"])
            ant_fam_hiper = st.selectbox("Antecedentes familiares de hipertensión", ["NO", "SI"])
            tec_repro_asistida = st.selectbox("Técnica de reproducción asistida", ["NO", "SI"])
            creatinina = st.number_input("Creatinina (mg/dL)", 0.1, 5.0, 0.8, 0.1)

        submitted = st.form_submit_button("Calcular riesgo")

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

        proba = PIPE.predict_proba(df)[0][1]
        pred  = int(proba >= THRESHOLD)
        label = REV_LABEL[pred]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        if label == "RIESGO":
            st.error(f"**Clasificación:** {label}\n\nProbabilidad estimada: **{proba*100:.2f}%**")
        else:
            st.success(f"**Clasificación:** {label}\n\nProbabilidad estimada: **{proba*100:.2f}%**")

        st.write("### Datos ingresados")
        st.dataframe(df)

        st.info("Este resultado debe interpretarse junto con la evaluación clínica profesional.")

# =====================================================
# TAB 2 — DISEÑO DEL MODELO
# =====================================================
with T2:

    st.header("📘 Diseño del Modelo")

    pos_label = [k for k, v in LABEL_MAP.items() if v == 1][0]

    st.subheader("🧩 Información del pipeline")

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

    st.subheader("🔧 Pasos del Pipeline")
    steps = [{"Paso": name, "Tipo": type(step).__name__} for name, step in PIPE.named_steps.items()]
    st.table(pd.DataFrame(steps))

    st.subheader("📊 Métricas del Modelo")
    st.table(pd.DataFrame(POLICY["test_metrics"].items(), columns=["Métrica", "Valor"]))

    st.subheader("📁 Variables de Entrada")
    st.table(pd.DataFrame({"Variable": FEATURES}))

# =====================================================
# TAB 3 — IMPORTANCIA DE VARIABLES (CORREGIDO)
# =====================================================
with T3:

    st.header("📊 Importancia de Variables")

    st.info(
        """
        ### 🧠 ¿Qué significa esta gráfica?

        Se toma un registro *neutro* y se modifica **una variable a la vez**.
        Se mide cuánto cambia la probabilidad estimada.
        
        - Barras altas → variable aumenta el riesgo.
        - Barras bajas → poca influencia.
        - Compatible con pipelines con OneHotEncoder + StandardScaler + SMOTE.
        """
    )

    st.write("### Sensibilidad del modelo")

    # Obtener columnas categóricas desde el pipeline REAL
    preprocessor = PIPE.named_steps["preprocessor"]
    numeric_cols = preprocessor.transformers_[0][2]
    categorical_cols = preprocessor.transformers_[1][2]

    # Crear registro neutro correcto
    base = {}
    for col in FEATURES:
        base[col] = "NO" if col in categorical_cols else 0

    df_base = pd.DataFrame([base])
    proba_base = PIPE.predict_proba(df_base)[0][1]

    # Calcular impacto variable por variable
    impacts = []
    for col in FEATURES:

        df_temp = df_base.copy()

        if col in categorical_cols:
            df_temp[col] = "SI"
        else:
            df_temp[col] = df_temp[col] + 1

        proba_new = PIPE.predict_proba(df_temp)[0][1]

        impacts.append({
            "Variable": col,
            "Impacto": float(proba_new - proba_base)
        })

    impacts_df = pd.DataFrame(impacts).sort_values("Impacto", ascending=False)

    # Gráfico
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(impacts_df["Variable"], impacts_df["Impacto"], color="#0077cc")
    ax.set_title("Importancia de cada variable")
    ax.set_ylabel("Cambio en probabilidad")
    ax.set_xticklabels(impacts_df["Variable"], rotation=45, ha="right")
    st.pyplot(fig)

    # Tabla final
    st.write("### Tabla de importancia (ordenada)")
    st.dataframe(impacts_df)
