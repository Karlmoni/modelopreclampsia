import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================
# Configuración general de la app
# ================================
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

# ================================
# Cargar artefactos (modelo, schema, policy)
# ================================
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():
    # Cargar JSONs
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

    # Cargar pipeline ganador
    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner_name}.joblib")
    pipe = joblib.load(pipe_path)

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY = load_artifacts()

# ==============================================
# Funciones auxiliares (mismas ideas del notebook)
# ==============================================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura tipos según INPUT_SCHEMA y alinea columnas en el orden esperado.
    """
    for c, t in INPUT_SCHEMA.items():
        if c not in df.columns:
            df[c] = np.nan

        t_str = str(t).lower()
        if t_str.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t_str in ("bool", "boolean"):
            df[c] = df[c].astype("bool")
        else:
            df[c] = df[c].astype("string")

    return df[FEATURES]


def predict_batch(records, thr=None):
    """
    records: dict o lista de dicts con las features de entrada.
    thr: umbral opcional, si no se pasa se usa THRESHOLD.
    """
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]  # Prob(RIESGO=1 | x)
    preds = (proba >= thr).astype(int)

    results = []
    for p, y in zip(proba, preds):
        results.append(
            {
                "proba": float(p),
                "pred_int": int(y),
                "pred_label": REV_LABEL[int(y)],
                "threshold": thr,
            }
        )
    return results

# ================================
# Sidebar: info del modelo
# ================================
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

# ================================
# Formulario de entrada de datos
# ================================
st.subheader("📋 Ingrese los datos clínicos de la paciente")

with st.form("form_paciente"):
    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad (años)", min_value=10, max_value=60, value=30)
        imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        p_sis = st.number_input(
            "Presión arterial sistólica (mmHg)",
            min_value=70,
            max_value=250,
            value=120,
        )
        p_dia = st.number_input(
            "Presión arterial diastólica (mmHg)",
            min_value=40,
            max_value=150,
            value=80,
        )

    with col2:
        hipertension = st.selectbox(
            "Antecedente de hipertensión",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
        )
        diabetes = st.selectbox(
            "Antecedente de diabetes",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
        )
        ant_fam_hiper = st.selectbox(
            "Antecedentes familiares de hipertensión",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
        )
        tec_repro_asistida = st.selectbox(
            "Uso de técnica de reproducción asistida",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
        )
        creatinina = st.number_input(
            "Creatinina (mg/dL)",
            min_value=0.1,
            max_value=5.0,
            value=0.8,
            step=0.1,
        )

    submitted = st.form_submit_button("Calcular riesgo")

# ================================
# Predicción
# ================================
if submitted:
    # Crear payload tal como espera el modelo
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

    results = predict_batch(payload)
    res = results[0]

    proba_pct = res["proba"] * 100
    label = res["pred_label"]

    st.markdown("---")
    st.subheader("🔍 Resultado del modelo")

    # Tarjeta principal
    if label == "RIESGO":
        st.error(
            f"**Clasificación:** {label}\n\n"
            f"Probabilidad estimada de riesgo: **{proba_pct:.2f}%** "
            f"(umbral = {res['threshold']:.2f})"
        )
    else:
        st.success(
            f"**Clasificación:** {label}\n\n"
            f"Probabilidad estimada de riesgo: **{proba_pct:.2f}%** "
            f"(umbral = {res['threshold']:.2f})"
        )

    # Mostrar tabla con los datos de entrada
    st.markdown("#### Datos ingresados")
    st.dataframe(pd.DataFrame([payload]))
    
    st.info(
        "Este resultado debe interpretarse siempre junto con la historia clínica "
        "y la evaluación de un profesional de la salud."
    )
