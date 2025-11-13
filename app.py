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

# ================================
# Cargar artefactos
# ================================
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


# ==============================================
# Funciones auxiliares
# ==============================================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
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
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]
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
# TABS
# ================================
tab_pred, tab_model = st.tabs(["🩺 Predicción", "📘 Diseño del Modelo"])


# ================================
# TAB 1 — PREDICCIÓN (TU CÓDIGO ORIGINAL)
# ================================
with tab_pred:

    st.title("🩺 Predicción de Riesgo de Preeclampsia")
    st.write(
        """
Esta aplicación usa un modelo de *Machine Learning* entrenado para estimar 
el **riesgo de preeclampsia** en gestantes.

> ⚠️ *Herramienta académica, no reemplaza evaluación clínica profesional.*
"""
    )

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

    # Formulario
    st.subheader("📋 Ingrese los datos clínicos de la paciente")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad (años)", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0, 0.1)
            p_sis = st.number_input("Presión sistólica", 70, 250, 120)
            p_dia = st.number_input("Presión diastólica", 40, 150, 80)

        with col2:
            hipertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            ant_fam_hiper = st.selectbox("Antecedentes familiares de hipertensión", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            tec_repro_asistida = st.selectbox("Técnica de reproducción asistida", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            creatinina = st.number_input("Creatinina (mg/dL)", 0.1, 5.0, 0.8, 0.1)

        submitted = st.form_submit_button("Calcular riesgo")

    # Resultado
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

        results = predict_batch(payload)
        res = results[0]

        proba_pct = res["proba"] * 100
        label = res["pred_label"]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        if label == "RIESGO":
            st.error(
                f"**Clasificación:** {label}\n\n"
                f"Probabilidad estimada: **{proba_pct:.2f}%** "
                f"(umbral = {res['threshold']:.2f})"
            )
        else:
            st.success(
                f"**Clasificación:** {label}\n\n"
                f"Probabilidad estimada: **{proba_pct:.2f}%** "
                f"(umbral = {res['threshold']:.2f})"
            )

        st.markdown("#### Datos ingresados")
        st.dataframe(pd.DataFrame([payload]))

        st.info("Este resultado debe interpretarse siempre junto con una evaluación médica.")


# ================================
# TAB 2 — DISEÑO DEL MODELO
# ================================
with tab_model:

    st.title("📘 Diseño del Modelo")

    # Información general
    st.subheader("🧩 Información del pipeline")
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

    # Pasos del pipeline
    st.subheader("🔧 Pasos del pipeline")
    steps = [{"Paso": name, "Tipo": type(step).__name__}
             for name, step in PIPE.named_steps.items()]

    st.table(pd.DataFrame(steps))

    # Métricas
    st.subheader("📊 Métricas del modelo")
    metrics_df = pd.DataFrame(POLICY["test_metrics"].items(), columns=["Métrica", "Valor"])
    st.table(metrics_df)

    # Variables
    st.subheader("📁 Variables de entrada")
    vars_df = pd.DataFrame({"Variable": FEATURES})
    st.table(vars_df)
