# GlucoPredict
# TCN-BiGRU Deep Learning Model | Single-Step Multi-Horizon Forecasting

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib, os, warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GlucoPredict",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR    = "."                    # files are in repo root on Streamlit Cloud
MODEL_FILE   = "tcn_bigru_final.keras"
FEAT_SCALER  = "feature_scaler.pkl"
TGT_SCALER   = "target_scaler.pkl"
HORIZONS     = [5, 10, 15, 30, 60]
NORMAL_RANGES = {
    "BGL":      (70,   180,  "mg/dL"),
    "DBP":      (60,   90,   "mmHg"),
    "SBP":      (90,   130,  "mmHg"),
    "HR":       (60,   100,  "bpm"),
    "Temp":     (36.1, 37.2, "C"),
    "SPO2":     (95,   100,  "%"),
    "Sweating": (0,    0.5,  ""),
    "Shivering":(0,    0.5,  ""),
}

# ─── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import keras
        import keras.backend as K

        def r2_metric(y_true, y_pred):
            ss_res = keras.ops.sum(keras.ops.square(y_true - y_pred))
            ss_tot = keras.ops.sum(keras.ops.square(y_true - keras.ops.mean(y_true)))
            return 1 - ss_res / (ss_tot + K.epsilon())

        mdl  = keras.models.load_model(
            os.path.join(MODEL_DIR, MODEL_FILE),
            custom_objects={"r2_metric": r2_metric}
        )
        feat = joblib.load(os.path.join(MODEL_DIR, FEAT_SCALER))
        tgt  = joblib.load(os.path.join(MODEL_DIR, TGT_SCALER))
        return mdl, feat, tgt, True
    except Exception as e:
        st.warning(f"Model not loaded: {e}  —  Running in demo mode.")
        return None, None, None, False

# ─── Prediction ─────────────────────────────────────────────────────────────────
def run_prediction(mdl, feat_sc, tgt_sc, raw):
    inp    = np.array(raw).reshape(1, -1)
    scaled = feat_sc.transform(inp).reshape(1, 1, 8)
    reg_n, cls = mdl.predict(scaled, verbose=0)
    bgl_preds  = tgt_sc.inverse_transform(reg_n)[0]
    return list(bgl_preds), int(np.argmax(cls[0])), float(np.max(cls[0]))

def demo_prediction(bgl):
    if bgl < 70:
        p = bgl + np.array([-3, -5, -7, -10, -13])
    elif bgl > 180:
        p = bgl + np.array([5, 9, 13, 20, 28])
    else:
        p = bgl + np.array([1, 3, 4, 6, 7]) + np.random.normal(0, 2, 5)
    cat = 0 if bgl < 70 else (2 if bgl >= 180 else 1)
    return list(np.clip(p, 30, 500)), cat, 0.93

# ─── Helpers ────────────────────────────────────────────────────────────────────
def cat_meta(cat, bgl):
    if cat == 0 or bgl < 70:
        return "Hypoglycemia",   "hypo",   "badge-hypo",   "rec-box-hypo"
    elif cat == 2 or bgl >= 180:
        return "Hyperglycemia",  "hyper",  "badge-hyper",  "rec-box-hyper"
    else:
        return "Normal Glycemia","normal", "badge-normal", "rec-box-normal"

def delta_html(delta):
    if abs(delta) < 1:   return "No change"
    elif delta > 0:      return f"+{delta:.1f} mg/dL"
    else:                return f"{delta:.1f} mg/dL"

def vital_status(name, val):
    lo, hi, _ = NORMAL_RANGES[name]
    if val < lo:   return "vt-low",  "Below normal"
    elif val > hi: return "vt-high", "Above normal"
    else:          return "vt-ok",   "Normal"

# ─── Main App ───────────────────────────────────────────────────────────────────
def main():
    st.title("🩺 GlucoPredict — TCN-BiGRU Glucose Forecasting")
    st.caption("Multi-horizon blood glucose prediction | Tamil Nadu clinical context")

    mdl, feat_sc, tgt_sc, loaded = load_model()

    st.sidebar.header("Patient Vitals Input")
    bgl       = st.sidebar.number_input("Blood Glucose (mg/dL)", 30.0, 500.0, 120.0, 1.0)
    dbp       = st.sidebar.number_input("DBP (mmHg)",            40.0, 130.0,  75.0, 1.0)
    sbp       = st.sidebar.number_input("SBP (mmHg)",            60.0, 200.0, 110.0, 1.0)
    hr        = st.sidebar.number_input("Heart Rate (bpm)",      30.0, 200.0,  72.0, 1.0)
    temp      = st.sidebar.number_input("Temperature (°C)",      34.0,  42.0,  36.6, 0.1)
    spo2      = st.sidebar.number_input("SpO2 (%)",              70.0, 100.0,  98.0, 0.1)
    sweating  = st.sidebar.slider("Sweating (0–1)",   0.0, 1.0, 0.0, 0.1)
    shivering = st.sidebar.slider("Shivering (0–1)",  0.0, 1.0, 0.0, 0.1)

    raw = [bgl, dbp, sbp, hr, temp, spo2, sweating, shivering]

    if st.sidebar.button(" Predict", use_container_width=True):
        with st.spinner("Running inference..."):
            if loaded:
                preds, cat, conf = run_prediction(mdl, feat_sc, tgt_sc, raw)
            else:
                preds, cat, conf = demo_prediction(bgl)

        label, key, badge_cls, rec_cls = cat_meta(cat, bgl)

        st.subheader(f"Prediction: {label}  |  Confidence: {conf*100:.1f}%")

        # Forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=HORIZONS, y=preds,
            mode="lines+markers",
            name="Predicted BGL",
            line=dict(color="#0d6efd", width=2),
            marker=dict(size=8)
        ))
        fig.add_hline(y=70,  line_dash="dash", line_color="red",   annotation_text="Hypo threshold")
        fig.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="Hyper threshold")
        fig.update_layout(
            title="Blood Glucose Forecast",
            xaxis_title="Minutes ahead",
            yaxis_title="BGL (mg/dL)",
            height=380,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Horizon table
        st.subheader("Horizon-wise Predictions")
        df = pd.DataFrame({
            "Horizon (min)": HORIZONS,
            "Predicted BGL (mg/dL)": [round(p, 1) for p in preds],
            "Change": [delta_html(p - bgl) for p in preds]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Vitals panel
        st.subheader("Vitals Status")
        names = ["BGL","DBP","SBP","HR","Temp","SPO2","Sweating","Shivering"]
        cols  = st.columns(4)
        for i, (name, val) in enumerate(zip(names, raw)):
            cls_, status = vital_status(name, val)
            _, unit = NORMAL_RANGES[name][1], NORMAL_RANGES[name][2]
            with cols[i % 4]:
                st.metric(label=f"{name} ({unit})", value=f"{val}", delta=status)

if __name__ == "__main__":
    main()
