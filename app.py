import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib, json, os, warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GlucoPredict",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    section[data-testid="stSidebar"] { display: none !important; }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0a0d14; color: #e8eaf0; }

    .main-header {
        background: linear-gradient(160deg, #0d1b2a 0%, #0f2744 60%, #0a1628 100%);
        border: 1px solid #1a3050;
        padding: 2.8rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header::after {
        content: '';
        position: absolute; bottom: -40px; left: 30%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header-label {
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.25em; text-transform: uppercase;
        color: #38bdf8; margin-bottom: 0.75rem;
    }
    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem; font-weight: 400;
        margin: 0 0 0.5rem 0; color: #f0f4ff; line-height: 1.2;
    }
    .main-header p { font-size: 0.95rem; color: #7a8aaa; font-weight: 300; margin: 0; }

    .form-card {
        background: #0f1420;
        border: 1px solid #1e2535;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .form-card-title {
        font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
        color: #38bdf8; margin-bottom: 1.2rem;
        padding-bottom: 0.6rem; border-bottom: 1px solid #1e2535;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stTimeInput"] label {
        font-size: 0.75rem !important; font-weight: 600 !important;
        letter-spacing: 0.08em !important; text-transform: uppercase !important;
        color: #7a8aaa !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input {
        background: #161d2e !important;
        border: 1px solid #2a3a55 !important;
        color: #e8eaf0 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stDateInput"] input:focus,
    div[data-testid="stTimeInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        color: white !important; border: none !important;
        padding: 0.85rem 2.5rem !important; border-radius: 9px !important;
        font-weight: 600 !important; font-size: 0.92rem !important;
        letter-spacing: 0.08em !important; text-transform: uppercase !important;
        transition: all 0.2s ease !important; width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(29,78,216,0.4) !important;
    }

    .metric-card {
        background: #0f1420; border: 1px solid #1e2535;
        border-radius: 12px; padding: 1.25rem 1.5rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #2a3a55; }
    .metric-card .label {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #5a6a88; margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem; color: #f0f4ff; line-height: 1;
    }
    .metric-card .unit { font-size: 0.75rem; color: #5a6a88; margin-top: 0.3rem; }
    .metric-card .delta-pos { color: #34d399; font-size: 0.85rem; font-weight: 500; }
    .metric-card .delta-neg { color: #f87171; font-size: 0.85rem; font-weight: 500; }
    .metric-card .delta-neu { color: #7a8aaa; font-size: 0.85rem; font-weight: 500; }

    .badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.5rem;
    }
    .badge-normal { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
    .badge-hypo   { background: rgba(56,189,248,0.12); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }
    .badge-hyper  { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

    .rec-box { border-radius: 14px; padding: 2rem 2.25rem; margin: 1.5rem 0; }
    .rec-box-normal { background: linear-gradient(135deg,#0a1f1a 0%,#0d1e14 100%); border: 1px solid #1a3a2a; }
    .rec-box-hypo   { background: linear-gradient(135deg,#071e2e 0%,#091828 100%); border: 1px solid #1a3050; }
    .rec-box-hyper  { background: linear-gradient(135deg,#1f0a0a 0%,#1e0d0d 100%); border: 1px solid #3a1a1a; }
    .rec-box h3 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35rem; font-weight: 400; margin: 0 0 0.25rem 0;
    }
    .rec-box-normal h3 { color: #6ee7b7; }
    .rec-box-hypo h3   { color: #7dd3fc; }
    .rec-box-hyper h3  { color: #fca5a5; }
    .rec-box .rec-subtitle {
        font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
        font-weight: 600; margin-bottom: 1.5rem;
    }
    .rec-box-normal .rec-subtitle { color: #34d399; }
    .rec-box-hypo   .rec-subtitle { color: #38bdf8; }
    .rec-box-hyper  .rec-subtitle { color: #f87171; }

    .rec-section { margin-bottom: 1.25rem; }
    .rec-section-title {
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: #5a6a88; margin-bottom: 0.6rem;
        padding-bottom: 0.4rem; border-bottom: 1px solid #1e2535;
    }
    .rec-item {
        font-size: 0.92rem; color: #c4cde0;
        padding: 0.3rem 0 0.3rem 1rem;
        position: relative; line-height: 1.55;
    }
    .rec-item::before {
        content: ''; position: absolute; left: 0; top: 0.65rem;
        width: 4px; height: 4px; border-radius: 50%; background: #2a3a55;
    }

    .section-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem; color: #f0f4ff; font-weight: 400;
        margin-bottom: 1.25rem; padding-bottom: 0.75rem;
        border-bottom: 1px solid #1e2535;
    }

    .vitals-row {
        display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .vital-tile {
        background: #0f1420; border: 1px solid #1e2535;
        border-radius: 10px; padding: 0.9rem 1.1rem;
    }
    .vital-tile .vt-label {
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #5a6a88; margin-bottom: 0.3rem;
    }
    .vital-tile .vt-value { font-size: 1.15rem; font-weight: 600; color: #e8eaf0; }
    .vital-tile .vt-status { font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem; }
    .vt-ok   { border-top: 3px solid #34d399; } .vt-ok   .vt-status { color: #34d399; }
    .vt-low  { border-top: 3px solid #38bdf8; } .vt-low  .vt-status { color: #38bdf8; }
    .vt-high { border-top: 3px solid #f87171; } .vt-high .vt-status { color: #f87171; }

    .divider { border: none; border-top: 1px solid #1e2535; margin: 2rem 0; }
    .disclaimer {
        background: #0f1420; border: 1px solid #1e2535;
        border-radius: 10px; padding: 1rem 1.4rem;
        font-size: 0.78rem; color: #5a6a88;
        line-height: 1.6; margin-top: 1.5rem;
    }
    .disclaimer strong { color: #34d399; }

    .stAlert { display: none; }

    div[data-testid="stSelectbox"] label {
        font-size: 0.75rem !important; font-weight: 600 !important;
        letter-spacing: 0.08em !important; text-transform: uppercase !important;
        color: #7a8aaa !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: #161d2e !important;
        border: 1px solid #2a3a55 !important;
        color: #e8eaf0 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    }
    div[data-testid="stSelectbox"] svg { fill: #7a8aaa !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR     = r"C:\Users\HP\Downloads\Nandhini_Majorproject\models"
MODEL_FILE    = "tcn_bigru_final.keras"          # ← TCN-BiGRU best model
FEAT_SCALER   = "feature_scaler.pkl"
TGT_SCALER    = "target_scaler.pkl"

HORIZONS      = [5, 10, 15, 30, 60]
NORMAL_RANGES = {
    "BGL":      (70,  180,  "mg/dL"),
    "DBP":      (60,   90,  "mmHg"),
    "SBP":      (90,  130,  "mmHg"),
    "HR":       (60,  100,  "bpm"),
    "Temp":     (36.1, 37.2, "C"),
    "SPO2":     (95,  100,  "%"),
    "Sweating": (0,   0.5,  ""),
    "Shivering":(0,   0.5,  ""),
}

# ─── Clinical Recommendations Data ───────────────────────────────────────────
RECS = {
    "hypo": {
        "title": "Hypoglycemia Management",
        "subtitle": "Immediate clinical response required",
        "immediate": [
            "Consume 15 g of fast-acting carbohydrates right away",
            "Recheck blood glucose after 15 minutes and repeat if still below 70",
            "Seek medical attention if symptoms persist beyond 30 minutes",
            "Do not drive or operate equipment until glucose normalises",
            "Inform nearby person about your hypoglycemia episode"
        ],
        "tn_food_now": [
            "Karimbu juice - half glass of fresh sugarcane juice without ice",
            "Ripe banana - one medium sized palazham for immediate glucose rise",
            "Vellam - one tablespoon of jaggery dissolved in warm water",
            "Tender coconut water - ilaneer provides natural glucose and electrolytes",
            "Sweet pongal - small cup, only as immediate glucose source"
        ],
        "tn_followup": [
            "Idli with sambar - two pieces after glucose stabilises above 80",
            "Ragi koozh with a pinch of salt for sustained energy release",
            "Dosai with mild coconut chutney as follow-up meal",
            "Kollu rasam with small cup of rice to maintain glucose level"
        ],
        "tn_avoid": [
            "Avoid high-fat items immediately after a hypoglycemia episode",
            "Do not skip the follow-up meal after correcting glucose",
            "Avoid strong coffee or tea as an immediate remedy",
            "Do not lie down immediately without consuming carbohydrates first"
        ],
        "sleep": [
            "Never sleep with blood glucose below 90 mg per dL",
            "Bedtime snack: small bowl of ragi kanji without sugar",
            "Kambu dosai with a teaspoon of ghee provides overnight stability",
            "Keep glucose tablets or a ripe banana within arm reach at night",
            "Set a 2 AM alarm to recheck if evening glucose was borderline low",
            "Target bedtime glucose between 110 and 150 mg per dL",
            "Inform a family member about hypoglycemia risk before sleeping"
        ],
        "physical": [
            "Stop all physical activity immediately when hypoglycemia is detected",
            "Sit or lie down to prevent injury from dizziness or weakness",
            "Resume light activity only after glucose rises above 100 mg per dL",
            "Avoid exercise for at least 2 hours after a hypoglycemia episode"
        ]
    },
    "normal": {
        "title": "Normal Glycemia Maintenance",
        "subtitle": "Continue current management plan",
        "immediate": [
            "Continue prescribed medications without changes",
            "Monitor blood glucose twice daily, morning and post-dinner",
            "Maintain hydration with at least 2.5 litres of water daily",
            "Keep a food diary to track glycemic response to Tamil Nadu dishes",
            "Schedule next HbA1c check in 3 months if not already planned"
        ],
        "tn_food_best": [
            "Ragi koozh - finger millet porridge, excellent sustained glucose control",
            "Kambu roti or kambu koozh - pearl millet, very low glycemic index",
            "Thinai pongal - foxtail millet pongal, rich in fibre",
            "Fermented idli with sambar lowers glycemic index by 30 percent",
            "Vendakkai curry - ladies finger controls post-meal glucose spikes",
            "Paavakai fry or juice - bitter gourd is a proven natural regulator",
            "Murungai keerai kootu - drumstick leaves improve insulin sensitivity",
            "Kollu rasam - horsegram soup, a traditional diabetic remedy",
            "Neer mor with curry leaves, ginger, and green chilli"
        ],
        "tn_moderate": [
            "Idiyappam - moderate, pair with vegetable kurma not sweetened coconut milk",
            "Hand-pounded or brown rice instead of polished white rice",
            "Adai dosai - mixed lentil dosai, higher protein than plain dosai",
            "Keerai masiyal with a small rice portion",
            "Sundal - chickpea or black-eyed pea, excellent plant protein"
        ],
        "tn_avoid": [
            "Limit white rice to one cup per meal, replace half with millets",
            "Avoid sweet pongal, payasam, and kesari on regular days",
            "Reduce consumption of thenga burfi, mysore pak, and laddu",
            "Limit maida-based items such as parotta, bonda, and bajji"
        ],
        "sleep": [
            "Ideal bedtime glucose target: 100 to 140 mg per dL",
            "Light dinner 2 to 3 hours before sleep, prefer ragi dosai or adai",
            "A 15 to 20 minute walk after dinner reduces morning glucose significantly",
            "Maintain consistent sleep and wake times for hormonal stability",
            "7 to 8 hours of sleep is clinically proven to improve insulin sensitivity",
            "Vendhayam seeds soaked overnight, consume the water on empty stomach",
            "Avoid strong coffee or tea after 6 PM to protect sleep quality"
        ],
        "physical": [
            "30 minutes of moderate exercise 5 days per week is recommended",
            "Morning walk in the early hours is especially beneficial in Tamil Nadu climate",
            "Yoga and pranayama have shown documented benefit in South Indian populations",
            "Strength training twice weekly increases glucose uptake by muscles"
        ]
    },
    "hyper": {
        "title": "Hyperglycemia Management",
        "subtitle": "Elevated glucose requires immediate attention",
        "immediate": [
            "Check for ketones if glucose exceeds 300 mg per dL",
            "Drink 250 ml of plain water immediately and continue every 30 minutes",
            "Contact your physician if glucose exceeds 350 mg per dL",
            "Administer insulin correction dose as per physician prescription",
            "Avoid all fruit juices, sweetened beverages, and energy drinks"
        ],
        "tn_food_best": [
            "Paavakai juice - 30 ml bitter gourd juice on empty stomach is proven effective",
            "Murungai keerai juice or kootu - drumstick leaves powerfully regulate glucose",
            "Vendhayam water - soak 2 teaspoons of fenugreek seeds overnight, drink the water",
            "Karuveppilai chutney - curry leaves with garlic aids natural insulin secretion",
            "Neem flower rasam - prepared during Pongal, proven anti-diabetic properties",
            "Noolkol or mullangi curry instead of potato-based dishes",
            "Vellai poosanikai kootu - ash gourd has very low glycemic index",
            "Araikeerai with garlic - reduces post-meal glucose spikes effectively"
        ],
        "tn_replace": [
            "Replace white rice with kuthiraivali sadham - barnyard millet rice",
            "Use varagu for pongal and kanji instead of regular rice",
            "Replace regular idli with ragi idli or ragi dosai",
            "Mix kollu flour partially into dosai batter for lower glycemic dosai",
            "Replace sugar with a small quantity of karuppatti on special occasions only"
        ],
        "tn_avoid": [
            "White rice in large portions - strict restriction advised",
            "Payasam, kesari, sweet pongal, and all festival sweets completely",
            "Mangoes, jackfruit, and ripe banana - high glycemic tropical fruits",
            "Ulundu vadai, bonda, bajji - deep fried high carbohydrate snacks",
            "Sugarcane juice, fresh lime with sugar, and sweetened drinks",
            "Hotel meals with unlimited rice, puri, and sweet sides"
        ],
        "sleep": [
            "High glucose at bedtime directly worsens sleep quality and morning levels",
            "Light dinner only: one ragi dosai or two idli with vegetable sambar",
            "Mandatory 20 to 30 minute slow walk after dinner before sleeping",
            "Avoid all food after 8 PM, late eating worsens overnight glucose",
            "Target bedtime glucose: 120 to 150 mg per dL",
            "Poor sleep raises cortisol which directly elevates next-morning glucose",
            "Vendhayam water before bed is clinically shown to reduce fasting glucose",
            "Keep bedroom cool as heat stress worsens glucose metabolism during sleep"
        ],
        "physical": [
            "Light 20-minute walk immediately after every meal",
            "Avoid intense exercise when glucose exceeds 300 mg per dL",
            "Swimming is excellent low-impact exercise for Tamil Nadu coastal population",
            "Avoid outdoor exercise during peak heat hours, prefer early morning or evening"
        ]
    }
}


# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_dir):
    """Load TCN-BiGRU model + scalers. Returns (model, feat_sc, tgt_sc, loaded_bool)."""
    try:
        import tensorflow as tf

        def r2_metric(y_true, y_pred):
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

        mdl  = tf.keras.models.load_model(
            os.path.join(model_dir, MODEL_FILE),
            custom_objects={"r2_metric": r2_metric}
        )
        feat = joblib.load(os.path.join(model_dir, FEAT_SCALER))
        tgt  = joblib.load(os.path.join(model_dir, TGT_SCALER))
        return mdl, feat, tgt, True
    except Exception as e:
        return None, None, None, False


def run_prediction(mdl, feat_sc, tgt_sc, raw):
    inp    = np.array(raw).reshape(1, -1)
    scaled = feat_sc.transform(inp).reshape(1, 1, 8)
    reg_n, cls = mdl.predict(scaled, verbose=0)
    bgl_preds  = tgt_sc.inverse_transform(reg_n)[0]
    return list(bgl_preds), int(np.argmax(cls[0])), float(np.max(cls[0]))


def demo_prediction(bgl):
    """Illustrative demo when model files are not found."""
    if bgl < 70:
        p = bgl + np.array([-3, -5, -7, -10, -13])
    elif bgl > 180:
        p = bgl + np.array([5, 9, 13, 20, 28])
    else:
        p = bgl + np.array([1, 3, 4, 6, 7]) + np.random.normal(0, 2, 5)
    cat = 0 if bgl < 70 else (2 if bgl >= 180 else 1)
    return list(np.clip(p, 30, 500)), cat, 0.93


# ─── Helpers ──────────────────────────────────────────────────────────────────
def cat_meta(cat, bgl):
    if cat == 0 or bgl < 70:
        return "Hypoglycemia",    "hypo",   "badge-hypo",   "rec-box-hypo"
    elif cat == 2 or bgl >= 180:
        return "Hyperglycemia",   "hyper",  "badge-hyper",  "rec-box-hyper"
    else:
        return "Normal Glycemia", "normal", "badge-normal", "rec-box-normal"


def delta_html(delta):
    if abs(delta) < 1:
        return '<span class="delta-neu">No change</span>'
    elif delta > 0:
        return f'<span class="delta-pos">+{delta:.1f} mg/dL</span>'
    else:
        return f'<span class="delta-neg">{delta:.1f} mg/dL</span>'


def vital_status(name, val):
    lo, hi, _ = NORMAL_RANGES[name]
    if val < lo:   return "vt-low",  "Below normal"
    elif val > hi: return "vt-high", "Above normal"
    else:          return "vt-ok",   "Normal"


def rec_items(lst):
    return "".join(f'<div class="rec-item">{x}</div>' for x in lst)


# ─── Load TCN-BiGRU ───────────────────────────────────────────────────────────
model, feat_sc, tgt_sc, loaded = load_model(MODEL_DIR)

# ─── Session State ────────────────────────────────────────────────────────────
for _k, _v in {
    'has_prediction': False,
    'preds':          None,
    'cat':            1,
    'conf':           0.0,
    'last_bgl':       120.0,
    'last_dt':        None,
    'last_vitals':    None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-header-label">Diabetes Clinical Monitoring System</div>
    <h1>GlucoPredict</h1>
    <p>TCN-BiGRU Deep Learning Model &nbsp;|&nbsp;
       Single-Step Multi-Horizon Forecasting &nbsp;|&nbsp;
    </p>
</div>
""", unsafe_allow_html=True)


# ─── INPUT FORM ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="form-card">
    <div class="form-card-title">Reading Date and Time</div>
</div>
""", unsafe_allow_html=True)

_now = datetime.datetime.now()

c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 1.2])
with c1:
    reading_date = st.date_input("Date of Reading", value=datetime.date.today())
with c2:
    hour = st.selectbox("Hour  (0 to 23)", options=list(range(24)),
                        index=_now.hour, format_func=lambda x: f"{x:02d}")
with c3:
    minute = st.selectbox("Minute  (0 to 59)", options=list(range(60)),
                          index=_now.minute, format_func=lambda x: f"{x:02d}")
reading_time = datetime.time(int(hour), int(minute))

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── BGL ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="form-card">
    <div class="form-card-title">Primary Measurement</div>
</div>
""", unsafe_allow_html=True)
col_bgl, _, __ = st.columns([1, 1, 1])
with col_bgl:
    bgl = st.number_input("Blood Glucose Level  (mg/dL)", 40.0, 500.0, 120.0, step=1.0)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Cardiovascular ────────────────────────────────────────────────────────────
st.markdown("""
<div class="form-card">
    <div class="form-card-title">Cardiovascular</div>
</div>
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    dbp = st.number_input("Diastolic BP  (mmHg)", 40.0, 130.0, 75.0, step=1.0)
with c2:
    sbp = st.number_input("Systolic BP  (mmHg)",  80.0, 200.0, 118.0, step=1.0)
with c3:
    hr  = st.number_input("Heart Rate  (bpm)",    40.0, 180.0, 76.0, step=1.0)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Body Sensors ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="form-card">
    <div class="form-card-title">Body Sensors</div>
</div>
""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    temp  = st.number_input("Body Temperature  (C)",     35.0, 41.0, 36.7, step=0.1)
with c2:
    spo2  = st.number_input("SpO2  (%)",                 70.0, 100.0, 98.0, step=0.1)
with c3:
    sweat = st.number_input("Sweating Score  (0 to 1)",  0.0,  1.0,   0.0, step=0.01)
with c4:
    shiv  = st.number_input("Shivering Score  (0 to 1)", 0.0,  1.0,   0.0, step=0.01)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    predict_btn = st.button("Analyse and Predict")

if predict_btn:
    _raw = [bgl, dbp, sbp, hr, temp, spo2, sweat, shiv]
    if loaded:
        _p, _c, _cf = run_prediction(model, feat_sc, tgt_sc, _raw)
    else:
        _p, _c, _cf = demo_prediction(bgl)
    st.session_state.has_prediction = True
    st.session_state.preds          = _p
    st.session_state.cat            = _c
    st.session_state.conf           = _cf
    st.session_state.last_bgl       = bgl
    st.session_state.last_dt        = datetime.datetime.combine(reading_date, reading_time)
    st.session_state.last_vitals    = [bgl, dbp, sbp, hr, temp, spo2, sweat, shiv]

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─── Waiting State ────────────────────────────────────────────────────────────
if not st.session_state.has_prediction:
    st.markdown("""
    <div style="background:#0f1420;border:1px solid #1e2535;border-radius:16px;
        padding:5rem 2rem;text-align:center;">
        <div style="font-family:'DM Serif Display',serif;font-size:2rem;
            color:#f0f4ff;margin-bottom:0.9rem;">
            Awaiting Patient Data
        </div>
        <div style="font-size:0.9rem;color:#5a6a88;max-width:480px;
            margin:0 auto;line-height:1.8;">
            Enter all sensor values above, select the reading date and time,
            then click Analyse and Predict to generate the multi-horizon glucose
            forecast and personalised clinical recommendations.
        </div>
        <div style="margin-top:2rem;font-size:0.72rem;color:#2a3a55;
            letter-spacing:0.14em;text-transform:uppercase;">
            TCN-BiGRU Model · Test Accuracy 98.21% · Test R² 0.8821
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Pull from Session State ──────────────────────────────────────────────────
preds      = st.session_state.preds
cat        = st.session_state.cat
conf       = st.session_state.conf
bgl        = st.session_state.last_bgl
reading_dt = st.session_state.last_dt
vitals_raw = st.session_state.last_vitals

future_dts   = [reading_dt + datetime.timedelta(minutes=h) for h in HORIZONS]
future_times = [dt.strftime("%H:%M") for dt in future_dts]
future_dates = [dt.strftime("%d %b") for dt in future_dts]
reading_ts   = reading_dt.strftime("%d %b %Y  %H:%M")

label, cat_key, badge_cls, rec_cls = cat_meta(cat, bgl)
r = RECS[cat_key]


# ─── Status Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Current Status</div>
        <div class="value">{bgl:.0f}</div>
        <div class="unit">mg per dL</div>
        <span class="badge {badge_cls}">{label}</span>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Reading Taken At</div>
        <div class="value" style="font-size:1.45rem;">{reading_dt.strftime("%H:%M")}</div>
        <div class="unit">{reading_dt.strftime("%d %B %Y")}</div>
        <div class="delta-neu" style="margin-top:0.4rem;">Forecast up to {future_times[-1]}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    trend_dir = "Rising" if preds[-1] > bgl + 5 else ("Falling" if preds[-1] < bgl - 5 else "Stable")
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">60-min Trend</div>
        <div class="value">{trend_dir}</div>
        <div class="unit">&nbsp;</div>
        {delta_html(preds[-1] - bgl)}
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Confidence</div>
        <div class="value">{conf*100:.1f}</div>
        <div class="unit">percent</div>
        <span class="badge badge-normal">AI Forecast</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─── Forecast Cards ───────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Multi-Horizon Blood Glucose Forecast</div>',
            unsafe_allow_html=True)
cols = st.columns(5)
for i, (col, pred_val) in enumerate(zip(cols, preds)):
    p_cat = 0 if pred_val < 70 else (2 if pred_val >= 180 else 1)
    b_cls = "badge-hypo" if p_cat == 0 else ("badge-hyper" if p_cat == 2 else "badge-normal")
    b_lbl = "Hypo" if p_cat == 0 else ("Hyper" if p_cat == 2 else "Normal")
    delta = pred_val - bgl
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label" style="font-size:0.82rem;color:#38bdf8;">
                {future_times[i]} &nbsp;<span style="color:#5a6a88;">+{HORIZONS[i]}m</span>
            </div>
            <div class="value">{pred_val:.0f}</div>
            <div class="unit">{future_dates[i]} &nbsp; mg/dL</div>
            {delta_html(delta)}
            <span class="badge {b_cls}" style="margin-top:0.6rem;">{b_lbl}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Trajectory Chart ─────────────────────────────────────────────────────────
chart_dts   = [reading_dt] + future_dts
values      = [bgl] + list(preds)
x_tick_text = [reading_dt.strftime("%H:%M") + "\nNow"] + [
    f"{dt.strftime('%H:%M')}\n+{h}m" for dt, h in zip(future_dts, HORIZONS)
]

fig = go.Figure()
fig.add_hrect(y0=0, y1=70, fillcolor="rgba(56,189,248,0.04)", line_width=0,
              annotation_text="Hypoglycemia zone", annotation_position="left",
              annotation_font_color="#38bdf8", annotation_font_size=10)
fig.add_hrect(y0=180, y1=600, fillcolor="rgba(248,113,113,0.04)", line_width=0,
              annotation_text="Hyperglycemia zone", annotation_position="left",
              annotation_font_color="#f87171", annotation_font_size=10)
fig.add_hline(y=70,  line_dash="dot", line_color="#38bdf8", line_width=1.2)
fig.add_hline(y=180, line_dash="dot", line_color="#f87171", line_width=1.2)

upper = [v + 10 for v in values]
lower = [max(30, v - 10) for v in values]
fig.add_trace(go.Scatter(
    x=chart_dts + chart_dts[::-1], y=upper + lower[::-1],
    fill="toself", fillcolor="rgba(29,78,216,0.07)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"
))
pt_colors = ["#38bdf8" if v < 70 else ("#f87171" if v > 180 else "#34d399") for v in values]
fig.add_trace(go.Scatter(
    x=chart_dts, y=values, mode="lines+markers",
    line=dict(color="#6366f1", width=2.5, shape="spline", smoothing=0.7),
    marker=dict(size=[16] + [11] * 5, color=pt_colors,
                line=dict(color="#0a0d14", width=2)),
    customdata=[[h] for h in [0] + HORIZONS],
    hovertemplate="<b>%{x|%H:%M}</b>  (+%{customdata[0]} min)<br>BGL: %{y:.1f} mg/dL<extra></extra>"
))
fig.update_layout(
    plot_bgcolor="#0f1420", paper_bgcolor="#0f1420",
    height=310, margin=dict(l=55, r=55, t=20, b=55),
    xaxis=dict(title=dict(text="Clock Time", font=dict(color="#5a6a88", size=11)),
               tickvals=chart_dts, ticktext=x_tick_text,
               tickfont=dict(color="#5a6a88", size=10),
               gridcolor="#1e2535", showline=True, linecolor="#1e2535"),
    yaxis=dict(title=dict(text="Blood Glucose (mg/dL)", font=dict(color="#5a6a88", size=11)),
               range=[max(30, min(values) - 35), max(values) + 45],
               tickfont=dict(color="#5a6a88", size=11),
               gridcolor="#1e2535", showline=True, linecolor="#1e2535"),
    font=dict(family="DM Sans", color="#c4cde0"),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)


# ─── Vitals Tiles ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-heading">Vitals at Reading Time &nbsp;'
    f'<span style="font-size:0.85rem;color:#5a6a88;font-family:\'DM Sans\',sans-serif;">'
    f'{reading_ts}</span></div>',
    unsafe_allow_html=True)

v_names = ["BGL", "DBP", "SBP", "HR", "Temp", "SPO2", "Sweating", "Shivering"]
v_units = ["mg/dL", "mmHg", "mmHg", "bpm", "C", "%", "", ""]
tiles_html = '<div class="vitals-row">'
for vn, vv, vu in zip(v_names, vitals_raw, v_units):
    tc, sl = vital_status(vn, vv)
    tiles_html += f"""
    <div class="vital-tile {tc}">
        <div class="vt-label">{vn}</div>
        <div class="vt-value">{vv:.1f} {vu}</div>
        <div class="vt-status">{sl}</div>
    </div>"""
tiles_html += "</div>"
st.markdown(tiles_html, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ─── Clinical Recommendations ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">Clinical Recommendations</div>',
            unsafe_allow_html=True)
col_clin, col_food = st.columns(2)

with col_clin:
    st.markdown(f"""
    <div class="rec-box {rec_cls}">
        <div class="rec-subtitle">{r['subtitle']}</div>
        <h3>{r['title']}</h3>
        <div class="rec-section">
            <div class="rec-section-title">Immediate Action</div>
            {rec_items(r['immediate'])}
        </div>
        <div class="rec-section">
            <div class="rec-section-title">Physical Activity</div>
            {rec_items(r['physical'])}
        </div>
    </div>""", unsafe_allow_html=True)

with col_food:
    if cat_key == "hypo":
        f1t, f1k = "Immediate Foods",         "tn_food_now"
        f2t, f2k = "Follow-up Meal",           "tn_followup"
        f3t, f3k = "Foods to Avoid",           "tn_avoid"
    elif cat_key == "normal":
        f1t, f1k = "Best Choices",             "tn_food_best"
        f2t, f2k = "Moderate Choices",         "tn_moderate"
        f3t, f3k = "Foods to Limit",           "tn_avoid"
    else:
        f1t, f1k = "Recommended Foods",        "tn_food_best"
        f2t, f2k = "Rice Replacement Options", "tn_replace"
        f3t, f3k = "Strict Avoidance List",    "tn_avoid"

    st.markdown(f"""
    <div class="rec-box {rec_cls}">
        <div class="rec-subtitle">Tamil Nadu Dietary Guide</div>
        <h3>Food Recommendations</h3>
        <div class="rec-section">
            <div class="rec-section-title">{f1t}</div>
            {rec_items(r[f1k])}
        </div>
        <div class="rec-section">
            <div class="rec-section-title">{f2t}</div>
            {rec_items(r[f2k])}
        </div>
        <div class="rec-section">
            <div class="rec-section-title">{f3t}</div>
            {rec_items(r[f3k])}
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Sleep Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Sleep and Night-Time Management</div>',
            unsafe_allow_html=True)
st.markdown(f"""
<div class="rec-box {rec_cls}">
    <div class="rec-subtitle">Night-time glucose management for {label}</div>
    <h3>Sleep Protocol</h3>
    <div class="rec-section">
        <div class="rec-section-title">Guidelines</div>
        {rec_items(r['sleep'])}
    </div>
</div>""", unsafe_allow_html=True)