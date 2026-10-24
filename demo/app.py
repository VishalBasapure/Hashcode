# ============================================================
#  app.py — Karnataka Crime Risk Prediction Dashboard
#  Integrates Model A (DCVI) + Model B (Bengaluru hotspot)
#
#  Install:
#    pip install streamlit pandas numpy scikit-learn folium
#               streamlit-folium plotly joblib
#
#  Run:
#    streamlit run app.py
#
#  Files needed:
#    data/output_data/dcvi_2024_predictions.csv
#    data/processed/karnataka_model_ready.csv
#    models/dcvi_model.pkl
#    models/dcvi_scalers.pkl
#    models/district_map.pkl
#    outputs-maps/karnataka_crime_2024_fixed.html   (run kar_map.py first)
#    police_data/kml_extracted_1.csv   (police outposts)
#    police_data/kml_extracted_2.csv   (police stations)
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import warnings, os, json
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Karnataka Crime Risk Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 16px; }
.stApp { background: #FFFFFF; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── Global text ── */
p, li, span, div, label { font-size: 16px !important; }
h1 { font-size: 2.2rem !important; font-weight: 700 !important; color: #111827 !important; }
h2 { font-size: 1.7rem !important; font-weight: 700 !important; color: #111827 !important; }
h3 { font-size: 1.4rem !important; font-weight: 600 !important; color: #111827 !important; }

/* ── Selectbox / radio labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label { font-size: 15px !important; font-weight: 600; color: #374151 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] { font-size: 15px !important; }

/* ── Metric cards — clean, flat, light ── */
[data-testid="metric-container"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px !important;
    transition: box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    font-size: 12px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #6B7280 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 30px !important;
    font-weight: 700;
    color: #111827 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 14px !important;
}

/* ── Risk badge — simple rounded chip ── */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 14px !important;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-HIGH   { background: #FEE2E2; color: #B91C1C; }
.badge-MEDIUM { background: #FEF3C7; color: #92400E; }
.badge-LOW    { background: #D1FAE5; color: #065F46; }

/* ── Section headings ── */
.sec-head {
    font-size: 13px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6B7280 !important;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
}

/* ── Info box ── */
.info-box {
    background: #EFF6FF;
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 14px !important;
    color: #1E3A8A !important;
    margin-bottom: 14px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #E5E7EB !important; border-radius: 8px; }
[data-testid="stDataFrame"] thead th {
    background: #F9FAFB !important;
    color: #374151 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-size: 12px !important;
}

/* ── Sidebar — clean light panel ── */
[data-testid="stSidebar"] { background: #F9FAFB; border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] * { color: #1F2937 !important; font-size: 16px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-size: 16px !important; }
[data-testid="stSidebar"] hr { border-color: #E5E7EB !important; }

/* ════════════════════════════════════════════════════════════
   PAGE HEADER — single source of truth for all 4 pages
   ════════════════════════════════════════════════════════════ */
.page-head-table {
    margin-bottom: 16px;
}
.page-head-tag-row { margin-bottom: 6px; }
.page-tag {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    padding: 4px 12px;
    border-radius: 6px;
    line-height: 1.4 !important;
}
.page-title {
    display: block !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.3 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ════════════════════════════════════════════════════════════
   DISTRICT SUMMARY CARD — Police Resource Planning page
   Simple white card, soft shadow, no heavy borders/ornamentation
   ════════════════════════════════════════════════════════════ */
.dossier-wrap {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 0;
    margin-bottom: 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    overflow: hidden;
}
.dossier-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 22px 28px 18px 28px;
    border-bottom: 1px solid #F3F4F6;
}
.dossier-title {
    font-size: 24px !important;
    font-weight: 700;
    color: #111827 !important;
    margin: 0;
}
.dossier-subtitle {
    font-size: 13px !important;
    color: #6B7280 !important;
    margin-top: 4px;
}
.dossier-seal {
    font-weight: 700;
    font-size: 13px !important;
    letter-spacing: 0.04em;
    padding: 7px 16px;
    border-radius: 20px;
    white-space: nowrap;
}
.seal-HIGH   { background: #FEE2E2; color: #B91C1C; }
.seal-MEDIUM { background: #FEF3C7; color: #92400E; }
.seal-LOW    { background: #D1FAE5; color: #065F46; }

.ledger-row {
    display: flex;
    border-bottom: 1px solid #F3F4F6;
    padding: 16px 28px;
    transition: background 0.15s ease;
}
.ledger-row:hover { background: #FAFAFA; }
.ledger-row:last-child { border-bottom: none; }
.ledger-label {
    font-size: 13px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9CA3AF !important;
    width: 220px;
    flex-shrink: 0;
    padding-top: 3px;
}
.ledger-value {
    font-size: 22px !important;
    font-weight: 700;
    color: #111827 !important;
}
.ledger-value-sub {
    font-size: 12px !important;
    color: #9CA3AF !important;
    margin-top: 2px;
}

.dossier-divider {
    font-size: 12px !important;
    font-weight: 700;
    color: #9CA3AF !important;
    text-align: left;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 28px 0 16px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
}

.directive-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.directive-num {
    font-size: 11px !important;
    font-weight: 700;
    color: #9CA3AF !important;
    letter-spacing: 0.06em;
}
.directive-title {
    font-size: 16px !important;
    font-weight: 700;
    color: #111827 !important;
    margin: 3px 0 7px 0;
}
.directive-body {
    font-size: 14px !important;
    color: #4B5563 !important;
    line-height: 1.7;
}

.priority-strip {
    font-size: 13px !important;
    color: #4B5563 !important;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 14px;
}

/* ════════════════════════════════════════════════════════════
   ANIMATED ODOMETER — rolling-digit counters for district stats
   ════════════════════════════════════════════════════════════ */
.odometer-wrap {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}
.odo-card {
    flex: 1;
    min-width: 150px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
}
.odo-label {
    font-size: 12px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9CA3AF !important;
    margin-bottom: 6px;
}
.odo-value {
    font-size: 30px !important;
    font-weight: 700;
    color: #111827 !important;
    font-variant-numeric: tabular-nums;
}
.odo-sub {
    font-size: 11px !important;
    color: #9CA3AF !important;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    pred     = pd.read_csv("data/output_data/dcvi_2024_predictions.csv")
    model_df = pd.read_csv("data/processed/karnataka_model_ready.csv")
    dist_names   = sorted(pred['district'].unique())
    dist_num_map = dict(enumerate(dist_names))

    pred['change_pct']       = ((pred['pred_crimes_2024'] - pred['actual_2023'])
                                 / pred['actual_2023'] * 100).round(1)
    pred['pred_crimes_2025'] = (pred['pred_crimes_2024']
                                 * (1 + pred['change_pct'] / 100)).round(0)

    p33 = np.percentile(pred['dcvi_score'], 33)
    p66 = np.percentile(pred['dcvi_score'], 66)
    pred['risk'] = pd.cut(pred['dcvi_score'], bins=[0, p33, p66, 100],
                          labels=['LOW','MEDIUM','HIGH'],
                          include_lowest=True).astype(str)

    d23  = model_df[model_df['year'] == 2023].copy()
    d23['district_name'] = d23['district'].map(dist_num_map)
    cats = d23[['district_name','crime_against_women','crime_against_children',
                'violent_crime','cyber_crime','other_crime']].set_index('district_name')
    pred = pred.merge(cats, left_on='district', right_index=True, how='left')

    hist = {}
    for dnum, dn in dist_num_map.items():
        rows = model_df[model_df['district']==dnum].sort_values('year')
        hist[dn] = rows[['year','total_crime']].set_index('year')['total_crime'].to_dict()

    return pred, hist, p33, p66


def load_metrics():
    default = {'r2': 0.776, 'mae': 205}
    if not os.path.exists('data/output_data/dcvi_metrics.json'):
        return default
    try:
        with open('data/output_data/dcvi_metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return {
            'r2': float(metrics.get('r2', default['r2'])),
            'mae': float(metrics.get('mae', default['mae']))
        }
    except Exception:
        return default

pred_df, hist_data, P33, P66 = load_data()
metrics = load_metrics()

COORDS = {
    "bagalkot":[16.18,75.69],"ballari":[15.15,76.93],
    "belagavi city":[15.85,74.50],"belagavi district":[15.70,74.90],
    "bengaluru city":[12.97,77.59],"bengaluru district":[13.10,77.45],
    "bidar":[17.91,77.52],"chamarajnagar":[11.92,76.94],
    "chikkaballapura":[13.44,77.73],"chikkamagaluru":[13.32,75.77],
    "chitradurga":[14.23,76.40],"dakshina kannada":[12.84,74.86],
    "davanagere":[14.46,75.92],"dharwad":[15.46,75.01],
    "gadag":[15.43,75.63],"hassan":[13.00,76.10],
    "haveri":[14.80,75.40],"hubballi dharwad city":[15.36,75.12],
    "k.g.f.":[12.97,78.27],"k.railways":[15.31,75.71],
    "kalaburgi":[17.33,76.83],"kalaburgi city":[17.40,76.90],
    "kodagu":[12.42,75.74],"kolar":[13.14,78.13],
    "koppal":[15.35,76.15],"mandya":[12.52,76.90],
    "mangaluru city":[12.91,74.85],"mysuru city":[12.30,76.65],
    "mysuru district":[12.20,76.40],"raichur":[16.20,77.35],
    "ramanagara":[12.72,77.28],"shimoga":[13.93,75.57],
    "tumakuru":[13.34,77.10],"udupi":[13.34,74.75],
    "uttara kannada":[14.79,74.13],"vijayanagara":[15.32,76.46],
    "vijayapura":[16.83,75.71],"yadgiri":[16.77,77.14],
}

RISK_CLR = {'HIGH':'#DC2626','MEDIUM':'#D97706','LOW':'#16A34A'}


def render_odometer_card(district, case_id, risk, dcvi, pred_crimes, actual_2023,
                          stations, outposts, coverage, gap, gap_clr):
    """Render the district summary card with JS-animated rolling-digit counters.
    Each numeric value counts up/down from its previous value whenever the
    selected district changes, using requestAnimationFrame for a smooth feel."""
    risk_bg = {'HIGH': '#FEE2E2', 'MEDIUM': '#FEF3C7', 'LOW': '#D1FAE5'}.get(risk, '#F3F4F6')
    risk_fg = {'HIGH': '#B91C1C', 'MEDIUM': '#92400E', 'LOW': '#065F46'}.get(risk, '#374151')

    html = f"""
    <div id="odo-root" style="font-family:'Inter','Segoe UI',sans-serif;">
      <div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.04);overflow:hidden;">

        <div style="display:flex;justify-content:space-between;align-items:flex-start;
                    padding:22px 28px 18px 28px;border-bottom:1px solid #F3F4F6;">
          <div>
            <p style="font-size:24px;font-weight:700;color:#111827;margin:0;">{district.title()}</p>
            <p style="font-size:13px;color:#6B7280;margin-top:4px;">
              CASE REF {case_id} &middot; REVIEW YEAR 2024 &middot; SOURCE: NCRB KARNATAKA</p>
          </div>
          <div style="background:{risk_bg};color:{risk_fg};font-weight:700;font-size:13px;
                      letter-spacing:0.04em;padding:7px 16px;border-radius:20px;white-space:nowrap;">
            {risk} RISK
          </div>
        </div>

        <div class="ledger-row" style="display:flex;border-bottom:1px solid #F3F4F6;padding:16px 28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
                      color:#9CA3AF;width:220px;flex-shrink:0;padding-top:3px;">Vulnerability Index</div>
          <div>
            <div style="font-size:22px;font-weight:700;color:#111827;">
              <span class="odo" data-target="{dcvi}" data-decimals="1">0</span> / 100
            </div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">DCVI composite score</div>
          </div>
        </div>

        <div class="ledger-row" style="display:flex;border-bottom:1px solid #F3F4F6;padding:16px 28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
                      color:#9CA3AF;width:220px;flex-shrink:0;padding-top:3px;">Crime Load</div>
          <div>
            <div style="font-size:22px;font-weight:700;color:#111827;">
              <span class="odo" data-target="{pred_crimes}" data-decimals="0">0</span>
              <span style="font-size:14px;font-weight:400;color:#9CA3AF;">predicted 2024</span>
            </div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">
              <span class="odo" data-target="{actual_2023}" data-decimals="0">0</span> recorded in 2023</div>
          </div>
        </div>

        <div class="ledger-row" style="display:flex;border-bottom:1px solid #F3F4F6;padding:16px 28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
                      color:#9CA3AF;width:220px;flex-shrink:0;padding-top:3px;">Deployed Infrastructure</div>
          <div>
            <div style="font-size:22px;font-weight:700;color:#111827;">
              <span class="odo" data-target="{stations}" data-decimals="0">0</span> stations
              &nbsp;+&nbsp;
              <span class="odo" data-target="{outposts}" data-decimals="0">0</span> outposts
            </div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">
              Source: KGIS police station &amp; outpost registry</div>
          </div>
        </div>

        <div class="ledger-row" style="display:flex;border-bottom:1px solid #F3F4F6;padding:16px 28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
                      color:#9CA3AF;width:220px;flex-shrink:0;padding-top:3px;">Coverage Index</div>
          <div>
            <div style="font-size:22px;font-weight:700;color:#111827;">
              <span class="odo" data-target="{coverage}" data-decimals="1">0</span> / 100
            </div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">
              Infrastructure relative to crime load &middot; higher is better</div>
          </div>
        </div>

        <div class="ledger-row" style="display:flex;padding:16px 28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;
                      color:#9CA3AF;width:220px;flex-shrink:0;padding-top:3px;">Resource Gap Score</div>
          <div>
            <div style="font-size:22px;font-weight:700;color:{gap_clr};">
              <span class="odo" data-target="{gap}" data-decimals="1" style="color:{gap_clr}">0</span> / 100
            </div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:2px;">
              Priority for intervention &middot; higher demands faster action</div>
          </div>
        </div>

      </div>
    </div>

    <script>
    (function() {{
      const els = document.querySelectorAll('.odo');
      const duration = 900;
      els.forEach(function(el) {{
        const target = parseFloat(el.getAttribute('data-target'));
        const decimals = parseInt(el.getAttribute('data-decimals'));
        const start = 0;
        const startTime = performance.now();

        function tick(now) {{
          const elapsed = now - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);  // ease-out cubic
          const value = start + (target - start) * eased;
          el.textContent = value.toLocaleString('en-IN', {{
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
          }});
          if (progress < 1) {{
            requestAnimationFrame(tick);
          }}
        }}
        requestAnimationFrame(tick);
      }});
    }})();
    </script>
    """
    components.html(html, height=560, scrolling=False)
    """Render the page title + form-number tag stacked vertically (tag on its
    own line above the title). This guarantees no overlap regardless of title
    length or viewport width — there is no shared line-box for the two
    elements to collide on."""
    st.markdown(
        f"<div class='page-head-table'>"
        f"<div class='page-head-tag-row'><span class='page-tag'>{form_tag}</span></div>"
        f"<span class='page-title'>{title}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── Police resource planning logic lives in its own module ────────────────────
# (KGIS district-code mapping, alias resolution, coverage/gap scoring, and the
# rule-based recommendation engine). app.py only renders; it doesn't clean data.
from police_resource_engine import (
    load_police_data,
    compute_coverage,
    build_recommendations,
)



# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px'>
        <div style='display:inline-flex;align-items:center;justify-content:center;
                    background:#3B82F6;border-radius:14px;
                    width:54px;height:54px;font-size:22px;
                    font-weight:700;color:#FFFFFF'>KA</div>
        <div style='font-size:20px;font-weight:700;
                    margin-top:12px;color:#111827;letter-spacing:-0.01em'>
            Karnataka Crime<br>Risk Dashboard
        </div>
        <div style='font-size:11px;font-weight:600;
                    letter-spacing:0.08em;color:#9CA3AF;margin-top:4px'>
            DISTRICT INTELLIGENCE SYSTEM
        </div>
    </div>
    <hr style='border-color:#E5E7EB;margin:8px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", label_visibility="collapsed",
        options=["📍  District Prediction",
                 "🗺️  Bengaluru Hotspot Map",
                 "📊  All Districts Ranking",
                 "🚔  Police Resource Planning"]
    )

    st.markdown("""
    <hr style='border-color:#E5E7EB;margin:12px 0 8px'>
    <div style='font-size:15px;line-height:2.2'>
        <b>Model A</b> — DCVI Prediction<br>
        <b>Model B</b> — KDE Hotspot Map<br>
        <b>Data</b> — NCRB Karnataka 2020–23<br>
        <b>R2</b> = {r2:.3f} | <b>MAE</b> = {mae:.0f}<br>
        <b>Algorithm</b> — Random Forest
    </div>
    """.format(r2=metrics['r2'], mae=metrics['mae']), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 1 — DISTRICT PREDICTION
# ════════════════════════════════════════════════════════════
if page == "📍  District Prediction":

    page_header("District Crime Risk Prediction", "FORM&nbsp;DCVI-01")
    st.markdown("<div class='info-box'>Select a district and year to see DCVI score, "
                "risk level, predicted crime count, and category breakdown.</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([4, 1])
    with c1:
        district = st.selectbox("Select district",
                                sorted(pred_df['district'].tolist()),
                                format_func=lambda x: x.title())
    with c2:
        year = st.selectbox("Year", [2024, 2025])

    row         = pred_df[pred_df['district'] == district].iloc[0]
    risk        = str(row['risk'])
    pred_crimes = int(row['pred_crimes_2024'] if year == 2024 else row['pred_crimes_2025'])
    change_pct  = float(row['change_pct'])
    arrow       = "↑" if change_pct > 0 else "↓"

    # Metric row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("DCVI Score",          f"{row['dcvi_score']:.1f} / 100")
    m2.metric(f"Predicted {year}",    f"{pred_crimes:,}")
    m3.metric("Actual 2023",          f"{int(row['actual_2023']):,}")
    m4.metric("Change vs 2023",
              f"{arrow} {abs(change_pct):.1f}%",
              delta=f"{change_pct:+.1f}%",
              delta_color="inverse" if change_pct > 0 else "normal")
    with m5:
        st.markdown(
            f"<div style='margin-top:8px'>"
            f"<p style='font-size:12px !important;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:.04em;"
            f"color:#9CA3AF;margin-bottom:14px'>RISK LEVEL</p>"
            f"<span class='badge badge-{risk}'>{risk}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin:18px 0 14px'>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<p class='sec-head'>Karnataka district map</p>",
                    unsafe_allow_html=True)

        m = folium.Map(location=[14.8, 76.5], zoom_start=6,
                       tiles="CartoDB positron")

        for _, r in pred_df.iterrows():
            d = str(r['district'])
            if d not in COORDS: continue
            lat, lon = COORDS[d]
            rc  = RISK_CLR.get(str(r['risk']), '#888')
            sel = (d == district)
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(9, r['dcvi_score'] / 7.5),
                color='#1e1e2e' if sel else rc,
                fill=True, fill_color=rc,
                fill_opacity=1.0 if sel else 0.72,
                weight=4 if sel else 1.5,
                tooltip=folium.Tooltip(
                    f"<b>{d.title()}</b><br>DCVI: {r['dcvi_score']:.1f}<br>Risk: {r['risk']}",
                    sticky=True)
            ).add_to(m)
            if sel:
                folium.Marker([lat+0.18, lon],
                    icon=folium.DivIcon(
                        html=f"<div style='font-size:25px;font-weight:700;"
                             f"color:#1e1e2e'>{d.title()}</div>",
                        icon_size=(160,20))
                ).add_to(m)

        st_folium(m, height=880, use_container_width=True, returned_objects=[])

    with right:
        st.markdown("<p class='sec-head'>Crime category breakdown — 2023 actual</p>",
                    unsafe_allow_html=True)

        cats   = ['Women','Children','Cyber']
        values = [int(row.get('crime_against_women',0)),
                  int(row.get('crime_against_children',0)),
                  int(row.get('cyber_crime',0)),]
        clrs   = ['#DC2626','#D97706','#16A34A']

        fig_bar = go.Figure(go.Bar(
            x=cats, y=values, marker_color=clrs, marker_line_width=0,
            text=[f"{v:,}" for v in values],
            textposition='outside', textfont=dict(size=13, color='#111827')
        ))
        fig_bar.update_layout(
            margin=dict(t=20,b=10,l=10,r=10), height=400,
            showlegend=False, bargap=0.55,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                       tickfont=dict(size=13,color='#9CA3AF')),
            xaxis=dict(showgrid=False, tickfont=dict(size=16,color='#111827'))
        )
        st.plotly_chart(fig_bar, use_container_width=True, key='cat_bar')

        st.markdown("<p class='sec-head'>Crime trend 2020–2025</p>",
                    unsafe_allow_html=True)

        hist     = hist_data.get(district, {})
        yrs_act  = [2020,2021,2022,2023]
        act_vals = [hist.get(y) for y in yrs_act]
        yrs_pred = [2023,2024,2025]
        prd_vals = [int(row['actual_2023']),
                    int(row['pred_crimes_2024']),
                    int(row['pred_crimes_2025'])]

        fig_ln = go.Figure()
        fig_ln.add_trace(go.Scatter(
            x=yrs_act, y=act_vals, mode='lines+markers', name='Actual',
            line=dict(color='#16A34A', width=3),
            marker=dict(size=8, color='#16A34A', line=dict(color='#FFFFFF',width=2))
        ))
        fig_ln.add_trace(go.Scatter(
            x=yrs_pred, y=prd_vals, mode='lines+markers', name='Predicted',
            line=dict(color='#DC2626', width=3, dash='dash'),
            marker=dict(size=8, color='#DC2626', line=dict(color='#FFFFFF',width=2))
        ))
        fig_ln.add_vline(x=2023, line_dash='dot', line_color='#D1D5DB',
                         annotation_text='Prediction starts',
                         annotation_font_size=20, annotation_font_color='#9CA3AF',
                         annotation_position='top right')
        fig_ln.update_layout(
            margin=dict(t=10,b=10,l=10,r=10), height=400,
            showlegend=True,
            legend=dict(orientation='h', y=1.12, x=0,
                        font=dict(size=15), bgcolor='rgba(0,0,0,0)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                       tickfont=dict(size=13,color='#9CA3AF')),
            xaxis=dict(showgrid=False, dtick=1,
                       tickfont=dict(size=14,color='#111827'))
        )
        st.plotly_chart(fig_ln, use_container_width=True, key='trend_line')


# ════════════════════════════════════════════════════════════
#  PAGE 2 — BENGALURU HOTSPOT MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️  Bengaluru Hotspot Map":

    page_header("Bengaluru Hotspot Survey", "FORM&nbsp;DCVI-02")
    st.markdown(
        "<div class='info-box'>Model B — KDE (Kernel Density Estimation) on 6,303 real "
        "GPS crime incidents. Red = highest crime density. Use layer control (top-right) "
        "to toggle heatmap and individual crime markers.</div>",
        unsafe_allow_html=True
    )

    # Try all filenames the user might have created
    possible = [
        "outputs-maps/bengaluru_hotspot_all.html",
        "outputs-maps/bengaluru_hotspot_murder.html",
        "outputs-maps/karnataka_crime_2024_fixed.html",
        "outputs-maps/karnataka_crime_2024.html",
        "outputs-maps/final_crime_heatmap.html",
        "outputs-maps/bengaluru_hotspot_all_crimes.html",
    ]

    html_file = next((f for f in possible if os.path.exists(f)), None)

    if html_file:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.success(f"✅ Loaded: **{html_file}**")
        st.components.v1.html(html_content, height=650, scrolling=False)
    else:
        st.warning("⚠️ Bengaluru Hotspot Map Not Found")
        st.markdown(
            "To generate the detailed hotspot map, please run the `hotspot_map.py` script from the `Map_File` directory:"
        )
        st.code("python Map_File/hotspot_map.py", language="bash")
        st.markdown("The app is looking for `outputs-maps/bengaluru_hotspot_all.html` or one of the following fallback files:")
        for f in possible:
            st.code(f)
        # Placeholder map
        ph = folium.Map(location=[12.9716,77.5946], zoom_start=7,
                        tiles='CartoDB dark_matter')
        folium.Marker([13.0716,77.1946],
                      tooltip="Run kar_map_fixed.py to load real hotspot data"
                      ).add_to(ph)
        st_folium(ph, height=800, use_container_width=True, returned_objects=[])


# ════════════════════════════════════════════════════════════
#  PAGE 3 — ALL DISTRICTS RANKING
# ════════════════════════════════════════════════════════════
elif page == "📊  All Districts Ranking":

    page_header("State-Wide District Register", "FORM&nbsp;DCVI-03")
    st.markdown(
        "<div class='info-box'>DCVI score (0–100) combines crime volume (35%), "
        "women+children ratio (35%), trend (20%), and crime variety (10%). "
        "Higher score = higher vulnerability.</div>",
        unsafe_allow_html=True
    )

    year_sel = st.radio("Prediction year", [2024, 2025], horizontal=True)
    col_pred = 'pred_crimes_2024' if year_sel == 2024 else 'pred_crimes_2025'

    # Create display dataframe with formatted values
    display_df = pred_df[['district','dcvi_score','risk',
                           col_pred,'actual_2023','change_pct']].copy()
    display_df.columns = ['District','DCVI Score','Risk',
                           f'Predicted {year_sel}','Actual 2023','Change %']
    display_df['District'] = display_df['District'].str.title()
    display_df = display_df.sort_values('DCVI Score', ascending=False).reset_index(drop=True)
    display_df.insert(0, 'Rank', range(1, len(display_df)+1))

    # Create formatted display version for table
    display_df_formatted = display_df.copy()
    display_df_formatted['DCVI Score'] = display_df_formatted['DCVI Score'].round(1).astype(str)
    display_df_formatted[f'Predicted {year_sel}'] = display_df_formatted[f'Predicted {year_sel}'].apply(lambda x: f"{int(x):,}")
    display_df_formatted['Actual 2023'] = display_df_formatted['Actual 2023'].apply(lambda x: f"{int(x):,}")
    display_df_formatted['Change %'] = display_df_formatted['Change %'].apply(lambda x: f"{x:+.1f}%")

    display_style = display_df_formatted.style.hide(axis='index').set_table_styles([
        {'selector': 'th', 'props': [('font-size', '14px'), ('padding', '12px 16px'),
                                      ('text-align','left'), ('background-color','#FFFFFF'),
                                      ('color','#111827')]},
        {'selector': 'td', 'props': [('font-size', '14px'), ('padding', '12px 16px'),
                                      ('text-align','left'), ('background-color','#FFFFFF'),
                                      ('color','#111827'), ('border-bottom','1px solid #F3F4F6')]},
        {'selector': 'thead th', 'props': [('font-size', '12px'), ('font-weight', '700'),
                                            ('background-color','#F9FAFB'), ('color','#374151'),
                                            ('text-transform','uppercase'), ('letter-spacing','0.03em')]},
        {'selector': 'tbody tr', 'props': [('height', '44px')]}
    ])
    st.dataframe(display_style, use_container_width=True, height=560)

    st.markdown("<div class='dossier-divider'>DISTRICT COMPARISON</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='sec-head'>All districts ranked by DCVI score</p>",
                unsafe_allow_html=True)

    # Keep numeric values for charting
    chart_df = display_df.sort_values('DCVI Score', ascending=True)
    fig_all  = px.bar(
        chart_df, x='DCVI Score', y='District',
        color='Risk',
        color_discrete_map=RISK_CLR,
        orientation='h', height=700, text='DCVI Score'
    )
    fig_all.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside',
        textfont=dict(size=15, color='#111827'),
        marker_line_width=0
    )
    fig_all.update_layout(
    margin=dict(t=80,b=20,l=10,r=80),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=True,

    legend=dict(
        orientation='h',
        y=1.12,
        x=0,
        font=dict(size=18),
        title=None
    ),
    xaxis=dict(showgrid=True, gridcolor='#F3F4F6', tickfont=dict(color='#9CA3AF')),
    yaxis=dict(showgrid=False, tickfont=dict(color='#111827'))
    )
    fig_all.add_vline(
    x=P33,
    line_dash='dash',
    line_color='#16A34A',
    annotation_text=""
    )

    fig_all.add_vline(
    x=P66,
    line_dash='dash',
    line_color='#DC2626',
    annotation_text=""
    )
    st.caption(
    f"Risk thresholds: LOW < {P33:.0f}, "
    f"MEDIUM {P33:.0f}-{P66:.0f}, "
    f"HIGH > {P66:.0f}"
)
    st.plotly_chart(fig_all, use_container_width=True, key='all_bar')

    st.markdown("<div class='dossier-divider'>STATE SUMMARY</div>",
                unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("High risk districts",   str((pred_df['risk']=='HIGH').sum()))
    s2.metric("Medium risk districts", str((pred_df['risk']=='MEDIUM').sum()))
    s3.metric("Low risk districts",    str((pred_df['risk']=='LOW').sum()))
    s4.metric("Avg DCVI score",        f"{pred_df['dcvi_score'].mean():.1f}")
    
# ════════════════════════════════════════════════════════════
#  PAGE 4 — POLICE RESOURCE PLANNING
# ════════════════════════════════════════════════════════════
elif page == "🚔  Police Resource Planning":

    page_header("District Resource Dossier", "FORM&nbsp;DCVI-04")
    st.markdown(
        "<div class='dossier-subtitle' style='margin-bottom:18px'>"
        "Cross-references predicted crime load against deployed station &amp; outpost "
        "strength to flag where coverage has not kept pace with risk.</div>",
        unsafe_allow_html=True
    )

    # Load police data and merge
    police_df, police_geo = load_police_data()
    merged_df = compute_coverage(pred_df, police_df)

    with st.expander("Data matching diagnostics"):
        check_df = merged_df[['district','kgis_root','police_stations',
                               'police_outposts','is_special']].copy()
        check_df['Status'] = check_df['is_special'].map(
            {True: 'Special jurisdiction (non-geographic)', False: 'Geographic district'})
        check_df = check_df.drop(columns=['is_special'])
        check_df.columns = ['Prediction District','Matched KGIS Root','Stations',
                             'Outposts','Status']
        real_gaps = ((check_df['Stations']==0) & (check_df['Outposts']==0)
                     & (check_df['Status']=='Geographic district')).sum()
        st.markdown(f"**{real_gaps}** geographic district(s) show zero infrastructure "
                    f"after alias resolution. Non-geographic units (e.g. railway police) "
                    f"are expected to show zero and are labeled accordingly below.")
        st.dataframe(check_df, use_container_width=True, height=300)

    # ── District Selector ─────────────────────────────────────
    district = st.selectbox(
        "Select district",
        sorted(merged_df['district'].tolist()),
        key="p4_district"
    )
    row = merged_df[merged_df['district'] == district].iloc[0]
    risk = row['risk']
    case_id = f"KA-{abs(hash(district)) % 9000 + 1000}"
    gap_clr = '#DC2626' if row['gap_score'] >= 65 else '#111827'

    # ── District Summary Card (animated odometer counters) ──────
    render_odometer_card(
        district=district,
        case_id=case_id,
        risk=risk,
        dcvi=float(row['dcvi_score']),
        pred_crimes=int(row['pred_crimes_2024']),
        actual_2023=int(row['actual_2023']),
        stations=int(row['police_stations']),
        outposts=int(row['police_outposts']),
        coverage=float(row['coverage_index']),
        gap=float(row['gap_score']),
        gap_clr=gap_clr
    )

    # ── Two-column layout: Coverage Ledger Bar + Crime Breakdown ──
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<p class='sec-head'>Coverage vs. Gap, at a glance</p>", unsafe_allow_html=True)
        fig_strip = go.Figure()
        fig_strip.add_trace(go.Bar(
            y=['Coverage Index', 'Gap Score'],
            x=[row['coverage_index'], row['gap_score']],
            orientation='h',
            marker_color=['#16A34A', '#DC2626'],
            text=[f"{row['coverage_index']:.1f}", f"{row['gap_score']:.1f}"],
            textposition='outside',
            textfont=dict(size=15, color='#111827'),
            width=0.5
        ))
        fig_strip.update_layout(
            height=200,
            margin=dict(t=10, b=10, l=10, r=40),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0, 110], showgrid=True, gridcolor='#F3F4F6',
                       tickfont=dict(size=12, color='#9CA3AF')),
            yaxis=dict(tickfont=dict(size=14, color='#111827')),
            showlegend=False
        )
        st.plotly_chart(fig_strip, use_container_width=True, key='strip_chart')

    with right:
        st.markdown("<p class='sec-head'>Case mix — 2023 recorded crime</p>", unsafe_allow_html=True)
        cats = {
            'Women':    row.get('crime_against_women', 0),
            'Children': row.get('crime_against_children', 0),
            'Violent':  row.get('violent_crime', 0),
            'Cyber':    row.get('cyber_crime', 0),
            'Other':    row.get('other_crime', 0),
        }
        cat_df = pd.DataFrame({'Category': list(cats.keys()),
                               'Count': list(cats.values())})
        fig_cat = px.bar(cat_df, x='Category', y='Count',
                         color_discrete_sequence=['#6B7280'],
                         text='Count', height=200)
        fig_cat.update_traces(marker_color='#6B7280', textposition='outside',
                              textfont=dict(size=13, color='#111827'))
        fig_cat.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(size=13, color='#111827')),
            yaxis=dict(tickfont=dict(size=12, color='#9CA3AF'), showgrid=True,
                       gridcolor='#F3F4F6')
        )
        st.plotly_chart(fig_cat, use_container_width=True, key='cat_bar')

    st.markdown("<div class='dossier-divider'>RECOMMENDED DIRECTIVES</div>",
                unsafe_allow_html=True)

    # ── Recommendations as numbered directives ─────────────────
    recs = build_recommendations(row)
    for i, (title, body) in enumerate(recs, start=1):
        clean_title = title.split(' ', 1)[1] if ' ' in title else title
        st.markdown(
            f"<div class='directive-card'>"
            f"<div class='directive-num'>DIRECTIVE {i:02d}</div>"
            f"<div class='directive-title'>{clean_title}</div>"
            f"<div class='directive-body'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='dossier-divider'>STATE-WIDE PRIORITY REGISTER</div>",
                unsafe_allow_html=True)

    # ── District Rankings Table ────────────────────────────────
    st.markdown(
        "<div class='priority-strip'>Register sorted by Gap Score — districts at the "
        "top warrant the most urgent resource review.</div>",
        unsafe_allow_html=True
    )

    rank_df = merged_df[['district','risk','dcvi_score','police_stations',
                          'police_outposts','coverage_index','gap_score',
                          'pred_crimes_2024']].copy()
    rank_df.columns = ['District','Risk','DCVI','Stations','Outposts',
                        'Coverage Index','Gap Score','Pred. Crimes 2024']
    rank_df['District']       = rank_df['District'].str.title()
    rank_df['DCVI']           = rank_df['DCVI'].round(1)
    rank_df['Coverage Index'] = rank_df['Coverage Index'].round(1)
    rank_df['Gap Score']      = rank_df['Gap Score'].round(1)
    rank_df['Pred. Crimes 2024'] = rank_df['Pred. Crimes 2024'].apply(
        lambda x: f"{int(x):,}")
    rank_df = rank_df.sort_values('Gap Score', ascending=False).reset_index(drop=True)
    rank_df.insert(0, 'Priority', range(1, len(rank_df)+1))

    st.dataframe(rank_df, use_container_width=True, height=500)

    # ── Gap Score Bar Chart ────────────────────────────────────
    st.markdown("<p class='sec-head' style='margin-top:18px'>Resource gap — full district comparison</p>",
                unsafe_allow_html=True)

    chart_data = merged_df.sort_values('gap_score', ascending=True)
    fig_gap = px.bar(
        chart_data,
        x='gap_score', y='district',
        color='risk',
        color_discrete_map={'HIGH':'#DC2626','MEDIUM':'#D97706','LOW':'#16A34A'},
        orientation='h', height=900,
        text='gap_score',
        labels={'gap_score': 'Gap Score (0-100)', 'district': 'District'}
    )
    fig_gap.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside',
        textfont=dict(size=12),
        marker_line_width=0
    )
    fig_gap.update_layout(
        margin=dict(t=20, b=20, l=10, r=80),
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0,
                    font=dict(size=14), title=None),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6',
                   tickfont=dict(size=13, color='#9CA3AF')),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color='#111827'))
    )
    st.plotly_chart(fig_gap, use_container_width=True, key='gap_bar')

    # ── GIS Map — Crime Risk + Police Infrastructure ───────────
    st.markdown("<div class='dossier-divider'>FIELD MAP</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='dossier-subtitle' style='margin-bottom:14px'>"
        "Risk markers sized by DCVI score, overlaid with deployed station "
        "and outpost positions from the KGIS registry.</div>",
        unsafe_allow_html=True
    )

    # Build district centroids from existing coords dict
    DIST_COORDS = {
        "bagalkot":[16.18,75.69], "ballari":[15.15,76.93],
        "bangalore rural":[13.10,77.45], "bangalore urban":[12.97,77.59],
        "belagavi city":[15.85,74.50], "belagavi district":[15.70,74.90],
        "belgaum":[15.70,74.90],
        "bengaluru city":[12.97,77.59], "bengaluru district":[13.10,77.45],
        "bidar":[17.91,77.52], "chamarajanagara":[11.92,76.94],
        "chamarajnagar":[11.92,76.94],
        "chikkaballapura":[13.44,77.73], "chikkamagaluru":[13.32,75.77],
        "chitradurga":[14.23,76.40], "dakshina kannada":[12.84,74.86],
        "davanagere":[14.46,75.92], "dharwad":[15.46,75.01],
        "gadag":[15.43,75.63], "hassan":[13.00,76.10],
        "haveri":[14.80,75.40], "hubballi dharwad city":[15.36,75.12],
        "k.g.f.":[12.96,78.27], "kgf":[12.96,78.27],
        "k.railways":[15.0,76.5], "krailways":[15.0,76.5],
        "kalaburagi":[17.33,76.82], "kodagu":[12.42,75.74],
        "kolar":[13.14,78.13], "koppal":[15.35,76.15],
        "mandya":[12.52,76.90], "mangaluru city":[12.91,74.85],
        "mysuru city":[12.30,76.65], "mysuru district":[12.20,76.40],
        "mysuru":[12.30,76.65],
        "raichur":[16.20,77.35], "ramanagara":[12.72,77.28],
        "shivamogga":[13.93,75.57], "shimoga":[13.93,75.57],
        "tumakuru":[13.34,77.10], "udupi":[13.34,74.75],
        "uttara kannada":[14.79,74.13], "vijayanagara":[15.32,76.46],
        "vijayapura":[16.83,75.71], "bijapur":[16.83,75.71],
        "yadgiri":[16.77,77.14], "yadgir":[16.77,77.14],
    }

    m4 = folium.Map(location=[15.0, 76.5], zoom_start=7,
                    tiles='CartoDB positron')

    risk_colors_folium = {'HIGH': '#DC2626', 'MEDIUM': '#D97706', 'LOW': '#16A34A'}

    # Plot district risk circles
    for _, drow in merged_df.iterrows():
        dname = drow['district'].lower().strip()
        coords = DIST_COORDS.get(dname)
        if not coords:
            continue
        clr = risk_colors_folium.get(drow['risk'], '#6B7280')
        popup_html = (
            f"<b>{drow['district'].title()}</b><br>"
            f"Risk: {drow['risk']}<br>"
            f"DCVI: {drow['dcvi_score']:.1f}<br>"
            f"Gap Score: {drow['gap_score']:.1f}<br>"
            f"Stations: {int(drow['police_stations'])} | "
            f"Outposts: {int(drow['police_outposts'])}<br>"
            f"Coverage: {drow['coverage_index']:.1f}/100"
        )
        radius = 12000 + (drow['dcvi_score'] / 100) * 18000
        folium.Circle(
            location=coords,
            radius=radius,
            color=clr, fill=True, fill_color=clr, fill_opacity=0.35,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{drow['district'].title()} — {drow['risk']}"
        ).add_to(m4)

    # Plot police stations & outposts (clustered)
    from folium.plugins import MarkerCluster
    sta_cluster = MarkerCluster(name="Police Stations").add_to(m4)
    oup_cluster = MarkerCluster(name="Police Outposts").add_to(m4)

    for _, prow in police_geo.iterrows():
        if prow['type'] == 'Station':
            folium.CircleMarker(
                location=[prow['Latitude'], prow['Longitude']],
                radius=5, color='#1d4ed8', fill=True,
                fill_color='#1d4ed8', fill_opacity=0.8,
                popup=str(prow['name']),
                tooltip=str(prow['name'])
            ).add_to(sta_cluster)
        else:
            folium.CircleMarker(
                location=[prow['Latitude'], prow['Longitude']],
                radius=4, color='#7c3aed', fill=True,
                fill_color='#7c3aed', fill_opacity=0.8,
                popup=str(prow['name']),
                tooltip=str(prow['name'])
            ).add_to(oup_cluster)

    folium.LayerControl(collapsed=False).add_to(m4)

    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:#FFFFFF;
                padding:14px 18px;border:1px solid #E5E7EB;border-radius:10px;font-size:12px;
                font-family:Inter,Segoe UI,sans-serif;color:#111827;
                box-shadow:0 2px 10px rgba(0,0,0,0.08)'>
        <b style='letter-spacing:0.03em'>LEGEND</b><br>
        <span style='color:#DC2626'>●</span> HIGH risk district<br>
        <span style='color:#D97706'>●</span> MEDIUM risk district<br>
        <span style='color:#16A34A'>●</span> LOW risk district<br>
        <span style='color:#1d4ed8'>●</span> Police station<br>
        <span style='color:#7c3aed'>●</span> Police outpost<br>
        <i style='font-size:10px;color:#9CA3AF'>Circle size &prop; DCVI score</i>
    </div>
    """
    m4.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m4, width=None, height=600, key='p4_map')
