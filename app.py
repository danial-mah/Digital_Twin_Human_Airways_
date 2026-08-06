"""
app.py — Human Airways Digital Twin Dashboard
==============================================
Landing page. Describes the project, dataset, and methodology.
Navigate to the analysis pages via the sidebar.

Course: Digital Twin Methods
Institution: Università degli Studi di Roma Tor Vergata
"""

import streamlit as st
from core.i18n import t, lang_selector

st.set_page_config(
    page_title="Human Airways Digital Twin",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

lang_selector()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title(t("app_title"))
st.markdown(t("app_subtitle"))
st.divider()

# ── Quick metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(t("app_met_doe"),    "100")
c2.metric(t("app_met_params"), "26")
c3.metric(t("app_met_nodes"),  "2,135,906")
c4.metric(t("app_met_disp"),   "~42,700  (stride 50)")
c5.metric(t("app_met_pres"),   t("app_met_pres_val"))

st.divider()

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader(t("app_overview"))
    if st.session_state.get("lang", "en") == "it":
        st.markdown("""
Il dataset proviene dalle simulazioni **Ansys Twin Builder / CFD** del
sistema respiratorio umano. Ognuna delle 100 esecuzioni del Piano degli Esperimenti (DOE)
usa una geometria diversa delle vie aeree, descritta da **26 parametri**:
area della glottide, area dell'epiglottide, dimensioni della trachea e angoli di ramificazione
a tre livelli dell'albero bronchiale.

Per ogni esecuzione il simulatore fornisce:
- Le **coordinate dei nodi della maglia 3D** (geometria deformata, ~48 MB/istantanea)
- Il **campo di pressione statica** in Pascal (~16 MB/istantanea)

Il pipeline del Gemello Digitale trasforma questi dati CFD grezzi in un surrogato
rapido e interattivo che può predire la pressione per qualsiasi nuova geometria
in millisecondi anziché ore.
""")
    else:
        st.markdown("""
The dataset comes from **Ansys Twin Builder / CFD** simulations of the
human respiratory system. Each of the 100 Design-of-Experiment (DOE) runs
uses a different airway geometry, described by **26 parameters**:
glottis area, epiglottis area, trachea dimensions, and branching angles
at three levels of the bronchial tree.

For each run the simulator provides:
- The **3D mesh node coordinates** (deformed geometry, ~48 MB/snapshot)
- The **static pressure field** in Pascals (~16 MB/snapshot)

The Digital Twin pipeline transforms this raw CFD data into a fast,
interactive surrogate that can predict pressure at any new geometry
in milliseconds rather than hours.
""")

    st.subheader(t("app_anatomy"))
    st.markdown("""
```
Mouth / Larynx
     │
  Glottis / Epiglottis
     │
  Upper Trachea (top / middle / bottom)
     ├── Left main bronchus (L)
     │     ├── Left-Left (LL) ─── LLL, LLR
     │     └── Left-Right (LR) ── LRL, LRR
     └── Right main bronchus (R)
           ├── Right-Left (RL) ── RLL, RLR
           └── Right-Right (RR) ─ RRL, RRR
```
""")

with right:
    st.subheader(t("app_methodology"))
    _lang = st.session_state.get("lang", "en")
    steps_en = [
        (t("app_step1_title"), "100 DOE snapshots explored: geometry parameters, pressure statistics, 3D point cloud visualisation."),
        (t("app_step2_title"), "SVD on the (100 × N×3) coordinate matrix. ~10 modes capture 95 % of shape variance."),
        (t("app_step3_title"), "Latin Hypercube Sampling generates 1000 parameter sets within the original DOE bounds."),
        (t("app_step4_title"), "RBF (thin-plate spline) trained on DOE data predicts pressure POD scores at any new geometry in <1 ms."),
        (t("app_step5_title"), "SVD on the (100 × N) pressure matrix. ~8 modes capture 99 % of pressure variance."),
        (t("app_step6_title"), "Coupled geometry–pressure RBF maps 26 input parameters to the full pressure field via POD reconstruction."),
        (t("app_step7_title"), "Shape morphing via POD sliders, pressure field browser, DOE explorer, regional analysis."),
        (t("app_step8_title"), "CSV export of DOE table, LHS samples, and predicted pressure fields."),
        (t("app_step9_title"), "VTK-compatible CSV of node coordinates for external CFD validation workflows."),
    ]
    steps_it = [
        (t("app_step1_title"), "Esplorate 100 istantanee DOE: parametri geometrici, statistiche di pressione, visualizzazione nuvola di punti 3D."),
        (t("app_step2_title"), "SVD sulla matrice di coordinate (100 × N×3). ~10 modi catturano il 95% della varianza di forma."),
        (t("app_step3_title"), "Il campionamento Latin Hypercube genera 1000 set di parametri nei limiti DOE originali."),
        (t("app_step4_title"), "RBF (thin-plate spline) addestrato sui dati DOE predice i punteggi POD di pressione per qualsiasi nuova geometria in <1 ms."),
        (t("app_step5_title"), "SVD sulla matrice di pressione (100 × N). ~8 modi catturano il 99% della varianza di pressione."),
        (t("app_step6_title"), "RBF accoppiato geometria–pressione mappa 26 parametri di input al campo di pressione completo tramite ricostruzione POD."),
        (t("app_step7_title"), "Morfing della forma tramite cursori POD, browser del campo di pressione, esploratore DOE, analisi regionale."),
        (t("app_step8_title"), "Esportazione CSV della tabella DOE, campioni LHS e campi di pressione predetti."),
        (t("app_step9_title"), "CSV compatibile VTK delle coordinate dei nodi per flussi di lavoro di validazione CFD esterni."),
    ]
    steps = steps_it if _lang == "it" else steps_en
    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

st.divider()

# ── Navigation guide ───────────────────────────────────────────────────────────
st.subheader(t("app_pages"))
_lang = st.session_state.get("lang", "en")
pages_en = [
    ("🔬 Geometry Explorer",  "Morph the airway shape via POD mode sliders."),
    ("🌡️ Pressure Field",     "3D pressure map for any of the 100 DOE snapshots."),
    ("📐 DOE Analysis",        "Parallel coordinates, scatter plots, correlation heatmap, LHS coverage."),
    ("📊 POD Analysis",        "Energy curves, mode shapes, and reconstruction error for geometry and pressure."),
    ("🔮 RBF / GP Inference",  "Predict the full pressure field at any new set of 26 geometry parameters. Compare RBF vs GP."),
    ("🫀 Regional Analysis",   "Mean pressure per anatomical region across snapshots."),
]
pages_it = [
    ("🔬 Esplora Geometria",  "Deforma la forma delle vie aeree tramite cursori POD."),
    ("🌡️ Campo di Pressione", "Mappa di pressione 3D per ognuna delle 100 istantanee DOE."),
    ("📐 Analisi DOE",         "Coordinate parallele, scatter plot, mappa di correlazione, copertura LHS."),
    ("📊 Analisi POD",         "Curve di energia, forme dei modi ed errore di ricostruzione per geometria e pressione."),
    ("🔮 Inferenza RBF / GP", "Predici il campo di pressione completo per qualsiasi set di 26 parametri geometrici. Confronta RBF vs GP."),
    ("🫀 Analisi Regionale",  "Pressione media per regione anatomica su tutte le istantanee."),
]
pages = pages_it if _lang == "it" else pages_en
cols = st.columns(3)
for i, (name, desc) in enumerate(pages):
    with cols[i % 3]:
        st.info(f"**{name}**\n\n{desc}")
