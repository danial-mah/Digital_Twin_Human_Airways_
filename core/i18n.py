"""
core/i18n.py
============
Internationalisation (i18n) for the Human Airways Digital Twin dashboard.
Supported languages: English (en) · Italian (it)

Usage in any page
-----------------
    from core.i18n import t, lang_selector
    lang_selector()          # renders the 🌐 radio in the sidebar
    st.title(t("app_title")) # looks up the current language automatically
"""

from __future__ import annotations
import streamlit as st

# ── Translation dictionary ─────────────────────────────────────────────────────

_T: dict[str, dict[str, str]] = {

# ══════════════════════════════════════════════════════════════════════════════
# SHARED / COMMON
# ══════════════════════════════════════════════════════════════════════════════
"lang_label": {
    "en": "🌐 Language",
    "it": "🌐 Lingua",
},

# Shared buttons
"btn_reset":         {"en": "Reset to Mean",          "it": "Azzera alla Media"},
"btn_random":        {"en": "Random Shape",            "it": "Forma Casuale"},
"btn_clear_chat":    {"en": "🗑 Clear chat",            "it": "🗑 Cancella chat"},
"btn_download":      {"en": "Download",                "it": "Scarica"},
"btn_compute":       {"en": "Compute",                 "it": "Calcola"},
"btn_run":           {"en": "Run",                     "it": "Esegui"},
"btn_generate":      {"en": "Generate",                "it": "Genera"},
"btn_predict":       {"en": "Predict",                 "it": "Predici"},

# Shared labels
"lbl_snapshot":      {"en": "DOE Run (Snapshot)",      "it": "Esecuzione DOE (Istantanea)"},
"lbl_opacity":       {"en": "Opacity",                 "it": "Opacità"},
"lbl_point_size":    {"en": "Point size",              "it": "Dimensione punti"},
"lbl_colour_map":    {"en": "Colour map",              "it": "Mappa cromatica"},
"lbl_colour_by":     {"en": "Colour by",               "it": "Colora per"},
"lbl_region":        {"en": "Region filter",           "it": "Filtro regione"},
"lbl_x_axis":        {"en": "X axis",                  "it": "Asse X"},
"lbl_y_axis":        {"en": "Y axis",                  "it": "Asse Y"},
"lbl_n_folds":       {"en": "Number of folds k",       "it": "Numero di fold k"},
"lbl_random_seed":   {"en": "Random seed",             "it": "Seme casuale"},
"lbl_modes":         {"en": "Modes",                   "it": "Modi"},
"lbl_frames":        {"en": "Frames",                  "it": "Fotogrammi"},
"lbl_resolution":    {"en": "Resolution (stride)",     "it": "Risoluzione (passo)"},

# Shared metrics
"met_min_pres":      {"en": "Min pressure",            "it": "Pressione min"},
"met_max_pres":      {"en": "Max pressure",            "it": "Pressione max"},
"met_mean_pres":     {"en": "Mean pressure",           "it": "Pressione media"},
"met_std_pres":      {"en": "Std deviation",           "it": "Dev. standard"},
"met_nodes":         {"en": "Nodes shown",             "it": "Nodi visualizzati"},
"met_delta_p":       {"en": "ΔP (resistance)",         "it": "ΔP (resistenza)"},
"met_resist_idx":    {"en": "Resistance index",        "it": "Indice di resistenza"},

# Shared sidebar
"sidebar_controls":  {"en": "Controls",                "it": "Controlli"},
"sidebar_settings":  {"en": "Settings",                "it": "Impostazioni"},
"sidebar_vis":       {"en": "Visualisation Controls",  "it": "Controlli visualizzazione"},

# ══════════════════════════════════════════════════════════════════════════════
# APP.PY — Landing page
# ══════════════════════════════════════════════════════════════════════════════
"app_title": {
    "en": "🫁 Human Airways Digital Twin Dashboard",
    "it": "🫁 Dashboard Gemello Digitale Vie Aeree Umane",
},
"app_subtitle": {
    "en": "**Digital Twin Methods** | Università degli Studi di Roma Tor Vergata",
    "it": "**Metodi per il Gemello Digitale** | Università degli Studi di Roma Tor Vergata",
},
"app_overview":      {"en": "Project Overview",        "it": "Panoramica del Progetto"},
"app_anatomy":       {"en": "Airway Anatomy Modelled", "it": "Anatomia delle Vie Aeree Modellata"},
"app_methodology":   {"en": "Methodology — 9 Steps",   "it": "Metodologia — 9 Passi"},
"app_pages":         {"en": "Dashboard Pages",         "it": "Pagine della Dashboard"},

"app_met_doe":       {"en": "DOE Snapshots",           "it": "Istantanee DOE"},
"app_met_params":    {"en": "Geometry Parameters",     "it": "Parametri Geometrici"},
"app_met_nodes":     {"en": "Mesh Nodes (full)",       "it": "Nodi della Maglia (completa)"},
"app_met_disp":      {"en": "Displayed Nodes",         "it": "Nodi Visualizzati"},
"app_met_pres":      {"en": "Pressure Field",          "it": "Campo di Pressione"},
"app_met_pres_val":  {"en": "Static (Pa)",             "it": "Statica (Pa)"},

"app_step1_title":   {"en": "1 · Inspect the database",               "it": "1 · Ispezione del database"},
"app_step2_title":   {"en": "2 · POD reduction of geometry",          "it": "2 · Riduzione POD della geometria"},
"app_step3_title":   {"en": "3 · Upscale with 1000 virtual shapes",   "it": "3 · Upscaling con 1000 forme virtuali"},
"app_step4_title":   {"en": "4 · Evaluate with RBF surrogate",        "it": "4 · Valutazione con surrogato RBF"},
"app_step5_title":   {"en": "5 · POD reduction of pressure",          "it": "5 · Riduzione POD della pressione"},
"app_step6_title":   {"en": "6 · RBF inference in reduced spaces",    "it": "6 · Inferenza RBF negli spazi ridotti"},
"app_step7_title":   {"en": "7 · Interactive Streamlit dashboard",    "it": "7 · Dashboard Streamlit interattiva"},
"app_step8_title":   {"en": "8 · Export for VR / post-processing",    "it": "8 · Esportazione per VR / post-elaborazione"},
"app_step9_title":   {"en": "9 · Export synthesised geometries",      "it": "9 · Esportazione geometrie sintetiche"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Geometry Explorer
# ══════════════════════════════════════════════════════════════════════════════
"geo_title": {
    "en": "🔬 Geometry Explorer — POD Shape Morphing",
    "it": "🔬 Esplora Geometria — Morfing della Forma con POD",
},
"geo_caption": {
    "en": "Adjust mode sliders to morph the airway geometry in real time.",
    "it": "Regola i cursori dei modi per deformare la geometria delle vie aeree in tempo reale.",
},
"geo_controls":      {"en": "**Controls**",            "it": "**Controlli**"},
"geo_pod_sliders":   {"en": "**POD Mode Sliders**",    "it": "**Cursori Modo POD**"},
"geo_visualisation": {"en": "**Visualisation**",       "it": "**Visualizzazione**"},
"geo_active_coeff":  {"en": "**Active coefficients**", "it": "**Coefficienti attivi**"},
"geo_modes_95":      {"en": "Modes for 95 %",          "it": "Modi per il 95%"},
"geo_modes_99":      {"en": "Modes for 99 %",          "it": "Modi per il 99%"},
"geo_total_modes":   {"en": "Total modes",             "it": "Modi totali"},
"geo_modes_display": {"en": "Modes to display",        "it": "Modi da visualizzare"},
"geo_load_snap":     {"en": "Load snapshot",           "it": "Carica istantanea"},
"geo_show_ghost":    {"en": "Show mean shape (ghost)", "it": "Mostra forma media (fantasma)"},
"geo_deviation":     {"en": "Shape deviation ‖c‖",     "it": "Deviazione forma ‖c‖"},
"geo_disp_mean":     {"en": "Displacement from mean",  "it": "Spostamento dalla media"},
"geo_mode_contrib":  {"en": "Mode contribution",       "it": "Contributo del modo"},
"geo_z_coord":       {"en": "Z-coordinate",            "it": "Coordinata Z"},
"geo_uniform":       {"en": "Uniform",                 "it": "Uniforme"},
"geo_choose":        {"en": "— choose —",              "it": "— scegli —"},
"geo_pod_energy":    {"en": "POD Energy",              "it": "Energia POD"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Pressure Field
# ══════════════════════════════════════════════════════════════════════════════
"pres_title": {
    "en": "🌡️ 3D Pressure Field Viewer",
    "it": "🌡️ Visualizzatore del Campo di Pressione 3D",
},
"pres_caption": {
    "en": "Static pressure (Pa) on the human airway mesh for each DOE simulation run.",
    "it": "Pressione statica (Pa) sulla maglia delle vie aeree umane per ogni esecuzione DOE.",
},
"pres_deformed":     {"en": "Deformed geometry",       "it": "Geometria deformata"},
"pres_params_hdr":   {"en": "Parameters",              "it": "Parametri"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DOE Analysis
# ══════════════════════════════════════════════════════════════════════════════
"doe_title": {
    "en": "📐 Design of Experiments — Parameter Space Analysis",
    "it": "📐 Piano degli Esperimenti — Analisi dello Spazio Parametrico",
},
"doe_caption": {
    "en": "Explore how the 26 geometry parameters are distributed across the 100 DOE snapshots and compare with a 1000-point LHS upscaling.",
    "it": "Esplora come i 26 parametri geometrici sono distribuiti nelle 100 istantanee DOE e confronta con il campionamento LHS a 1000 punti.",
},
"doe_highlight":     {"en": "Highlight run",           "it": "Evidenzia esecuzione"},
"doe_tab_parallel":  {"en": "📊 Parallel Coordinates", "it": "📊 Coordinate Parallele"},
"doe_tab_scatter":   {"en": "🔵 Pairwise Scatter",     "it": "🔵 Scatter a Coppie"},
"doe_tab_corr":      {"en": "🔥 Correlation Heatmap",  "it": "🔥 Mappa Correlazione"},
"doe_tab_lhs":       {"en": "🎯 LHS Coverage",         "it": "🎯 Copertura LHS"},
"doe_sub_parallel":  {"en": "All 100 DOE Runs — 26 Parameters",      "it": "Tutte le 100 esecuzioni DOE — 26 Parametri"},
"doe_sub_ranges":    {"en": "Parameter Ranges",        "it": "Intervalli dei Parametri"},
"doe_sub_pairwise":  {"en": "Pairwise Parameter Scatter",            "it": "Scatter a Coppie dei Parametri"},
"doe_sub_corr":      {"en": "Pearson Correlation Matrix — 26 Parameters", "it": "Matrice di Correlazione Pearson — 26 Parametri"},
"doe_sub_lhs":       {"en": "LHS Upscaling: 100 DOE → 1000 Virtual Shapes", "it": "Upscaling LHS: 100 DOE → 1000 Forme Virtuali"},
"doe_discrepancy":   {"en": "DOE L2-discrepancy (100 pts)",           "it": "Discrepanza L2 DOE (100 pt)"},
"doe_lhs_discr":     {"en": "LHS L2-discrepancy (1000 pts)",          "it": "Discrepanza L2 LHS (1000 pt)"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POD Analysis
# ══════════════════════════════════════════════════════════════════════════════
"pod_title": {
    "en": "📊 POD Analysis — Dimensionality Reduction",
    "it": "📊 Analisi POD — Riduzione della Dimensionalità",
},
"pod_caption": {
    "en": "Singular Value Decomposition (SVD) applied to geometry and pressure snapshots. Identify how many modes capture the essential physics.",
    "it": "Decomposizione ai Valori Singolari (SVD) applicata alle istantanee di geometria e pressione. Identifica quanti modi catturano la fisica essenziale.",
},
"pod_tab_energy":    {"en": "📈 Energy Curves",        "it": "📈 Curve di Energia"},
"pod_tab_modes":     {"en": "🌊 Mode Shapes",          "it": "🌊 Forme dei Modi"},
"pod_tab_recon":     {"en": "🔁 Reconstruction",       "it": "🔁 Ricostruzione"},
"pod_tab_val":       {"en": "✅ Validation",            "it": "✅ Validazione"},
"pod_sub_energy":    {"en": "Cumulative Energy vs Number of Modes",   "it": "Energia Cumulativa vs Numero di Modi"},
"pod_sub_sv":        {"en": "Singular Value Spectrum",                "it": "Spettro dei Valori Singolari"},
"pod_sub_modes":     {"en": "Mode Shapes — 3D Visualisation",         "it": "Forme dei Modi — Visualizzazione 3D"},
"pod_sub_recon":     {"en": "Pressure Reconstruction",               "it": "Ricostruzione della Pressione"},
"pod_sub_err_modes": {"en": "Reconstruction Error vs Number of Modes","it": "Errore di Ricostruzione vs Numero di Modi"},
"pod_sub_kfold":     {"en": "K-Fold POD Reconstruction Validation",  "it": "Validazione Ricostruzione POD con K-Fold"},
"pod_geo_modes95":   {"en": "Geo: modes for 95 %",   "it": "Geo: modi per 95%"},
"pod_geo_modes99":   {"en": "Geo: modes for 99 %",   "it": "Geo: modi per 99%"},
"pod_pres_modes95":  {"en": "Pres: modes for 95 %",  "it": "Press: modi per 95%"},
"pod_pres_modes99":  {"en": "Pres: modes for 99 %",  "it": "Press: modi per 99%"},
"pod_geo_label":     {"en": "Geometry (Coordinates)", "it": "Geometria (Coordinate)"},
"pod_pres_label":    {"en": "Pressure Field",         "it": "Campo di Pressione"},
"pod_n_modes":       {"en": "Number of Modes",        "it": "Numero di Modi"},
"pod_cum_energy":    {"en": "Cumulative Energy (%)",  "it": "Energia Cumulativa (%)"},
"pod_mode_field":    {"en": "Field",                  "it": "Campo"},
"pod_mode_number":   {"en": "Mode number",            "it": "Numero del modo"},
"pod_snap_recon":    {"en": "Snapshot to reconstruct","it": "Istantanea da ricostruire"},
"pod_modes_used":    {"en": "Number of modes used",   "it": "Numero di modi usati"},
"pod_rel_err":       {"en": "Relative L2 error",      "it": "Errore L2 relativo"},
"pod_original":      {"en": "Original",               "it": "Originale"},
"pod_reconstructed": {"en": "Reconstructed",          "it": "Ricostruito"},
"pod_error_map":     {"en": "Error Map",              "it": "Mappa degli Errori"},
"pod_run_kfold":     {"en": "Run K-Fold POD Reconstruction", "it": "Esegui Ricostruzione POD K-Fold"},
"pod_n_folds":       {"en": "Number of folds",        "it": "Numero di fold"},
"pod_kfold_modes":   {"en": "POD modes",              "it": "Modi POD"},
"pod_geo_mode":      {"en": "Geometry mode",          "it": "Modo geometrico"},
"pod_pres_mode":     {"en": "Pressure mode",          "it": "Modo pressione"},
"pod_disp_mag":      {"en": "Displacement magnitude", "it": "Ampiezza spostamento"},
"pod_pres_mode_val": {"en": "Pressure mode value",    "it": "Valore modo pressione"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RBF / GP Inference
# ══════════════════════════════════════════════════════════════════════════════
"rbf_title": {
    "en": "🔮 Surrogate Models — RBF & Gaussian Process Comparison",
    "it": "🔮 Modelli Surrogato — Confronto RBF e Processo Gaussiano",
},
"rbf_caption": {
    "en": "Radial Basis Function and Gaussian Process surrogates map 26 geometry parameters to the full pressure field via the POD reduced space. Compare accuracy and speed.",
    "it": "I surrogati RBF e Processo Gaussiano mappano 26 parametri geometrici al campo di pressione completo tramite lo spazio ridotto POD. Confronta accuratezza e velocità.",
},
"rbf_sidebar_hdr":   {"en": "RBF Settings",           "it": "Impostazioni RBF"},
"rbf_pod_modes":     {"en": "POD modes for RBF",      "it": "Modi POD per RBF"},
"rbf_kernel":        {"en": "RBF kernel",              "it": "Nucleo RBF"},
"gp_sidebar_hdr":    {"en": "GP Settings",             "it": "Impostazioni GP"},
"gp_kernel":         {"en": "GP kernel",               "it": "Nucleo GP"},
"gp_restarts":       {"en": "Hyperparameter restarts", "it": "Riavvii iperparametri"},
"rbf_tab_predict":   {"en": "🎯 Predict New Shape",    "it": "🎯 Predici Nuova Forma"},
"rbf_tab_loo":       {"en": "📉 LOO Validation",       "it": "📉 Validazione LOO"},
"rbf_tab_kfold":     {"en": "🔁 K-Fold Validation",   "it": "🔁 Validazione K-Fold"},
"rbf_tab_conv":      {"en": "📈 Convergence Study",    "it": "📈 Studio di Convergenza"},
"rbf_tab_lhs":       {"en": "🌐 1000 Virtual Shapes",  "it": "🌐 1000 Forme Virtuali"},
"rbf_tab_demo":      {"en": "🏥 Patient Demo",         "it": "🏥 Demo Paziente"},
"rbf_tab_compare":   {"en": "⚖️ GP vs RBF",            "it": "⚖️ GP vs RBF"},
"rbf_sub_predict":   {"en": "Set Geometry Parameters → Predict Pressure Field", "it": "Imposta Parametri Geometrici → Predici il Campo di Pressione"},
"rbf_sub_loo":       {"en": "Leave-One-Out Cross-Validation",                   "it": "Validazione Incrociata Leave-One-Out"},
"rbf_sub_kfold":     {"en": "K-Fold Cross-Validation",                          "it": "Validazione Incrociata K-Fold"},
"rbf_sub_conv":      {"en": "Convergence Study — LOO Error vs Number of POD Modes", "it": "Studio di Convergenza — Errore LOO vs Numero di Modi POD"},
"rbf_sub_lhs":       {"en": "Database Upscaling: 1000 Virtual Shapes via LHS + RBF", "it": "Ampliamento Database: 1000 Forme Virtuali con LHS + RBF"},
"rbf_sub_compare":   {"en": "⚖️ Gaussian Process vs RBF — Accuracy & Speed Benchmark", "it": "⚖️ Processo Gaussiano vs RBF — Benchmark Accuratezza e Velocità"},
"rbf_btn_predict":   {"en": "🔮 Predict Pressure Field",                        "it": "🔮 Predici Campo di Pressione"},
"rbf_btn_loo":       {"en": "Run LOO validation",                               "it": "Esegui validazione LOO"},
"rbf_btn_kfold":     {"en": "Run K-Fold CV",                                    "it": "Esegui CV K-Fold"},
"rbf_btn_conv":      {"en": "Run Convergence Study",                            "it": "Esegui Studio di Convergenza"},
"rbf_btn_compare":   {"en": "▶ Run GP vs RBF Comparison",                       "it": "▶ Esegui Confronto GP vs RBF"},
"rbf_predicted_ok":  {"en": "Pressure field predicted!",                        "it": "Campo di pressione predetto!"},
"rbf_max_k_conv":    {"en": "Max modes to test",                                "it": "Massimo modi da testare"},
"rbf_cv_method":     {"en": "Validation method",                                "it": "Metodo di validazione"},
"rbf_optimal_k":     {"en": "Optimal k (min error)",                            "it": "k ottimale (errore min)"},
"rbf_min_cv_err":    {"en": "Min CV error",                                     "it": "Errore CV minimo"},
"rbf_mean_cv_err":   {"en": "Mean CV error",                                    "it": "Errore CV medio"},
"rbf_std_cv_err":    {"en": "Std CV error",                                     "it": "Dev. std errore CV"},
"rbf_best_fold":     {"en": "Best fold",                                        "it": "Fold migliore"},
"rbf_worst_fold":    {"en": "Worst fold",                                       "it": "Fold peggiore"},
"rbf_mean_loo":      {"en": "Mean LOO error",                                   "it": "Errore LOO medio"},
"rbf_max_loo":       {"en": "Max LOO error",                                    "it": "Errore LOO massimo"},
"rbf_min_loo":       {"en": "Min LOO error",                                    "it": "Errore LOO minimo"},
"rbf_std_loo":       {"en": "Std LOO error",                                    "it": "Dev. std errore LOO"},
"rbf_cv_folds":      {"en": "K-fold splits for comparison",                     "it": "Divisioni K-fold per il confronto"},
"rbf_train_time":    {"en": "RBF train",                                        "it": "Addestramento RBF"},
"gp_train_time":     {"en": "GP train",                                         "it": "Addestramento GP"},
"rbf_inference":     {"en": "RBF inference",                                    "it": "Inferenza RBF"},
"gp_inference":      {"en": "GP inference",                                     "it": "Inferenza GP"},
"gp_mean_cv_err":    {"en": "GP mean CV err",                                   "it": "Errore CV medio GP"},
"cmp_training_time": {"en": "Training Time",                                    "it": "Tempo di Addestramento"},
"cmp_kfold_acc":     {"en": "K-Fold Cross-Validation Accuracy",                 "it": "Accuratezza K-Fold"},
"cmp_inf_speed":     {"en": "Inference Speed (per single query)",                "it": "Velocità di Inferenza (per singola query)"},
"cmp_pred_field":    {"en": "Predicted Pressure Field — Mean Shape",             "it": "Campo di Pressione Predetto — Forma Media"},
"cmp_summary":       {"en": "Full Summary Table",                               "it": "Tabella Riassuntiva Completa"},
"cmp_summary_hdr":   {"en": "Summary",                                          "it": "Riepilogo"},
"cmp_winner":        {"en": "Winner",                                           "it": "Vincitore"},
"rbf_1000_shapes":   {"en": "Predicting pressure for 1000 LHS shapes…",         "it": "Predizione pressione per 1000 forme LHS…"},
"rbf_mean_dp":       {"en": "Mean ΔP (1000 shapes)",                            "it": "ΔP medio (1000 forme)"},
"rbf_max_dp":        {"en": "Max ΔP",                                           "it": "ΔP massimo"},
"rbf_min_dp":        {"en": "Min ΔP",                                           "it": "ΔP minimo"},

# Patient demo
"demo_stenosis":     {"en": "Try it: simulate a stenosed trachea",              "it": "Prova: simula una stenosi tracheale"},
"demo_trachea":      {"en": "Trachea diameter (mm)",                            "it": "Diametro tracheale (mm)"},
"demo_glottis":      {"en": "Glottis area (mm²)",                              "it": "Area della glottide (mm²)"},
"demo_high_res":     {"en": "High resistance — obstructed airway",              "it": "Alta resistenza — via aerea ostruita"},
"demo_slight_res":   {"en": "Slightly elevated resistance",                     "it": "Resistenza leggermente elevata"},
"demo_normal_res":   {"en": "Normal resistance range",                          "it": "Resistenza nella norma"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Regional Analysis
# ══════════════════════════════════════════════════════════════════════════════
"reg_title": {
    "en": "🫀 Regional Analysis — Pressure per Anatomical Region",
    "it": "🫀 Analisi Regionale — Pressione per Regione Anatomica",
},
"reg_caption": {
    "en": "Average static pressure (Pa) in each named section of the airway model, from the glottis down to the third-generation bronchial branches.",
    "it": "Pressione statica media (Pa) in ogni sezione denominata del modello delle vie aeree, dalla glottide fino ai rami bronchiali di terza generazione.",
},
"reg_sort_by":       {"en": "Sort bar chart by",      "it": "Ordina grafico per"},
"reg_sort_mean":     {"en": "Mean pressure",          "it": "Pressione media"},
"reg_sort_anat":     {"en": "Anatomical order",       "it": "Ordine anatomico"},
"reg_tab_snap":      {"en": "📍 Per-Snapshot View",   "it": "📍 Vista per Istantanea"},
"reg_tab_heat":      {"en": "🗺️ Cross-Snapshot Heatmap", "it": "🗺️ Mappa Termica Trasversale"},
"reg_tab_export":    {"en": "📥 Export CSV",          "it": "📥 Esporta CSV"},
"reg_sub_radar":     {"en": "Radar Chart",            "it": "Grafico Radar"},
"reg_sub_heatmap":   {"en": "All 100 Snapshots × Anatomical Regions", "it": "Tutte 100 Istantanee × Regioni Anatomiche"},
"reg_sub_variable":  {"en": "Most Variable Regions (across 100 runs)", "it": "Regioni più Variabili (su 100 esecuzioni)"},
"reg_sub_export":    {"en": "Export Regional Statistics",              "it": "Esporta Statistiche Regionali"},
"reg_export_sel":    {"en": "Selected snapshot regional stats",        "it": "Statistiche regionali istantanea selezionata"},
"reg_export_all":    {"en": "All snapshots × all regions (wide format)","it": "Tutte istantanee × tutte regioni (formato largo)"},
"reg_dl_sel":        {"en": "Download CSV",                            "it": "Scarica CSV"},
"reg_dl_all":        {"en": "Download full heatmap CSV (100 × regions)","it": "Scarica CSV completa (100 × regioni)"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Design Space
# ══════════════════════════════════════════════════════════════════════════════
"ds_title": {
    "en": "🗺️ Design Space — What Drives Airway Pressure?",
    "it": "🗺️ Spazio di Progetto — Cosa Guida la Pressione delle Vie Aeree?",
},
"ds_caption": {
    "en": "Explore which geometric parameters matter most and how the design space is structured.",
    "it": "Esplora quali parametri geometrici sono più importanti e come è strutturato lo spazio di progetto.",
},
"ds_tab_sens":       {"en": "📊 Parameter Sensitivity",   "it": "📊 Sensibilità Parametrica"},
"ds_tab_land":       {"en": "🌐 Design Landscape",        "it": "🌐 Paesaggio di Progetto"},
"ds_tab_grid":       {"en": "🔢 Param–Pressure Grid",     "it": "🔢 Griglia Param–Pressione"},
"ds_tab_sobol":      {"en": "🎲 Sobol Indices",           "it": "🎲 Indici di Sobol"},
"ds_sub_sensitivity":{"en": "Which parameter drives mean airway pressure the most?", "it": "Quale parametro influenza di più la pressione media delle vie aeree?"},
"ds_sub_landscape":  {"en": "Geometry POD space",         "it": "Spazio POD geometrico"},
"ds_sub_sobol":      {"en": "Variance-Based Global Sensitivity — Sobol Indices",    "it": "Sensibilità Globale Basata sulla Varianza — Indici di Sobol"},
"ds_sample_n":       {"en": "Sample size N (higher = more accurate, slower)",       "it": "Dimensione campione N (più alto = più accurato, più lento)"},
"ds_output_var":     {"en": "Output variable",            "it": "Variabile di output"},
"ds_mean_pres":      {"en": "Mean pressure",              "it": "Pressione media"},
"ds_delta_p":        {"en": "ΔP (airway resistance)",     "it": "ΔP (resistenza vie aeree)"},
"ds_top_n":          {"en": "Show top N parameters",      "it": "Mostra i primi N parametri"},
"ds_btn_sobol":      {"en": "Compute Sobol Indices",      "it": "Calcola Indici di Sobol"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — Animations
# ══════════════════════════════════════════════════════════════════════════════
"anim_title": {
    "en": "🎬 Animated POD Visualisations",
    "it": "🎬 Visualizzazioni POD Animate",
},
"anim_caption": {
    "en": "Animations are built on demand — configure the options then click **Generate**. Use the ▶ Play button inside each chart.",
    "it": "Le animazioni vengono generate su richiesta — configura le opzioni e clicca **Genera**. Usa il pulsante ▶ Riproduci nel grafico.",
},
"anim_tab_geo":      {"en": "🔬 Geometry Mode Sweep",     "it": "🔬 Scansione Modo Geometrico"},
"anim_tab_reel":     {"en": "🎞️ Pressure Snapshot Reel",  "it": "🎞️ Sequenza Istantanee Pressione"},
"anim_tab_pres":     {"en": "💨 Pressure Mode Sweep",     "it": "💨 Scansione Modo Pressione"},
"anim_sub_geo":      {"en": "Airway Geometry — POD Mode Sweep",     "it": "Geometria Vie Aeree — Scansione Modo POD"},
"anim_sub_reel":     {"en": "Pressure Field — DOE Snapshot Reel",   "it": "Campo di Pressione — Sequenza Istantanee DOE"},
"anim_sub_pres":     {"en": "Pressure Field — POD Mode Sweep",      "it": "Campo di Pressione — Scansione Modo POD"},
"anim_pod_mode":     {"en": "POD Mode",                  "it": "Modo POD"},
"anim_sigma":        {"en": "σ range",                   "it": "Intervallo σ"},
"anim_every_n":      {"en": "Show every N-th snapshot",  "it": "Mostra ogni N-esima istantanea"},
"anim_btn_geo":      {"en": "Generate Geometry Animation","it": "Genera Animazione Geometria"},
"anim_btn_reel":     {"en": "Generate Snapshot Reel",    "it": "Genera Sequenza Istantanee"},
"anim_btn_pres":     {"en": "Generate Pressure Animation","it": "Genera Animazione Pressione"},
"anim_error_precomp":{"en": "Precomputed POD files not found. Run `scripts/precompute.py --pod-only` first.", "it": "File POD pre-calcolati non trovati. Esegui prima `scripts/precompute.py --pod-only`."},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Ask AI
# ══════════════════════════════════════════════════════════════════════════════
"ask_title": {
    "en": "🤖 Ask AI — Human Airways Digital Twin Q&A",
    "it": "🤖 Chiedi all'IA — Q&A Gemello Digitale Vie Aeree Umane",
},
"ask_caption": {
    "en": "Ask anything about the project. Works fully offline — no API key needed.",
    "it": "Fai qualsiasi domanda sul progetto. Funziona completamente offline — nessuna chiave API necessaria.",
},
"ask_api_active":    {"en": "Claude API connected — AI-generated answers enabled.", "it": "API Claude connessa — risposte generate dall'IA abilitate."},
"ask_offline_sem":   {"en": "Offline mode — semantic search (sentence-transformers) + local answers.", "it": "Modalità offline — ricerca semantica + risposte locali."},
"ask_offline_tfidf": {"en": "Offline mode — TF-IDF search + local answers. All KB docs indexed.",     "it": "Modalità offline — ricerca TF-IDF + risposte locali. Tutti i documenti KB indicizzati."},
"ask_ai_hdr":        {"en": "🫁 Human Airways AI Assistant",                                          "it": "🫁 Assistente IA Vie Aeree Umane"},
"ask_voice_caption": {
    "en": "Type a question or click **🎙 Voice** (Chrome / Edge · internet required for STT). Toggle **🔊** to have answers read aloud via your browser.",
    "it": "Digita una domanda o clicca **🎙 Voce** (Chrome / Edge · internet richiesto per STT). Attiva **🔊** per ascoltare le risposte tramite il browser.",
},
"ask_tts_toggle":    {"en": "🔊 Read answers aloud",      "it": "🔊 Leggi le risposte ad alta voce"},
"ask_suggested":     {"en": "**Suggested questions**",    "it": "**Domande suggerite**"},
"ask_kb_docs":       {"en": "**Knowledge base:** {n} docs","it": "**Base di conoscenza:** {n} documenti"},
"ask_placeholder":   {"en": "Ask anything about the project…", "it": "Fai qualsiasi domanda sul progetto…"},
"ask_spinner":       {"en": "Generating answer…",         "it": "Generazione risposta…"},
"ask_sources":       {"en": "📄 Sources",                 "it": "📄 Fonti"},
"ask_btn_clear":     {"en": "🗑 Clear chat",              "it": "🗑 Cancella chat"},
"ask_voice_btn":     {"en": "🎙 Voice",                   "it": "🎙 Voce"},
"ask_stop_btn":      {"en": "⏹ Stop",                     "it": "⏹ Ferma"},
"ask_listening":     {"en": "🟢 Listening…",             "it": "🟢 In ascolto…"},
"ask_submitted":     {"en": "✅ Submitted!",              "it": "✅ Inviato!"},
"ask_stt_error":     {"en": "❌ Not supported — use Chrome or Edge", "it": "❌ Non supportato — usa Chrome o Edge"},
"ask_welcome": {
    "en": ("Hello! I'm your Human Airways Digital Twin assistant. "
           "Ask me anything about the project — POD, RBF, CFD, anatomy, "
           "validation, or how to use the dashboard.\n\n"
           "You can also use the **🎙 Voice** button to speak your question "
           "(requires internet for speech recognition)."),
    "it": ("Ciao! Sono il tuo assistente del Gemello Digitale delle Vie Aeree Umane. "
           "Chiedimi qualsiasi cosa sul progetto — POD, RBF, CFD, anatomia, "
           "validazione o come usare la dashboard.\n\n"
           "Puoi anche usare il pulsante **🎙 Voce** per fare la tua domanda "
           "(richiede internet per il riconoscimento vocale)."),
},

# Suggested questions — English
"ask_q1_en":  {"en": "What is POD and how is it used?",              "it": "Cos'è il POD e come viene utilizzato?"},
"ask_q2_en":  {"en": "Why does pressure need fewer modes than geometry?", "it": "Perché la pressione richiede meno modi della geometria?"},
"ask_q3_en":  {"en": "Which parameter affects pressure the most?",   "it": "Quale parametro influenza di più la pressione?"},
"ask_q4_en":  {"en": "How accurate is the RBF surrogate?",           "it": "Quanto è accurato il surrogato RBF?"},
"ask_q5_en":  {"en": "What is airway resistance?",                   "it": "Cos'è la resistenza delle vie aeree?"},
"ask_q6_en":  {"en": "How do Sobol indices differ from Pearson?",    "it": "In cosa differiscono gli indici di Sobol da Pearson?"},
"ask_q7_en":  {"en": "What is a Digital Twin?",                      "it": "Cos'è un Gemello Digitale?"},
"ask_q8_en":  {"en": "How many nodes are in the mesh?",              "it": "Quanti nodi contiene la maglia?"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — VR Viewer
# ══════════════════════════════════════════════════════════════════════════════
"vr_title": {
    "en": "🥽 3D / VR Viewer",
    "it": "🥽 Visualizzatore 3D / VR",
},
"vr_caption": {
    "en": "Interactive 3D airway viewer. **Enter VR** button appears on WebXR-compatible devices (Meta Quest, etc.).",
    "it": "Visualizzatore 3D interattivo delle vie aeree. Il pulsante **Entra in VR** appare su dispositivi compatibili WebXR (Meta Quest, ecc.).",
},
"vr_layers":         {"en": "Layers",                  "it": "Livelli"},
"vr_show_mesh":      {"en": "Show airway mesh",        "it": "Mostra maglia vie aeree"},
"vr_show_cloud":     {"en": "Show pressure cloud",     "it": "Mostra nuvola di pressione"},
"vr_dark_bg":        {"en": "Dark background",         "it": "Sfondo scuro"},
"vr_colour_pres":    {"en": "Pressure (Pa)",           "it": "Pressione (Pa)"},
"vr_colour_z":       {"en": "Height (Z)",              "it": "Altezza (Z)"},
"vr_colour_idx":     {"en": "Node index",              "it": "Indice nodo"},
"vr_dl_html":        {"en": "⬇️ Download standalone HTML", "it": "⬇️ Scarica HTML autonomo"},
"vr_warn_missing":   {"en": "Snapshot geometry not found — showing reference mesh with zero pressure.", "it": "Geometria istantanea non trovata — visualizzazione maglia di riferimento con pressione zero."},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — Mesh Viewer
# ══════════════════════════════════════════════════════════════════════════════
"mesh_title": {
    "en": "🔲 Surface Mesh Viewer",
    "it": "🔲 Visualizzatore Maglia Superficiale",
},
"mesh_caption": {
    "en": "The airway rendered as a solid triangulated surface. Pressure is mapped onto the surface so you can see exactly where resistance builds up.",
    "it": "Le vie aeree rese come superficie triangolata solida. La pressione è mappata sulla superficie per vedere esattamente dove si accumula la resistenza.",
},
"mesh_tab_surface":  {"en": "🔲 Surface Mesh",         "it": "🔲 Maglia Superficiale"},
"mesh_tab_compare":  {"en": "⚖️ Mesh vs Point Cloud",  "it": "⚖️ Maglia vs Nuvola di Punti"},
"mesh_tab_how":      {"en": "ℹ️ How the mesh is built", "it": "ℹ️ Come viene costruita la maglia"},
"mesh_vertices":     {"en": "Mesh vertices",           "it": "Vertici della maglia"},
"mesh_faces":        {"en": "Mesh faces",              "it": "Facce della maglia"},
"mesh_overlay":      {"en": "Overlay point cloud",     "it": "Sovrapponi nuvola di punti"},
"mesh_stl_loaded":   {"en": "Pre-computed STL loaded", "it": "STL pre-calcolato caricato"},
"mesh_stl_fly":      {"en": "Generating mesh on-the-fly (cached after first load)", "it": "Generazione maglia al volo (memorizzata al primo caricamento)"},
"mesh_dl_stl":       {"en": "Download STL for this snapshot", "it": "Scarica STL per questa istantanea"},
"mesh_warn":         {"en": "Mesh unavailable — showing point cloud.", "it": "Maglia non disponibile — visualizzazione nuvola di punti."},
"mesh_sub_how":      {"en": "From point cloud to mesh",      "it": "Dalla nuvola di punti alla maglia"},
"mesh_sub_quality":  {"en": "Mesh quality note",             "it": "Nota sulla qualità della maglia"},
"mesh_sub_compress": {"en": "Mesh compression ratio",        "it": "Rapporto di compressione della maglia"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — Delivery Packages
# ══════════════════════════════════════════════════════════════════════════════
"pkg_title": {
    "en": "📦 Delivery Packages",
    "it": "📦 Pacchetti di Consegna",
},
"pkg_caption": {
    "en": "Three specialised data packages produced from this digital twin. Each targets a different downstream team and workflow.",
    "it": "Tre pacchetti dati specializzati prodotti da questo gemello digitale. Ognuno è destinato a un team e un flusso di lavoro diverso.",
},
"pkg_tab_vr":        {"en": "🥽 VR Archive",           "it": "🥽 Archivio VR"},
"pkg_tab_helyx":     {"en": "🏗️ HELYX Package",        "it": "🏗️ Pacchetto HELYX"},
"pkg_tab_domino":    {"en": "🧪 Domino Dataset",        "it": "🧪 Dataset Domino"},
"pkg_what":          {"en": "What is it?",              "it": "Cos'è?"},
"pkg_contents":      {"en": "Package contents",         "it": "Contenuto del pacchetto"},
"pkg_snapshots":     {"en": "Snapshots",                "it": "Istantanee"},
"pkg_nodes":         {"en": "Nodes / snapshot",         "it": "Nodi / istantanea"},
"pkg_stride":        {"en": "Stride",                   "it": "Passo"},
"pkg_virtual":       {"en": "Virtual shapes",           "it": "Forme virtuali"},
"pkg_geo_params":    {"en": "Geometry parameters",      "it": "Parametri geometrici"},
"pkg_pres_modes":    {"en": "Pressure POD modes",       "it": "Modi POD pressione"},
"pkg_mean_dp":       {"en": "Mean DeltaP",              "it": "ΔP medio"},
"pkg_dl_vr":         {"en": "Download vr_archive.zip",  "it": "Scarica vr_archive.zip"},
"pkg_dl_helyx":      {"en": "Download helyx_package.zip","it": "Scarica helyx_package.zip"},
"pkg_dl_domino":     {"en": "Download domino_dataset.zip","it": "Scarica domino_dataset.zip"},
"pkg_vr_ready":      {"en": "**VR Archive** — ready",              "it": "**Archivio VR** — pronto"},
"pkg_vr_run":        {"en": "**VR Archive** — run `export_vr_archive.py`", "it": "**Archivio VR** — esegui `export_vr_archive.py`"},
"pkg_helyx_ready":   {"en": "**HELYX Package** — ready",           "it": "**Pacchetto HELYX** — pronto"},
"pkg_helyx_run":     {"en": "**HELYX Package** — run `export_helyx.py`",   "it": "**Pacchetto HELYX** — esegui `export_helyx.py`"},
"pkg_domino_ready":  {"en": "**Domino Dataset** — ready",          "it": "**Dataset Domino** — pronto"},
"pkg_domino_run":    {"en": "**Domino Dataset** — run `export_domino.py`", "it": "**Dataset Domino** — esegui `export_domino.py`"},
"pkg_vr_cap":        {"en": "Standalone WebXR point-cloud viewer for all 100 snapshots", "it": "Visualizzatore WebXR autonomo per le 100 istantanee"},
"pkg_helyx_cap":     {"en": "STL meshes + parameter tables for OpenFOAM / snappyHexMesh", "it": "Maglie STL + tabelle parametri per OpenFOAM / snappyHexMesh"},
"pkg_domino_cap":    {"en": "1000-shape ML dataset with POD bases and pressure fields",   "it": "Dataset ML 1000 forme con basi POD e campi di pressione"},
"pkg_files_archive": {"en": "**Files in archive:**",               "it": "**File nell'archivio:**"},
"pkg_actual_stl":    {"en": "Actual STL meshes",                   "it": "Maglie STL reali"},
"pkg_virtual_stl":   {"en": "Virtual STL meshes",                  "it": "Maglie STL virtuali"},
"pkg_geo_params_drive":{"en": "Which geometry parameters drive airway resistance?", "it": "Quali parametri geometrici influenzano la resistenza delle vie aeree?"},
"pkg_corr_caption":  {"en": "Pearson correlation between each parameter and DeltaP across all 1000 shapes.", "it": "Correlazione di Pearson tra ogni parametro e ΔP su tutte le 1000 forme."},
"pkg_dataset_files": {"en": "Dataset files",                       "it": "File del dataset"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 13 — AI Surrogate
# ══════════════════════════════════════════════════════════════════════════════
"ais_title": {
    "en": "🤖 AI Surrogate — NVIDIA PhysicsNeMo",
    "it": "🤖 Surrogato IA — NVIDIA PhysicsNeMo",
},
"ais_caption": {
    "en": "A neural network trained with NVIDIA PhysicsNeMo predicts the full 3D airway pressure field in milliseconds — 1000 virtual patients, no CFD required.",
    "it": "Una rete neurale addestrata con NVIDIA PhysicsNeMo predice il campo di pressione 3D completo in millisecondi — 1000 pazienti virtuali, nessun CFD necessario.",
},
"ais_tab_what":      {"en": "📖 What is DoMINO?",       "it": "📖 Cos'è DoMINO?"},
"ais_tab_train":     {"en": "🧪 Training Journey",      "it": "🧪 Percorso di Addestramento"},
"ais_tab_predict":   {"en": "🔬 Live Prediction",       "it": "🔬 Predizione Live"},
"ais_tab_1000":      {"en": "🌐 1000 Virtual Airways",  "it": "🌐 1000 Vie Aeree Virtuali"},
"ais_tab_helyx":     {"en": "⚙️ HELYX Delivery",        "it": "⚙️ Consegna HELYX"},
"ais_sub_domino":    {"en": "DoMINO — Decomposable Multi-scale Iterative Neural Operator", "it": "DoMINO — Operatore Neurale Iterativo Multi-scala Decomponibile"},
"ais_sub_impl":      {"en": "Our implementation",                                         "it": "La nostra implementazione"},
"ais_sub_train":     {"en": "Three steps to get from 21.8% error to 4.3%",               "it": "Tre passi per passare dal 21.8% al 4.3% di errore"},
"ais_sub_predict":   {"en": "Predict pressure on any of the 100 real CFD snapshots",      "it": "Predici la pressione su una qualsiasi delle 100 istantanee CFD reali"},
"ais_snap_slider":   {"en": "Snapshot (1–100)",         "it": "Istantanea (1–100)"},
"ais_true_dp":       {"en": "True ΔP",                  "it": "ΔP reale"},
"ais_pred_dp":       {"en": "Predicted ΔP",             "it": "ΔP predetto"},
"ais_rmse":          {"en": "RMSE",                     "it": "RMSE"},
"ais_rel_err":       {"en": "Relative error",           "it": "Errore relativo"},
"ais_shapes":        {"en": "Shapes inferred",          "it": "Forme inferite"},
"ais_inf_time":      {"en": "Inference time",           "it": "Tempo di inferenza"},
"ais_dp_range":      {"en": "ΔP range (NN)",            "it": "Intervallo ΔP (NN)"},
"ais_mean_dp":       {"en": "Mean ΔP (NN)",             "it": "ΔP medio (NN)"},
"ais_speedup":       {"en": "Speedup vs CFD",           "it": "Accelerazione vs CFD"},
"ais_warn_no_model": {"en": "No trained model found. Run `python notebooks/physicsnemo_train.py` first.", "it": "Nessun modello addestrato trovato. Esegui prima `python notebooks/physicsnemo_train.py`."},
"ais_spinner_1000":  {"en": "Running inference on 1000 shapes…",  "it": "Inferenza su 1000 forme…"},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 14 — Drug Deposition
# ══════════════════════════════════════════════════════════════════════════════
"drug_title": {
    "en": "💊 Drug Particle Deposition Simulator",
    "it": "💊 Simulatore di Deposizione di Particelle Farmaceutiche",
},
"drug_caption": {
    "en": "Predicts where inhaled drug particles deposit in the airway. Combines the RBF pressure surrogate with respiratory deposition physics.",
    "it": "Predice dove si depositano le particelle del farmaco inalato nelle vie aeree. Combina il surrogato RBF con la fisica della deposizione respiratoria.",
},
"drug_hdr_patient":  {"en": "Patient",                 "it": "Paziente"},
"drug_hdr_airflow":  {"en": "Airflow",                 "it": "Flusso d'aria"},
"drug_hdr_particle": {"en": "Particle",                "it": "Particella"},
"drug_velocity":     {"en": "Inlet velocity (m/s)",    "it": "Velocità ingresso (m/s)"},
"drug_diameter":     {"en": "Particle diameter (µm)",  "it": "Diametro particella (µm)"},
"drug_inject_v":     {"en": "Injection velocity (m/s)","it": "Velocità iniezione (m/s)"},
"drug_density":      {"en": "Particle density (kg/m³)","it": "Densità particella (kg/m³)"},
"drug_tab_map":      {"en": "🌡️ Deposition Map",        "it": "🌡️ Mappa di Deposizione"},
"drug_tab_region":   {"en": "📊 Regional Breakdown",   "it": "📊 Analisi Regionale"},
"drug_tab_optimal":  {"en": "📈 Optimal Particle Size", "it": "📈 Dimensione Ottimale Particella"},
"drug_tab_physics":  {"en": "🔬 Physics Model",        "it": "🔬 Modello Fisico"},
"drug_overall":      {"en": "Overall deposition",      "it": "Deposizione totale"},
"drug_upper":        {"en": "Upper airway (throat)",   "it": "Vie aeree superiori (gola)"},
"drug_trachea":      {"en": "Trachea",                 "it": "Trachea"},
"drug_bronchi":      {"en": "Bronchi",                 "it": "Bronchi"},
"drug_efficiency":   {"en": "Delivery efficiency score","it": "Punteggio efficienza consegna"},
"drug_impaction":    {"en": "Impaction mean",          "it": "Media impazione"},
"drug_sedimentation":{"en": "Sedimentation mean",     "it": "Media sedimentazione"},
"drug_diffusion":    {"en": "Diffusion mean",          "it": "Media diffusione"},
"drug_mech_breakdown":{"en": "##### Mechanism breakdown (node means)", "it": "##### Analisi meccanismi (medie nodo)"},
"drug_optimal_intro":{"en": "Sweep particle size from **0.5 µm to 20 µm** and see how each deposition mechanism and the overall bronchial delivery efficiency change.", "it": "Esplora la dimensione delle particelle da **0,5 µm a 20 µm** e osserva come cambiano i meccanismi di deposizione e l'efficienza di consegna bronchiale."},

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 15 — Patient Comparison
# ══════════════════════════════════════════════════════════════════════════════
"cmp2_title": {
    "en": "⚖️ Two-Patient Airway Comparison",
    "it": "⚖️ Confronto delle Vie Aeree tra Due Pazienti",
},
"cmp2_caption": {
    "en": "Compare airway pressure fields, resistance, and drug delivery for two patients side by side.",
    "it": "Confronta i campi di pressione, la resistenza e la consegna del farmaco per due pazienti affiancati.",
},
"cmp2_patient_a":    {"en": "Patient A",               "it": "Paziente A"},
"cmp2_patient_b":    {"en": "Patient B",               "it": "Paziente B"},
"cmp2_input_mode":   {"en": "Input mode",              "it": "Modalità input"},
"cmp2_doe_snap":     {"en": "DOE Snapshot",            "it": "Istantanea DOE"},
"cmp2_custom_rbf":   {"en": "Custom RBF",              "it": "RBF personalizzato"},
"cmp2_snap_a":       {"en": "Snapshot A",              "it": "Istantanea A"},
"cmp2_snap_b":       {"en": "Snapshot B",              "it": "Istantanea B"},
"cmp2_delta_map":    {"en": "Show delta map",          "it": "Mostra mappa delta"},
"cmp2_tab_3d":       {"en": "🫁 3D Pressure Fields",   "it": "🫁 Campi di Pressione 3D"},
"cmp2_tab_delta":    {"en": "🌡️ Delta Pressure Map",   "it": "🌡️ Mappa Pressione Delta"},
"cmp2_tab_regional": {"en": "📊 Regional Comparison",  "it": "📊 Confronto Regionale"},
"cmp2_tab_params":   {"en": "📐 Parameter Comparison", "it": "📐 Confronto Parametri"},
"cmp2_dp_a":         {"en": "ΔP  A (Pa)",              "it": "ΔP  A (Pa)"},
"cmp2_dp_b":         {"en": "ΔP  B (Pa)",              "it": "ΔP  B (Pa)"},
"cmp2_resist_a":     {"en": "Resist. A (%)",           "it": "Resist. A (%)"},
"cmp2_resist_b":     {"en": "Resist. B (%)",           "it": "Resist. B (%)"},
"cmp2_max_diff":     {"en": "Max ΔP diff",             "it": "Diff. ΔP massima"},
"cmp2_higher":       {"en": "Higher resistance",       "it": "Resistenza maggiore"},
"cmp2_spinner":      {"en": "Running RBF…",            "it": "Esecuzione RBF…"},

}  # end _T

# ── Italian RAG documentation strings (added to core/rag.py separately) ───────
# (Defined here for reference; actual insertion is in core/rag.py)

# ── Public API ─────────────────────────────────────────────────────────────────

def get_lang() -> str:
    """Return the active language code ('en' or 'it')."""
    return st.session_state.get("lang", "en")


def t(key: str, **kwargs) -> str:
    """
    Return the translated string for *key* in the current language.
    Falls back to English if the key or language is missing.
    Supports .format()-style kwargs: t("ask_kb_docs", n=29)
    """
    lang = get_lang()
    entry = _T.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def lang_selector() -> None:
    """
    Render the language radio button in the sidebar.
    Call once at the top of every page (after set_page_config).
    """
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    with st.sidebar:
        st.radio(
            t("lang_label"),
            options=["en", "it"],
            format_func=lambda x: "🇬🇧 English" if x == "en" else "🇮🇹 Italiano",
            key="lang",
            horizontal=True,
        )
        st.divider()
