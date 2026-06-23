"""
Dashboard del Modelo Predictivo Mundial 2026 (Streamlit).

Ejecutar:  streamlit run app/dashboard.py

Pestañas:
  🗓️ Fixture · 🗺️ Torneo · 🏆 Llaves · 👥 Selecciones · ⚽ Partido
  📊 Ranking · 🎲 Simular · 🗞️ Datos & noticias
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, all_teams, canonical, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from src.models.ensemble import EnsemblePredictor  # noqa: E402
from src.data.sources import load_fixture, load_standings, head_to_head  # noqa: E402
from src.data.squads import load_squads, load_coaches  # noqa: E402
from src.data.squad_features import squad_metrics  # noqa: E402
from src.data.player_ratings import (  # noqa: E402
    rated_squad, assign_to_formation, lineup_strength, team_lineup_baseline,
    best_xi, FORMATIONS)
from app.pitch import pitch_match_svg, pitch_team_svg  # noqa: E402
from app.teams_visual import flag_img, team_color  # noqa: E402

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, select { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
.block-container {padding-top: 1.4rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}

/* Hero header */
.hero {background: linear-gradient(125deg, #0e7a3b 0%, #115e59 45%, #1e293b 100%);
       border-radius: 16px; padding: 22px 28px; margin: 0 0 20px;
       box-shadow: 0 10px 34px rgba(0,0,0,.4); border:1px solid rgba(255,255,255,.07);
       position:relative; overflow:hidden;}
.hero::after {content:"⚽"; position:absolute; right:24px; top:50%; transform:translateY(-50%);
       font-size:5.5rem; opacity:.10;}
.hero h1 {margin:0; font-size:1.95rem; font-weight:800; color:#fff; letter-spacing:-.6px;}
.hero p {margin:5px 0 0; color:#d1e0d8; font-size:.92rem; max-width:760px;}
.hero .chips {display:flex; gap:9px; margin-top:15px; flex-wrap:wrap;}
.hero .chip {background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.16);
       border-radius:999px; padding:5px 14px; font-size:.8rem; color:#f1f5f9;}
.hero .chip b {color:#fff; font-weight:700;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap:3px; border-bottom:1px solid rgba(255,255,255,.07);}
.stTabs [data-baseweb="tab"] {border-radius:9px 9px 0 0; padding:9px 15px; font-weight:600; font-size:.9rem;}
.stTabs [aria-selected="true"] {background:rgba(22,163,74,.18); color:#4ade80 !important;}

/* Metrics como tarjetas */
[data-testid="stMetric"] {background:#131c2e; border:1px solid rgba(255,255,255,.06);
       border-radius:12px; padding:13px 16px;}
[data-testid="stMetricValue"] {font-size:1.55rem; font-weight:800; letter-spacing:-.5px;}
[data-testid="stMetricLabel"] {opacity:.75;}

/* Botones */
.stButton button, .stDownloadButton button {border-radius:10px; font-weight:600; border:1px solid rgba(255,255,255,.1);}
.stButton button[kind="primary"] {background:#16a34a; border:none;}
hr {margin:1.1rem 0; border-color:rgba(255,255,255,.07);}
h3 {letter-spacing:-.3px;}

/* Partidos del fixture */
.match {display:flex; align-items:center; gap:10px; padding:9px 14px; margin:6px 0;
        border:1px solid rgba(255,255,255,.07); border-radius:12px; background:#101a2c;
        transition:border-color .15s, transform .15s;}
.match:hover {border-color:rgba(22,163,74,.4); transform:translateX(2px);}
.match .grp {font-size:.68rem; font-weight:800; background:#16a34a; color:#06140c;
        border-radius:6px; padding:2px 7px; min-width:26px; text-align:center;}
.match .home {flex:1; text-align:right; font-weight:600;}
.match .away {flex:1; text-align:left; font-weight:600;}
.match .mid {min-width:155px; text-align:center;}
.score {font-size:1.2rem; font-weight:800; letter-spacing:1px;}
.score.win-h, .score.win-a {color:#4ade80;}
.pbar {display:flex; height:8px; border-radius:5px; overflow:hidden; margin:4px 0;}
.pbar .h {background:#2563eb;} .pbar .d {background:#64748b;} .pbar .a {background:#dc2626;}
.pct {font-size:.72rem; color:#94a3b8; display:flex; justify-content:space-between;}
.xg {font-size:.7rem; color:#64748b; text-align:center;}
.daterow {font-size:.78rem; font-weight:700; color:#4ade80; margin:16px 0 4px;
        text-transform:uppercase; letter-spacing:.6px;}

/* Bracket */
.bracket {display:flex; gap:16px; overflow-x:auto; padding:12px 4px;}
.round {display:flex; flex-direction:column; justify-content:space-around; min-width:160px; gap:8px;}
.rtitle {font-size:.72rem; font-weight:800; color:#4ade80; text-transform:uppercase;
        text-align:center; margin-bottom:4px; letter-spacing:.6px;}
.bmatch {border:1px solid rgba(255,255,255,.08); border-radius:9px; overflow:hidden; background:#101a2c;}
.bteam {padding:6px 10px; font-size:.8rem; display:flex; justify-content:space-between; gap:8px;}
.bteam.win {background:rgba(22,163,74,.18); font-weight:700;}
.bteam.lose {opacity:.5;}
.vs {text-align:center; font-weight:800; color:#94a3b8; font-size:1.05rem; margin-bottom:6px;}

/* Cuadro de dos lados (llaves) */
.bracket2 {display:flex; align-items:stretch; gap:6px; overflow-x:auto; padding:10px 2px;}
.side {display:flex; gap:8px;}
.col {display:flex; flex-direction:column; justify-content:space-around; min-width:132px; gap:6px;}
.col .ct {font-size:.62rem; font-weight:800; color:#4ade80; text-transform:uppercase;
        text-align:center; letter-spacing:.5px; margin-bottom:2px;}
.tie {border:1px solid rgba(255,255,255,.08); border-radius:8px; overflow:hidden; background:#101a2c;}
.tie .t {padding:4px 7px; font-size:.74rem; display:flex; align-items:center; gap:5px;}
.tie .t.w {background:rgba(22,163,74,.2); font-weight:700;}
.tie .t.l {opacity:.45;}
.tie .t .pl {font-size:.6rem; color:#64748b; margin-left:auto;}
.center {display:flex; flex-direction:column; justify-content:center; align-items:center;
        min-width:150px; gap:6px; padding:0 4px;}
.center .ftitle {font-size:.72rem; font-weight:800; color:#f59e0b; text-transform:uppercase; letter-spacing:1px;}
.center .champ {margin-top:8px; text-align:center; font-weight:800; font-size:1.05rem;
        background:linear-gradient(135deg,#f59e0b,#b45309); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; background-clip:text;}
.cruce {display:flex; gap:10px; align-items:center; padding:6px 10px; margin:3px 0;
        border:1px solid rgba(255,255,255,.07); border-radius:9px; background:#101a2c; font-size:.85rem;}
.cruce .gl {font-weight:800; background:#16a34a; color:#06140c; border-radius:6px; padding:2px 8px;}
.cruce .opt {flex:1;}
</style>
""", unsafe_allow_html=True)

MESES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando modelos (primer arranque puede tardar) ...")
def get_ensemble():
    p = MODELS_DIR / "ensemble.joblib"
    if not p.exists():
        # Bootstrap para deploys limpios (Streamlit Cloud): entrena si falta.
        try:
            from src.pipeline import train_all
            train_all(fetch=True, fetch_news=False)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudieron entrenar los modelos automáticamente: {e}")
            return None
    return EnsemblePredictor.load(p) if p.exists() else None


@st.cache_data
def get_config():
    return load_config()


@st.cache_data(show_spinner=False)
def get_fixture():
    return load_fixture()


@st.cache_data(show_spinner=False)
def get_standings():
    return load_standings()


@st.cache_data(show_spinner=False)
def get_squads():
    return load_squads()


@st.cache_data(show_spinner=False)
def get_squad_metrics():
    return squad_metrics()


@st.cache_data(show_spinner=False)
def get_coaches():
    return load_coaches()


@st.cache_data(show_spinner=False)
def predict_cached(home, away, neutral=True):
    """Predicción memoizada por (home, away): el ensemble es inmutable entre reruns."""
    return get_ensemble().predict(home, away, neutral=neutral)


@st.cache_data(show_spinner=False)
def market_probs_map():
    """{(home, away): (p_home, p_draw, p_away)} de las cuotas guardadas (sin gastar API)."""
    from src.evaluation.odds_benchmark import ODDS_CSV, implied_probs
    if not ODDS_CSV.exists():
        return {}
    df = pd.read_csv(ODDS_CSV, comment="#").dropna(subset=["home_team", "away_team"])
    out = {}
    for o in df.itertuples(index=False):
        try:
            out[(canonical(o.home_team, get_config()), canonical(o.away_team, get_config()))] = \
                implied_probs(o.odds_home, o.odds_draw, o.odds_away)
        except Exception:  # noqa: BLE001
            continue
    return out


def predict_mode(home, away, mode="🤖 Modelo", w=0.5):
    """Predicción según la fuente elegida. El mercado solo cubre partidos con cuota;
    sin cuota cae al modelo. Devuelve (dict_pred, tiene_mercado)."""
    base = dict(predict_cached(home, away))
    mk = market_probs_map().get((home, away))
    if mode.endswith("Modelo") or mk is None:
        return base, (mk is not None)
    ph, pd_, pa = mk
    if mode.endswith("Mercado"):
        base.update(p_home=ph, p_draw=pd_, p_away=pa)
        return base, True
    # Blend: w*modelo + (1-w)*mercado, renormalizado
    bh = w * base["p_home"] + (1 - w) * ph
    bd = w * base["p_draw"] + (1 - w) * pd_
    ba = w * base["p_away"] + (1 - w) * pa
    s = bh + bd + ba
    base.update(p_home=bh / s, p_draw=bd / s, p_away=ba / s)
    return base, True


@st.cache_resource(show_spinner=False)
def get_simulator(mode="modelo", blend_w=0.5):
    from src.simulation.tournament import TournamentSimulator
    return TournamentSimulator(get_ensemble(), get_config(),
                               market=market_probs_map(), mode=mode, blend_w=blend_w)


@st.cache_data(show_spinner=False)
def get_bracket(mode="modelo", blend_w=0.5):
    return get_simulator(mode, blend_w).official_bracket()


@st.cache_data(show_spinner=False)
def get_h2h(a, b):
    return head_to_head(a, b)


@st.cache_data(show_spinner=False)
def get_tracking():
    from src.evaluation.tracking import evaluate_tracking
    return evaluate_tracking()


@st.cache_data(show_spinner=False)
def get_tilts():
    from src.evaluation.backtest import _load, evaluate_tilts
    return evaluate_tilts(_load())


@st.cache_data(show_spinner=False)
def get_wc_backtest():
    """Backtest out-of-sample del Mundial: entrena hasta el 11/6 y predice lo jugado."""
    from src.evaluation.backtest import (_load, train_ensemble_asof, per_match_table,
                                         compare_models)
    m = _load()
    test = m[m["tournament"].str.contains("FIFA World Cup", na=False)
             & ~m["tournament"].str.contains("qualif", case=False, na=False)]
    test = test[test["date"] >= "2026-06-11"]
    if test.empty:
        return None
    bt_ens = train_ensemble_asof(m, "2026-06-11")
    return {"compare": compare_models(bt_ens, test), "per_match": per_match_table(bt_ens, test),
            "n": len(test)}


@st.cache_data(show_spinner=False)
def get_calibration():
    """Calibración + ablation sobre el holdout de 12 meses (muestra grande, señal real)."""
    import pandas as _pd
    from src.evaluation.backtest import (_load, train_ensemble_asof, compare_models,
                                         calibration_table)
    m = _load()
    cut = m["date"].max() - _pd.DateOffset(months=12)
    test = m[(m["date"] >= cut) & (m["date"] < "2026-06-11")]
    test = test[test["match_weight"] >= 2.0]
    ens_h = train_ensemble_asof(m, cut.strftime("%Y-%m-%d"))
    return {"compare": compare_models(ens_h, test), "calib": calibration_table(ens_h, test),
            "n": len(test)}


@st.cache_data(show_spinner=False)
def get_rated(team):
    return rated_squad(team)


@st.cache_data(show_spinner=False)
def get_baseline(team, formation):
    return team_lineup_baseline(team, formation)


@st.cache_resource(show_spinner="Entrenando el modelo cuántico (primer arranque) ...")
def get_quantum_model():
    """Carga el modelo cuántico entrenado (multiclase 1-X-2)."""
    from src.models.quantum import QuantumMatchClassifier
    p = MODELS_DIR / "quantum.joblib"
    if not p.exists():
        # Bootstrap: entrena el cuántico + exporta el JSON si faltan.
        try:
            import quantum_match
            quantum_match.main()
        except Exception:  # noqa: BLE001
            return None
    return QuantumMatchClassifier.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def get_quantum():
    """Carga features por selección + grilla de decisión para la visualización."""
    import json
    p = PROCESSED_DIR / "quantum_demo.json"
    if not p.exists():
        return None
    qd = json.loads(p.read_text(encoding="utf-8"))
    qd["feat"] = {t["name"]: t for t in qd["teams"]}
    if not qd.get("zmax"):
        qd["zmax"] = max(abs(v) for row in qd["z"] for v in row) or 1.0
    return qd


def quantum_probs(model, qd, home, away):
    """Probabilidades cuánticas {H, D, A} para el partido (None si falta data)."""
    from src.models.quantum import FEATURES
    f = qd["feat"]
    if model is None or home not in f or away not in f:
        return None
    h, a = f[home], f[away]
    vals = {
        "elo_diff": h["elo"] - a["elo"],
        "form_pts_diff": h["form_pts"] - a["form_pts"],
        "form_gf_diff": h["form_gf"] - a["form_gf"],
        "form_ga_diff": h["form_ga"] - a["form_ga"],
        "rest_diff": 0.0,   # WC: se asume descanso parejo
        "neutral": 1.0,     # WC: cancha neutral
    }
    return model.predict_proba([vals[name] for name in FEATURES])


def last_update():
    p = PROCESSED_DIR / "last_update.txt"
    return p.read_text(encoding="utf-8")[:19].replace("T", " ") if p.exists() else "—"


def _fmt_date(d):
    return f"{d.day} {MESES[d.month]} {d.year}"


def build_xi(team, formation, chosen=None):
    pool = get_rated(team)
    if pool.empty:
        return pool
    if chosen:
        df = pool[pool["player"].isin(chosen)]
        if len(df) < 11:
            extra = pool[~pool["player"].isin(chosen)].head(11 - len(df))
            df = pd.concat([df, extra])
    else:
        df = pool
    return assign_to_formation(df, formation)


cfg = get_config()
ens = get_ensemble()

# Hero header con estado del torneo
_chips = ""
if ens is not None:
    _fx = get_fixture()
    if not _fx.empty:
        _jug = int(_fx["played"].sum())
        _chips = (f'<span class="chip">🏟️ <b>{_jug}</b> jugados</span>'
                  f'<span class="chip">📅 <b>{len(_fx) - _jug}</b> por jugar</span>'
                  f'<span class="chip">🔄 actualizado {last_update()[:10]}</span>')
st.markdown(
    f'<div class="hero"><h1>⚽ Predictor Mundial 2026</h1>'
    f'<p>Ensemble Elo · Poisson (Dixon-Coles) · XGBoost — predicciones, alineaciones en cancha, '
    f'llaves y benchmark medible contra el mercado.</p>'
    f'<div class="chips">{_chips}</div></div>', unsafe_allow_html=True)

if ens is None:
    st.warning("No hay modelos entrenados. Corré primero: `python -m src.pipeline`")
    st.stop()

# Selector global de fuente de predicción
with st.sidebar:
    st.markdown("### 🎚️ Fuente de predicción")
    MODE = st.radio("Modo", ["🤖 Modelo", "🔀 Blend", "🏦 Mercado"], index=0,
                    help="Aplica a las predicciones de partidos (Fixture y Partido).")
    BLEND_W = 0.5
    if MODE.endswith("Blend"):
        BLEND_W = st.slider("Peso del modelo", 0.0, 1.0, 0.5, 0.1)
        st.caption(f"{BLEND_W:.0%} modelo · {1-BLEND_W:.0%} mercado")
    st.caption("El **mercado** son las cuotas reales (the-odds-api); solo cubre partidos "
               "próximos con cuota. Sin cuota → cae al modelo.")
    n_mkt = len(market_probs_map())
    st.caption(f"📊 {n_mkt} partidos con cuota de mercado disponibles.")

MODE_KEY = MODE.split()[-1].lower()   # "modelo" | "blend" | "mercado"
teams = sorted(all_teams(cfg))
tabs = st.tabs(["⚛️ Cuántico", "🗓️ Fixture", "🗺️ Torneo", "🏆 Llaves", "👥 Selecciones",
                "⚽ Partido", "📊 Ranking", "🎲 Simular",
                "📈 Rendimiento", "🗞️ Datos"])
(tab_quantum, tab_fix, tab_tourn, tab_brkt, tab_teams, tab_match, tab_rank,
 tab_sim, tab_perf, tab_data) = tabs


def _theme_fig(fig, h=None):
    """Aplica la paleta del dashboard a un gráfico Plotly (fondo transparente, fuente)."""
    has_title = bool(getattr(fig.layout.title, "text", None))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1", family="Inter, sans-serif"),
                      margin=dict(l=10, r=10, t=34 if has_title else 10, b=10))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    if h:
        fig.update_layout(height=h)
    return fig


def _match_html(row, pred):
    grp = "" if row.group == "—" else row.group
    if row.played:
        hs, as_ = int(row.home_score), int(row.away_score)
        cls = "win-h" if hs > as_ else ("win-a" if as_ > hs else "")
        mid = f'<span class="score {cls}">{hs} - {as_}</span>'
    else:
        ph, pd_, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
        mid = (f'<div class="pbar"><div class="h" style="width:{ph*100:.0f}%"></div>'
               f'<div class="d" style="width:{pd_*100:.0f}%"></div>'
               f'<div class="a" style="width:{pa*100:.0f}%"></div></div>'
               f'<div class="pct"><span>{ph*100:.0f}%</span><span>{pd_*100:.0f}%</span>'
               f'<span>{pa*100:.0f}%</span></div>'
               f'<div class="xg">xG {pred["xg_home"]}–{pred["xg_away"]} · prob {pred["likely_score"]}</div>')
    return (f'<div class="match"><span class="grp">{grp}</span>'
            f'<span class="home">{row.home} {flag_img(row.home, 22)}</span>'
            f'<span class="mid">{mid}</span>'
            f'<span class="away">{flag_img(row.away, 22)} {row.away}</span></div>')


# ============================ FIXTURE ====================================== #
with tab_fix:
    fx = get_fixture()
    if fx.empty:
        st.info("No hay fixture. Corré `python -m src.pipeline`.")
    else:
        played = int(fx["played"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidos", len(fx))
        c2.metric("Jugados", played)
        c3.metric("Pendientes", len(fx) - played)
        c4.metric("Última actualización", last_update()[:10])

        # Exportar fixture + predicciones a CSV
        exp = []
        for r in fx.itertuples(index=False):
            row = {"fecha": r.date.date().isoformat(), "grupo": r.group,
                   "local": r.home, "visitante": r.away}
            if r.played:
                row["resultado"] = f"{int(r.home_score)}-{int(r.away_score)}"
            else:
                p = predict_cached(r.home, r.away)
                row.update(p_local=round(p["p_home"], 3), empate=round(p["p_draw"], 3),
                           p_visitante=round(p["p_away"], 3),
                           xg=f"{p['xg_home']}-{p['xg_away']}", marcador_prob=p["likely_score"])
            exp.append(row)
        st.download_button("⬇️ Descargar fixture + predicciones (CSV)",
                           pd.DataFrame(exp).to_csv(index=False).encode("utf-8"),
                           file_name="wc26_fixture_predicciones.csv", mime="text/csv")

        f1, f2 = st.columns([2, 3])
        groups = ["Todos"] + sorted(g for g in fx["group"].unique() if g != "—")
        sel_g = f1.selectbox("Grupo", groups)
        view = f2.radio("Ver", ["Todos", "Solo pendientes", "Solo jugados"],
                        horizontal=True, label_visibility="collapsed")
        data = fx.copy()
        if sel_g != "Todos":
            data = data[data["group"] == sel_g]
        if view == "Solo pendientes":
            data = data[~data["played"]]
        elif view == "Solo jugados":
            data = data[data["played"]]

        preds = {i: predict_mode(r.home, r.away, MODE, BLEND_W)[0]
                 for i, r in data[~data["played"]].iterrows()}
        st.markdown(f'<div class="pct" style="max-width:760px;margin:6px 0 0">'
                    f'<span>Fuente: <b>{MODE}</b></span>'
                    f'<span>🔵 local</span><span>⚪ empate</span><span>🔴 visitante</span></div>',
                    unsafe_allow_html=True)
        html, cur = [], None
        for i, r in data.iterrows():
            d = _fmt_date(r.date)
            if d != cur:
                html.append(f'<div class="daterow">{d}</div>'); cur = d
            html.append(_match_html(r, preds.get(i)))
        st.markdown("".join(html), unsafe_allow_html=True)

# ============================ TORNEO (grupos) ============================== #
with tab_tourn:
    st.subheader("Fase de grupos")
    st.caption("Posiciones reales. 🟢 1º y 2º clasifican · 🟡 mejores 8 terceros también.")
    standings = get_standings()

    def _style_group(df):
        def color(row):
            if row.name <= 2:
                return ["background-color: rgba(34,197,94,.18)"] * len(row)
            if row.name == 3:
                return ["background-color: rgba(234,179,8,.15)"] * len(row)
            return [""] * len(row)
        return df.style.apply(color, axis=1)

    letters = list(standings.keys())
    for i in range(0, len(letters), 3):
        cols = st.columns(3)
        for col, g in zip(cols, letters[i:i + 3]):
            with col:
                st.markdown(f"**Grupo {g}**")
                col.dataframe(_style_group(standings[g][["team", "PJ", "Pts", "DG", "GF"]]),
                              width="stretch")

    st.divider()
    st.markdown("#### 🎯 Escenarios de clasificación")
    st.caption("Qué necesita cada selección según puntos y partidos que le quedan en el grupo.")
    sc_g = st.selectbox("Grupo", letters, key="scen_g")
    sc = get_simulator(MODE_KEY, BLEND_W).group_scenarios(sc_g)
    if sc:
        scdf = pd.DataFrame(sc)[["team", "pts", "jugados", "restan", "max_pts", "estado"]]
        scdf.columns = ["Selección", "Pts", "Jugados", "Restan", "Máx pts", "Situación"]

        def _color_estado(v):
            c = {"Clasificado": "rgba(34,197,94,.22)", "Depende": "rgba(234,179,8,.15)"}.get(v)
            return f"background-color: {c}" if c else ""
        st.dataframe(scdf.style.map(_color_estado, subset=["Situación"]),
                     width="stretch", hide_index=True)

    st.divider()
    st.subheader("Camino al título (proyección)")
    if st.button("🎲 Proyectar torneo", type="primary", key="proj"):
        with st.spinner("Simulando 3000 torneos ..."):
            st.session_state["proj_res"] = get_simulator(MODE_KEY, BLEND_W).run(n_sims=3000)
    if "proj_res" in st.session_state:
        res = st.session_state["proj_res"].head(20)
        ren = {"P_16avos": "16avos", "P_8vos": "8vos", "P_4tos": "4tos",
               "P_semis": "Semis", "P_final": "Final", "P_campeon": "Campeón"}
        heat = res.set_index("team")[list(ren)].mul(100).rename(columns=ren).reset_index()
        heat.columns = ["Selección"] + list(ren.values())
        st.dataframe(heat, width="stretch", height=560, hide_index=True,
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=100,
                                    format="%.0f%%") for c in ren.values()})
    else:
        st.info("Tocá «Proyectar torneo» para ver las probabilidades por ronda.")

# ============================ LLAVES (bracket) ============================= #
with tab_brkt:
    st.subheader("Llaves eliminatorias — cuadro oficial")
    st.caption(f"Cuadro OFICIAL de la FIFA (mapa posición-de-grupo → llave). Fuente de "
               f"probabilidad: **{MODE}**. En partidos de grupos con cuota se aplica el "
               f"filtro; los cruces de eliminatorias (sin cuota) usan el modelo. Los dos "
               f"lados convergen en la final.")
    b = get_bracket(MODE_KEY, BLEND_W)

    def _tie(m, mini=False):
        wa = "w" if m["winner"] == m["a"] else "l"
        wb = "w" if m["winner"] == m["b"] else "l"
        pa = f'<span class="pl">{int(m["pa"]*100)}%</span>' if not mini else ""
        pb = f'<span class="pl">{int(m["pb"]*100)}%</span>' if not mini else ""
        la = f'<span class="pl">{m["la"]}</span>' if m.get("la") else ""
        lb = f'<span class="pl">{m["lb"]}</span>' if m.get("lb") else ""
        return (f'<div class="tie">'
                f'<div class="t {wa}">{flag_img(m["a"], 16)} {m["a"]} {la}{pa}</div>'
                f'<div class="t {wb}">{flag_img(m["b"], 16)} {m["b"]} {lb}{pb}</div></div>')

    cols_lbl = ["16avos", "8vos", "4tos", "Semis"]

    def _side(rounds, reverse=False):
        order = list(range(len(rounds)))
        if reverse:
            order = order[::-1]
        html = ['<div class="side">']
        for ci in order:
            html.append('<div class="col">')
            html.append(f'<div class="ct">{cols_lbl[ci]}</div>')
            for m in rounds[ci]:
                html.append(_tie(m, mini=(ci > 0)))
            html.append("</div>")
        html.append("</div>")
        return "".join(html)

    fin = b["final"]
    center = (f'<div class="center"><div class="ftitle">★ Final ★</div>{_tie(fin)}'
              f'<div class="champ">🏆 {flag_img(b["champion"], 22)} {b["champion"]}</div></div>')
    st.markdown(f'<div class="bracket2">{_side(b["left"])}{center}'
                f'{_side(b["right"], reverse=True)}</div>', unsafe_allow_html=True)

    # Con quién se cruza cada grupo según salga 1º o 2º
    st.divider()
    st.markdown("#### 🔀 ¿Con quién se cruza cada grupo en 16avos?")
    st.caption("Según el mapa oficial: depende de si la selección sale 1ª o 2ª de su grupo.")
    cruces = b["cruces"]
    gcols = st.columns(3)
    for i, g in enumerate(sorted(cruces)):
        c = cruces[g]
        o1 = c.get("1"); o2 = c.get("2")
        txt = ""
        if o1:
            txt += f'<div>Si sale <b>1º</b> → vs {flag_img(o1[1], 16)} <b>{o1[1]}</b> <span style="color:#64748b">({o1[0]})</span></div>'
        if o2:
            txt += f'<div>Si sale <b>2º</b> → vs {flag_img(o2[1], 16)} <b>{o2[1]}</b> <span style="color:#64748b">({o2[0]})</span></div>'
        gcols[i % 3].markdown(
            f'<div class="cruce"><span class="gl">{g}</span><div class="opt">{txt}</div></div>',
            unsafe_allow_html=True)

# ============================ SELECCIONES ================================== #
with tab_teams:
    st.subheader("Selecciones y planteles")
    sel = st.selectbox("Selección", teams,
                       index=teams.index("Argentina") if "Argentina" in teams else 0, key="selteam")
    rank_df = ens.elo.ranking(all_teams(cfg)).reset_index(drop=True)
    rank_df.index += 1
    pos = rank_df.index[rank_df["team"] == sel]
    elo_pos = int(pos[0]) if len(pos) else "—"
    grp = next((g for g, t in cfg["groups"].items() if sel in [canonical(x, cfg) for x in t]), "—")

    coaches = get_coaches()
    crow = coaches[coaches["team"] == sel] if not coaches.empty else pd.DataFrame()
    dt_html = ""
    if not crow.empty:
        cr = crow.iloc[0]
        img = cr.get("coach_img") or ""
        av = (f'<img src="{img}" width="46" height="46" style="border-radius:50%;object-fit:cover;'
              f'border:2px solid {team_color(sel)};vertical-align:middle">' if img else "")
        dt_html = (f'<div style="text-align:right">{av}<div style="font-size:.72rem;color:#94a3b8">DT</div>'
                   f'<div style="font-weight:600">{cr["coach"]}</div></div>')

    h1, h2 = st.columns([4, 1])
    h1.markdown(f"<h3 style='margin:4px 0'>{flag_img(sel, 40)} &nbsp;{sel}</h3>", unsafe_allow_html=True)
    if dt_html:
        h2.markdown(dt_html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Elo", f"{ens.elo.rating(sel):.0f}")
    c2.metric("Ranking del torneo", f"#{elo_pos}")
    c3.metric("Grupo", grp)

    metrics = get_squad_metrics()
    mrow = metrics[metrics["team"] == sel]
    if not mrow.empty:
        mr = mrow.iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Edad promedio", f"{mr['avg_age']:.1f}")
        d2.metric("Experiencia", f"{mr['avg_caps']:.0f} caps/jug")
        d3.metric("Goles del plantel", int(mr["total_goals"]))
        d4.metric("Clubes de élite", f"{int(mr['elite_clubs'])}/26")

    colA, colB = st.columns([3, 4])
    with colA:
        form = st.selectbox("Formación", list(FORMATIONS), key="teamform")
        xi = best_xi(sel, form)
        st.markdown(pitch_team_svg(xi, sel), unsafe_allow_html=True)
        st.caption(f"Mejor XI estimado · fuerza {lineup_strength(xi['rating'].tolist()):.1f}/100")
    with colB:
        fx = get_fixture()
        nexts = (fx[((fx["home"] == sel) | (fx["away"] == sel)) & (~fx["played"])].head(4)
                 if not fx.empty else pd.DataFrame())
        if not nexts.empty:
            st.markdown("**Próximos partidos**")
            for r in nexts.itertuples(index=False):
                rival = r.away if r.home == sel else r.home
                p = predict_cached(r.home, r.away)
                mine = p["p_home"] if r.home == sel else p["p_away"]
                st.write(f"🆚 **{rival}** · {_fmt_date(r.date)} — gana {sel}: **{mine*100:.0f}%**")

        # Camino al título proyectado (bracket oficial)
        path = get_simulator(MODE_KEY, BLEND_W).title_path(sel, get_bracket(MODE_KEY, BLEND_W))
        if path:
            st.markdown("**🏆 Camino al título proyectado**")
            for s in path:
                me_adv = s["winner"] == sel
                rival = s["b"] if s["a"] == sel else (s["a"] if s["b"] == sel else None)
                if rival:
                    line = f"{flag_img(rival, 16)} {rival}"
                else:
                    line = f"{s['a']} vs {s['b']}"
                icon = "✅" if me_adv else "⬜"
                st.markdown(f"<div style='font-size:.86rem;margin:1px 0'>{icon} <b>{s['ronda']}</b> · "
                            f"{line} <span style='color:#64748b'>(favorito: {s['winner']})</span></div>",
                            unsafe_allow_html=True)
        sq = get_squads()
        team_sq = sq[sq["team"] == sel]
        if not team_sq.empty:
            st.markdown("**Plantel**")
            order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
            t = team_sq.assign(_o=team_sq["position"].map(order).fillna(9)).sort_values(["_o", "number"])
            show = t[["player_img", "number", "position", "player", "club", "caps", "goals"]].copy() \
                if "player_img" in t else t[["number", "position", "player", "club", "caps", "goals"]].copy()
            cfgcols = {"#": st.column_config.NumberColumn(format="%d"),
                       "Caps": st.column_config.NumberColumn(format="%d"),
                       "Goles": st.column_config.NumberColumn(format="%d")}
            if "player_img" in t:
                show.columns = ["", "#", "Pos", "Jugador", "Club", "Caps", "Goles"]
                cfgcols[""] = st.column_config.ImageColumn("", width="small")
            else:
                show.columns = ["#", "Pos", "Jugador", "Club", "Caps", "Goles"]
            st.dataframe(show, width="stretch", hide_index=True, height=420, column_config=cfgcols)

    with st.expander("📊 Comparar los 48 planteles"):
        comp = get_squad_metrics()[["team", "avg_age", "avg_caps", "total_goals", "elite_clubs"]].copy()
        comp.columns = ["Selección", "Edad", "Caps/jug", "Goles", "En élite"]
        st.dataframe(comp, width="stretch", hide_index=True, height=460,
                     column_config={"En élite": st.column_config.ProgressColumn(
                         "En élite", min_value=0, max_value=26, format="%d/26")})

# ============================ PARTIDO (cancha + lineups) =================== #
with tab_match:
    st.subheader("Predictor de partido con alineaciones")
    st.caption("Elegí los dos equipos, armá la formación y el XI de cada uno, "
               "y la predicción se ajusta según quién juega (titulares vs suplentes).")

    fx = get_fixture()
    pend = fx[~fx["played"]] if not fx.empty else pd.DataFrame()
    preset = {"— elegir manualmente —": (None, None)}
    for r in pend.head(40).itertuples(index=False):
        preset[f"{r.home} vs {r.away} · {_fmt_date(r.date)}"] = (r.home, r.away)
    chosen_match = st.selectbox("Cargar un partido del fixture", list(preset))
    pre_h, pre_a = preset[chosen_match]

    c1, c2 = st.columns(2)
    home = c1.selectbox("Local", teams, index=teams.index(pre_h) if pre_h in teams else
                        (teams.index("Argentina") if "Argentina" in teams else 0), key="mh")
    away = c2.selectbox("Visitante", teams, index=teams.index(pre_a) if pre_a in teams else
                        (teams.index("Brazil") if "Brazil" in teams else 1), key="ma")

    if home == away:
        st.info("Elegí dos selecciones distintas.")
    else:
        jugado = fx[(fx["home"] == home) & (fx["away"] == away) & (fx["played"])] if not fx.empty else pd.DataFrame()
        if not jugado.empty:
            rr = jugado.iloc[0]
            st.success(f"⚽ Ya jugado: **{home} {int(rr.home_score)}–{int(rr.away_score)} {away}** "
                       f"· armá las alineaciones y compará con lo que el modelo esperaba.")
        fh = c1.selectbox("Formación local", list(FORMATIONS), key="fh")
        fa = c2.selectbox("Formación visitante", list(FORMATIONS), key="fa")
        pool_h, pool_a = get_rated(home), get_rated(away)
        def_h = best_xi(home, fh)["player"].tolist()
        def_a = best_xi(away, fa)["player"].tolist()
        sel_h = c1.multiselect(f"XI {home} ({len(def_h)})", pool_h["player"].tolist(),
                               default=def_h, key=f"xih_{home}_{fh}", max_selections=11)
        sel_a = c2.multiselect(f"XI {away} ({len(def_a)})", pool_a["player"].tolist(),
                               default=def_a, key=f"xia_{away}_{fa}", max_selections=11)

        xi_h = build_xi(home, fh, sel_h or def_h)
        xi_a = build_xi(away, fa, sel_a or def_a)
        str_h = lineup_strength(xi_h["rating"].tolist())
        str_a = lineup_strength(xi_a["rating"].tolist())
        base_h = get_baseline(home, fh)
        base_a = get_baseline(away, fa)

        base_pred = ens.predict(home, away, neutral=True)
        pred = ens.predict_with_lineups(home, away, str_h, str_a, base_h, base_a, neutral=True)

        pcol, fcol = st.columns([3, 4])
        with pcol:
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Gana {home}", f"{pred['p_home']*100:.1f}%",
                      f"{(pred['p_home']-base_pred['p_home'])*100:+.1f}")
            m2.metric("Empate", f"{pred['p_draw']*100:.1f}%")
            m3.metric(f"Gana {away}", f"{pred['p_away']*100:.1f}%",
                      f"{(pred['p_away']-base_pred['p_away'])*100:+.1f}")
            st.caption(f"Fuerza de alineación — {home}: **{str_h:.1f}** (mejor {base_h:.1f}) · "
                       f"{away}: **{str_a:.1f}** (mejor {base_a:.1f})")
            bar = go.Figure(go.Bar(
                x=[pred["p_home"], pred["p_draw"], pred["p_away"]], y=[home, "Empate", away],
                orientation="h", text=[f"{v*100:.0f}%" for v in (pred["p_home"], pred["p_draw"], pred["p_away"])],
                marker_color=["#2563eb", "#9ca3af", "#dc2626"]))
            bar.update_layout(height=180, margin=dict(l=8, r=8, t=8, b=8),
                              xaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(bar, width="stretch")
            st.caption(f"xG {pred['xg_home']}–{pred['xg_away']} · marcador probable {pred['likely_score']} "
                       f"· ajuste por alineación {pred.get('lineup_shift',0):+.2f}")
        with fcol:
            st.markdown(
                f"<div class='vs'>{flag_img(home, 26)} {home} &nbsp;—&nbsp; {away} {flag_img(away, 26)}</div>",
                unsafe_allow_html=True)
            st.markdown(pitch_match_svg(xi_h, xi_a, home, away), unsafe_allow_html=True)

        # Comparación de las 3 fuentes (si hay cuota de mercado para este partido)
        mk = market_probs_map().get((home, away))
        st.divider()
        st.markdown("#### 🎚️ Modelo vs Blend vs Mercado")
        if mk is None:
            st.caption("Este partido no tiene cuota de mercado (no es un próximo partido con "
                       "cuotas). Solo se muestra el modelo.")
            mdl = predict_cached(home, away)
            st.dataframe(pd.DataFrame([{"Fuente": "🤖 Modelo", home: mdl["p_home"],
                                        "Empate": mdl["p_draw"], away: mdl["p_away"]}])
                         .style.format({home: "{:.0%}", "Empate": "{:.0%}", away: "{:.0%}"}),
                         width="stretch", hide_index=True)
        else:
            mdl = predict_cached(home, away)
            bl, _ = predict_mode(home, away, "🔀 Blend", BLEND_W)
            rows = [{"Fuente": "🤖 Modelo", home: mdl["p_home"], "Empate": mdl["p_draw"], away: mdl["p_away"]},
                    {"Fuente": f"🔀 Blend ({BLEND_W:.0%}/{1-BLEND_W:.0%})", home: bl["p_home"],
                     "Empate": bl["p_draw"], away: bl["p_away"]},
                    {"Fuente": "🏦 Mercado", home: mk[0], "Empate": mk[1], away: mk[2]}]
            st.dataframe(pd.DataFrame(rows).style.format(
                {home: "{:.0%}", "Empate": "{:.0%}", away: "{:.0%}"}),
                width="stretch", hide_index=True)
            st.caption("El blend mezcla modelo y mercado; ajustá el peso en la barra lateral.")

        # Historial de enfrentamientos directos
        h2h = get_h2h(home, away)
        st.divider()
        if h2h["played"] == 0:
            st.caption(f"🆚 Sin enfrentamientos directos entre {home} y {away} en el histórico (desde 2002).")
        else:
            st.markdown(f"#### 🆚 Historial directo · {h2h['played']} partidos (desde 2002)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric(f"Gana {home}", h2h["wins_a"])
            g2.metric("Empates", h2h["draws"])
            g3.metric(f"Gana {away}", h2h["wins_b"])
            g4.metric("Goles", f"{h2h['gf_a']}–{h2h['gf_b']}")
            st.caption("Últimos cruces:  " +
                       "  ·  ".join(f"{r['result']} ({r['date'][:4]})" for r in h2h["recent"]))

# ============================ CUÁNTICO ===================================== #
with tab_quantum:
    st.subheader("⚛️ Clasificador Cuántico Variacional (QML)")
    qmodel = get_quantum_model()
    qd = get_quantum()
    if qd is None or qmodel is None:
        st.info("No hay modelo cuántico. Entrenalo con:  `python quantum_match.py`")
    else:
        st.caption("Red neuronal cuántica (PennyLane) de **4 qubits** con *data re-uploading*, "
                   "entrenada sobre el histórico. Codifica 4 features (diferencias de Elo, forma "
                   "en puntos y goles a favor/en contra) como rotaciones, las procesa con un "
                   "circuito variacional entrenable y mide ⟨Z⟩ en 3 qubits → softmax. Predice "
                   "**1-X-2** (local / empate / visitante), directamente comparable al ensemble.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precisión (test)", f"{qd['test_acc']*100:.1f}%")
        m2.metric("Partidos de entrenamiento", f"{qd['n_matches']:,}")
        m3.metric("Qubits", "4")
        m4.metric("Clases", "1-X-2")

        st.divider()
        st.markdown("#### Cuántico vs Ensemble en un partido")
        qc1, qc2 = st.columns(2)
        qh = qc1.selectbox("Local", teams,
                           index=teams.index("Argentina") if "Argentina" in teams else 0, key="qh")
        qa = qc2.selectbox("Visitante", teams,
                           index=teams.index("Brazil") if "Brazil" in teams else 1, key="qa")

        if qh == qa:
            st.info("Elegí dos selecciones distintas.")
        else:
            qp = quantum_probs(qmodel, qd, qh, qa)
            ens_p = predict_cached(qh, qa)
            if qp is None:
                st.warning("Falta data de alguna selección en el modelo cuántico.")
            else:
                labels = [f"Gana {qh}", "Empate", f"Gana {qa}"]
                qv = [qp["H"], qp["D"], qp["A"]]
                ev = [ens_p["p_home"], ens_p["p_draw"], ens_p["p_away"]]
                q_pick = labels[int(np.argmax(qv))]
                e_pick = labels[int(np.argmax(ev))]

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown(
                        f"<div style='text-align:center'>{flag_img(qh,30)} <b>{qh}</b> vs "
                        f"<b>{qa}</b> {flag_img(qa,30)}</div>", unsafe_allow_html=True)
                    bar = go.Figure()
                    bar.add_trace(go.Bar(y=labels, x=qv, orientation="h", name="⚛️ Cuántico",
                                         marker_color="#a855f7",
                                         text=[f"{v*100:.0f}%" for v in qv]))
                    bar.add_trace(go.Bar(y=labels, x=ev, orientation="h", name="🧮 Ensemble",
                                         marker_color="#16a34a",
                                         text=[f"{v*100:.0f}%" for v in ev]))
                    bar.update_layout(barmode="group", xaxis_tickformat=".0%",
                                      yaxis=dict(autorange="reversed"),
                                      legend=dict(orientation="h", y=1.15))
                    st.plotly_chart(_theme_fig(bar, 260), width="stretch")
                    if q_pick == e_pick:
                        st.success(f"✅ Ambos modelos coinciden: **{q_pick}**.")
                    else:
                        st.warning(f"⚠️ Discrepan — cuántico: **{q_pick}** · ensemble: **{e_pick}**.")
                with cc2:
                    gx, gy, z = qd["gx"], qd["gy"], qd["z"]
                    f = qd["feat"]
                    ed = f[qh]["elo"] - f[qa]["elo"]
                    fd = f[qh]["form_pts"] - f[qa]["form_pts"]
                    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, zmid=0, colorscale="RdBu",
                                               showscale=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(
                        x=[ed], y=[fd], mode="markers",
                        marker=dict(size=15, color="white", line=dict(color="black", width=2))))
                    fig.update_layout(title="Frontera de decisión (P local − P visitante)",
                                      xaxis_title="ventaja de Elo", yaxis_title="ventaja de forma",
                                      showlegend=False)
                    st.plotly_chart(_theme_fig(fig, 340), width="stretch")
                    st.caption("Azul: gana local · Rojo: gana visitante. Las otras 2 features = 0.")

        st.divider()
        st.markdown("#### Dónde más discrepan (partidos pendientes del fixture)")
        st.caption("Distancia total entre las distribuciones 1-X-2 de ambos modelos. Las mayores "
                   "brechas marcan dónde el enfoque cuántico se aparta del ensemble: buen punto "
                   "de partida para analizar y mejorar.")
        fxq = get_fixture()
        pendq = fxq[~fxq["played"]] if not fxq.empty else pd.DataFrame()
        rows = []
        for r in pendq.itertuples(index=False):
            qp = quantum_probs(qmodel, qd, r.home, r.away)
            if qp is None:
                continue
            ep = predict_cached(r.home, r.away)
            qv = [qp["H"], qp["D"], qp["A"]]
            ev = [ep["p_home"], ep["p_draw"], ep["p_away"]]
            picks = [f"gana {r.home}", "empate", f"gana {r.away}"]
            dist = 0.5 * sum(abs(a - b) for a, b in zip(qv, ev))
            rows.append({"Partido": f"{r.home} vs {r.away}",
                         "⚛️ Cuántico": picks[int(np.argmax(qv))],
                         "🧮 Ensemble": picks[int(np.argmax(ev))],
                         "Coincide": "✓" if np.argmax(qv) == np.argmax(ev) else "✗",
                         "Distancia": round(dist * 100)})
        if rows:
            dfq = pd.DataFrame(rows).sort_values("Distancia", ascending=False)
            st.dataframe(dfq, width="stretch", hide_index=True, height=400,
                         column_config={"Distancia": st.column_config.ProgressColumn(
                             "Distancia", min_value=0, max_value=100, format="%d")})
        else:
            st.info("No hay partidos pendientes para comparar.")

        with st.expander("💡 Notas del modelo y próximas mejoras"):
            st.markdown(
                "Versión actual: **4 qubits, multiclase 1-X-2**, 4 features (Elo + forma "
                "pts/GF/GA), entropía cruzada, salida por softmax sobre ⟨Z⟩ de 3 qubits.\n\n"
                "- **Más features / qubits**: sumar localía, descanso o fuerza del plantel (6-8 qubits).\n"
                "- **Otro ansatz / codificación**: probar `AmplitudeEmbedding` o más capas.\n"
                "- **Ensamblar**: incorporar la distribución cuántica como un 4º modelo del "
                "ensemble y validar el log-loss en la pestaña Rendimiento.\n"
                "- Reentrenar tras nuevos datos:  `python quantum_match.py`")


# ============================ RANKING ====================================== #
with tab_rank:
    st.subheader("Ranking Elo de las selecciones")
    rank = ens.elo.ranking(all_teams(cfg)).reset_index(drop=True)
    rank.index += 1
    fig = px.bar(rank.head(24), x="elo", y="team", orientation="h",
                 color="elo", color_continuous_scale="Greens")
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(_theme_fig(fig, 600), width="stretch")
    st.dataframe(rank.style.format({"elo": "{:.0f}"}), width="stretch")

# ============================ SIMULAR ====================================== #
with tab_sim:
    st.subheader("Simulación Monte Carlo del torneo")
    n_sims = st.slider("Simulaciones", 500, 20000, 3000, step=500)
    if st.button("▶️ Correr simulación", type="primary"):
        with st.spinner(f"Simulando {n_sims:,} torneos ..."):
            st.session_state["sim_res"] = get_simulator(MODE_KEY, BLEND_W).run(n_sims=n_sims)
    if "sim_res" in st.session_state:
        res = st.session_state["sim_res"]
        top = res.head(15)
        fig = px.bar(top, x="P_campeon", y="team", orientation="h",
                     text=top["P_campeon"].map(lambda v: f"{v*100:.1f}%"),
                     color="P_campeon", color_continuous_scale="YlOrRd")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat=".0%",
                          coloraxis_showscale=False)
        st.plotly_chart(_theme_fig(fig, 550), width="stretch")
        pct = ["P_16avos", "P_8vos", "P_4tos", "P_semis", "P_final", "P_campeon"]
        st.dataframe(res.style.format({c: "{:.1%}" for c in pct}),
                     width="stretch", hide_index=True)

# ============================ RENDIMIENTO ================================== #
with tab_perf:
    st.subheader("Rendimiento del modelo")
    st.caption("Evaluaciones honestas, todas out-of-sample. RPS = métrico estándar 1X2 "
               "(menor = mejor). La regla: si el modelo no le gana a «siempre local» o no se "
               "acerca al mercado, no aporta.")

    pc1, pc2 = st.columns(2)
    if pc1.button("📊 Evaluar sobre el Mundial"):
        with st.spinner("Entrenando as-of 11/6 ..."):
            st.session_state["wc_bt"] = get_wc_backtest()
    if pc2.button("📈 Calibración sobre histórico"):
        with st.spinner("Entrenando as-of hace 12 meses ..."):
            st.session_state["calib_bt"] = get_calibration()

    if st.session_state.get("wc_bt"):
        bt = st.session_state["wc_bt"]
        st.markdown(f"**Mundial 2026 · {bt['n']} partidos jugados (out-of-sample)**")
        st.dataframe(bt["compare"], width="stretch")
        st.caption("⚠️ En el Mundial el modelo todavía no le gana a «siempre local»: muestra "
                   "chica + equipos parejos + sorpresas.")
        with st.expander("Detalle partido a partido"):
            st.dataframe(bt["per_match"], width="stretch", hide_index=True)

    if st.session_state.get("calib_bt"):
        cb = st.session_state["calib_bt"]
        st.markdown(f"**Histórico · {cb['n']} partidos competitivos (señal real)**")
        st.dataframe(cb["compare"], width="stretch")
        cal = cb["calib"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.33, 1], y=[0.33, 1], mode="lines",
                                 line=dict(dash="dash", color="#64748b"), name="ideal"))
        fig.add_trace(go.Scatter(x=cal["confianza_media"], y=cal["acierto_real"],
                                 mode="markers+lines", marker=dict(size=11, color="#16a34a"),
                                 line=dict(color="#16a34a"), name="modelo"))
        fig.update_layout(xaxis_title="Confianza del modelo", yaxis_title="Acierto real",
                          title="Calibración: cuando dice X%, ¿acierta X%?",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(_theme_fig(fig, 340), width="stretch")
        st.caption("Si los puntos siguen la diagonal, las probabilidades son confiables.")

    st.divider()
    # --- Tracking en vivo ---
    st.markdown("#### 📡 Tracking en vivo (modelo vs mercado, partidos reales)")
    st.caption("Compara las predicciones congeladas ANTES de cada partido (modelo y mercado) "
               "contra el resultado real. El benchmark más honesto; se acumula con cada fecha.")
    trk = get_tracking()
    if not trk or trk.get("n", 0) == 0:
        st.info("Todavía no hay partidos jugados con snapshot previo — recién empieza a "
                "acumular. El update diario va guardando predicciones antes de cada fecha.")
    else:
        cols = st.columns(3 if "mercado" in trk else 1)
        m = trk["modelo"]
        cols[0].metric("Modelo · log-loss", f"{m['logloss']:.3f}", f"RPS {m['rps']:.3f}")
        if "mercado" in trk:
            mk = trk["mercado"]
            cols[1].metric("Mercado · log-loss", f"{mk['logloss']:.3f}", f"RPS {mk['rps']:.3f}")
            gap = trk["modelo_en_mismos"]["logloss"] - mk["logloss"]
            cols[2].metric("Brecha vs mercado", f"{gap:+.3f}", delta_color="off")
        st.caption(f"{trk['n']} partidos con snapshot" +
                   (f" · {trk.get('n_mercado', 0)} con cuota" if "mercado" in trk else ""))

    st.divider()
    # --- Validación de tilts ---
    st.markdown("#### 🎛️ ¿Sirven los ajustes? (noticias / anfitrión)")
    st.caption("Modelo con cada ajuste prendido/apagado sobre el Mundial jugado. Si una fila "
               "no mejora el log-loss vs «base», ese ajuste no aporta.")
    if st.button("Validar los ajustes"):
        st.session_state["tilts"] = get_tilts()
    tl = st.session_state.get("tilts")
    if tl is not None and not tl.empty:
        st.dataframe(tl, width="stretch")
        st.caption("Evidencia débil (muestra chica + noticias con sentimiento actual). "
                   "El tracking en vivo dará el veredicto definitivo.")

    st.divider()
    # --- Campeón: modelo vs mercado ---
    st.markdown("#### 🏆 Campeón: modelo vs mercado")
    st.caption("Mi probabilidad de campeón (simulación) vs la del mercado (cuotas outright, en vivo).")
    if st.button("Comparar probabilidad de campeón vs mercado"):
        from src.evaluation.odds_benchmark import champion_benchmark
        sim = st.session_state.get("sim_res")
        if sim is None:
            with st.spinner("Simulando 4000 torneos ..."):
                sim = get_simulator(MODE_KEY, BLEND_W).run(n_sims=4000)
                st.session_state["sim_res"] = sim
        st.session_state["champ_bt"] = champion_benchmark(
            sim.set_index("team")["P_campeon"].to_dict())
    chb = st.session_state.get("champ_bt")
    if chb is not None and not chb.empty:
        from scipy.stats import spearmanr
        rho = spearmanr(chb["mercado"], chb["modelo"]).correlation
        k1, k2, k3 = st.columns(3)
        k1.metric("Correlación con el mercado", f"{rho:.2f}",
                  help="Spearman del ranking. 1.0 = orden idéntico al mercado.")
        k2.metric("Favorito del mercado", chb.iloc[0]["team"])
        k3.metric("Favorito del modelo", chb.sort_values("modelo", ascending=False).iloc[0]["team"])
        top = chb.head(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["team"], x=top["mercado"] * 100, orientation="h",
                             name="Mercado", marker_color="#f59e0b"))
        fig.add_trace(go.Bar(y=top["team"], x=top["modelo"] * 100, orientation="h",
                             name="Modelo", marker_color="#16a34a"))
        fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"),
                          xaxis_title="% campeón", legend=dict(orientation="h"))
        st.plotly_chart(_theme_fig(fig, 460), width="stretch")
    elif chb is not None:
        st.info("No se pudieron traer las cuotas de campeón (revisá la API key).")

    st.divider()
    # --- Modelo vs mercado en partidos (cuotas h2h) ---
    st.markdown("#### 🏦 Modelo vs Mercado (partidos con cuotas jugados)")
    if st.button("Comparar contra las cuotas"):
        from src.evaluation.odds_benchmark import benchmark
        st.session_state["odds_bt"] = benchmark(ens) or "empty"
    ob = st.session_state.get("odds_bt")
    if ob == "empty":
        st.info("No hay partidos jugados con cuota guardada todavía (el tracking las acumula).")
    elif ob:
        tb = pd.DataFrame({"Modelo": ob["modelo"], "Mercado": ob["mercado"]}).T
        tb.columns = ["log_loss", "RPS", "accuracy"]
        st.dataframe(tb[["log_loss", "RPS", "accuracy"]].round(4), width="stretch")
        st.caption(f"{ob['n']} partidos con cuotas. Cuanto más cerca del mercado, mejor.")

# ============================ DATOS & NOTICIAS ============================= #
with tab_data:
    st.subheader("Actualización de datos")
    cu1, cu2 = st.columns([3, 1])
    cu1.write(f"**Última actualización:** {last_update()}")
    if cu2.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown(
        "Reentrenar con lo último:\n```\npython -m src.pipeline\n```\n"
        "Regenerar grupos / planteles:\n```\npython -m src.data.sources --regen-groups\n"
        "python -m src.data.squads\n```\n"
        "Resultados nuevos del Mundial → `data/raw/wc26_manual.csv`.")

    st.divider()
    st.subheader("Sentimiento de noticias por selección")
    from src.data.news import load_sentiment, load_headlines
    s = load_sentiment()
    if s.empty:
        st.info("Sin datos de noticias. Corré `python -m src.data.news`.")
    else:
        s = s.sort_values("news_sentiment", ascending=False)
        fig = px.bar(s, x="news_sentiment", y="team", orientation="h",
                     color="news_sentiment", color_continuous_scale="RdYlGn", range_color=[-1, 1])
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(_theme_fig(fig, max(400, len(s) * 22)), width="stretch")

        st.divider()
        st.subheader("🗞️ Titulares analizados")
        hl = load_headlines()
        if hl.empty:
            st.info("Sin titulares. Corré `python -m src.data.news`.")
        else:
            has_rel = "relevant" in hl.columns
            teams_news = ["Todas"] + sorted(hl["team"].unique())
            cf, cm, cr = st.columns([2, 2, 2])
            st_ = cf.selectbox("Selección", teams_news, key="newsteam")
            ma = cm.select_slider("Intensidad", ["Todos", "No neutrales", "Fuertes"], value="Todos")
            only_rel = cr.toggle("Solo los que usa el modelo", value=False) if has_rel else False
            v = hl if st_ == "Todas" else hl[hl["team"] == st_]
            if has_rel and only_rel:
                v = v[v["relevant"]]
            if ma == "No neutrales":
                v = v[v["sentiment"].abs() >= 0.2]
            elif ma == "Fuertes":
                v = v[v["sentiment"].abs() >= 0.5]
            v = v.sort_values("sentiment")
            cols = ["team", "title", "source", "sentiment"] + (["relevant"] if has_rel else []) + ["link"]
            show = v[cols].copy()
            show.columns = ["Selección", "Titular", "Medio", "Sent."] + (["Usado"] if has_rel else []) + ["Link"]
            colcfg = {"Sent.": st.column_config.NumberColumn(format="%.2f"),
                      "Link": st.column_config.LinkColumn("Abrir", display_text="🔗"),
                      "Titular": st.column_config.TextColumn(width="large")}
            if has_rel:
                colcfg["Usado"] = st.column_config.CheckboxColumn()
            st.dataframe(show, width="stretch", hide_index=True, height=460, column_config=colcfg)
