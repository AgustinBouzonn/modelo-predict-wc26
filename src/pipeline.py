"""
Orquestador: ingesta -> entrenamiento de los 3 modelos -> ensemble -> guardado.

Uso:
    python -m src.pipeline                 # ingesta + entrena todo
    python -m src.pipeline --no-fetch      # reentrena con datos ya descargados
    python -m src.pipeline --no-news       # saltea scraping de noticias
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import PROCESSED_DIR, load_config, all_teams, canonical
from .data import sources
from .models.elo import EloModel
from .models.poisson import DixonColesModel
from .models.ml_model import MLModel
from .models.ensemble import EnsemblePredictor, load_or_build_sentiment


def load_matches(fetch: bool = True) -> pd.DataFrame:
    if fetch:
        return sources.update_all()
    path = PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        return sources.update_all()
    return pd.read_parquet(path)


def train_all(fetch: bool = True, fetch_news: bool = True, n_sims: int = 0) -> EnsemblePredictor:
    cfg = load_config()
    if fetch:
        # Auto-traer resultados en vivo (the-odds-api /scores) antes de procesar
        try:
            from .data.live_results import update_manual
            update_manual()
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] resultados live no disponibles: {type(e).__name__}")
    matches = load_matches(fetch=fetch)
    print(f"\nEntrenando con {len(matches):,} partidos "
          f"({matches['date'].min().date()} a {matches['date'].max().date()})\n")

    print("• Elo ...")
    elo = EloModel().fit(matches)
    beta = float(cfg.get("elo_confederation_beta", 0.5))
    offset = elo.apply_confederation_correction(matches, beta=beta) if beta > 0 else {}
    elo.save()
    if offset:
        print("  corrección por confederación (offset Elo):")
        for c, v in sorted(offset.items(), key=lambda x: -x[1]):
            print(f"    {c:<9} {v:+.0f}")

    print("• Poisson (Dixon-Coles) ...")
    poisson = DixonColesModel().fit(matches)
    poisson.save()

    print("• ML (XGBoost) ...")
    ml = MLModel(device=cfg.get("ml_device", "auto")).fit(matches, conf_offset=offset)
    ml.save()
    print(f"  device usado: {getattr(ml, 'device_used', 'cpu')} "
          f"(config ml_device={cfg.get('ml_device', 'auto')})")

    if fetch_news:
        print("• Noticias + sentimiento ...")
        try:
            from .data.news import build_sentiment_table
            build_sentiment_table()
        except Exception as e:  # noqa: BLE001
            print(f"  [aviso] noticias no disponibles: {e}")

    sentiment = load_or_build_sentiment(cfg)
    weights = cfg.get("ensemble_weights", {"elo": 0.4, "poisson": 0.35, "ml": 0.25})
    hosts = [canonical(h, cfg) for h in cfg.get("tournament", {}).get("hosts", [])]
    ens = EnsemblePredictor(elo=elo, poisson=poisson, ml=ml,
                            weights=weights, sentiment=sentiment,
                            hosts=hosts, host_tilt=float(cfg.get("host_advantage_tilt", 0.0)))
    ens.save()
    print("\n✓ Ensemble entrenado y guardado en models/ensemble.joblib")

    # Top del ranking Elo de las selecciones del torneo
    rank = elo.ranking(all_teams(cfg)).head(10)
    print("\nTop-10 Elo (selecciones WC26):")
    for r in rank.itertuples(index=False):
        print(f"  {r.team:<22} {r.elo:7.0f}")

    if n_sims > 0:
        from .simulation.tournament import TournamentSimulator
        print(f"\nSimulando torneo ({n_sims:,} corridas) ...")
        sim = TournamentSimulator(ens, cfg)
        res = sim.run(n_sims=n_sims)
        print(res.head(10).to_string(index=False))
    return ens


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline de entrenamiento WC26")
    ap.add_argument("--no-fetch", action="store_true", help="No redescargar datos")
    ap.add_argument("--no-news", action="store_true", help="No scrapear noticias")
    ap.add_argument("--sims", type=int, default=0, help="Correr N simulaciones del torneo")
    args = ap.parse_args()
    train_all(fetch=not args.no_fetch, fetch_news=not args.no_news, n_sims=args.sims)
