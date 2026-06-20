"""
Simulación Monte Carlo del Mundial 2026.

Formato 2026: 12 grupos de 4 (todos contra todos). Avanzan los 2 primeros
de cada grupo + los 8 mejores terceros = 32 equipos a la fase eliminatoria
(dieciseisavos -> octavos -> cuartos -> semis -> final).

Para que miles de simulaciones corran rápido, se cachean por cruce:
  - las probabilidades H/D/A del ensemble
  - la distribución de marcadores del Poisson, partida por resultado,
    para muestrear diferencias/goles coherentes con el resultado del ensemble.

El emparejamiento del bracket usa siembra por Elo (fuerte vs débil). El mapa
oficial posición-de-grupo -> llave puede personalizarse si se desea exactitud.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import load_config, all_teams
from ..models.ensemble import EnsemblePredictor


class TournamentSimulator:
    def __init__(self, ensemble: EnsemblePredictor, cfg: dict | None = None, seed: int = 42,
                 knockout_temp: float | None = None):
        self.ens = ensemble
        self.cfg = cfg or load_config()
        self.groups = {g: [self._canon(t) for t in teams]
                       for g, teams in self.cfg["groups"].items()}
        # Temperatura del knockout (>1 aplana hacia 50/50: a partido único hay más
        # azar que el que sugiere el rating). 1.0 = sin ajuste.
        self.knockout_temp = (knockout_temp if knockout_temp is not None
                              else float(self.cfg.get("knockout_temperature", 1.0)))
        self.rng = np.random.default_rng(seed)
        self._prob_cache: dict[tuple[str, str], dict] = {}
        self._score_cache: dict[tuple[str, str], dict] = {}

    def _canon(self, team: str) -> str:
        from ..config import canonical
        return canonical(team, self.cfg)

    # ------------------------------------------------------------------ #
    def _probs(self, a: str, b: str) -> dict:
        key = (a, b)
        if key not in self._prob_cache:
            self._prob_cache[key] = self.ens.predict(a, b, neutral=True)
        return self._prob_cache[key]

    def _score_dist(self, a: str, b: str) -> dict:
        """Distribución de marcadores agrupada por resultado (H/D/A) para muestreo."""
        key = (a, b)
        if key not in self._score_cache:
            mat = self.ens.poisson.score_matrix(a, b, neutral=True)
            buckets = {"H": [], "D": [], "A": []}
            n = mat.shape[0]
            for i in range(n):
                for j in range(n):
                    p = mat[i, j]
                    if p <= 0:
                        continue
                    res = "H" if i > j else ("A" if i < j else "D")
                    buckets[res].append(((i, j), p))
            norm = {}
            for res, lst in buckets.items():
                if lst:
                    scores, ps = zip(*lst)
                    ps = np.array(ps) / sum(ps)
                    norm[res] = (scores, ps)
                else:
                    norm[res] = (([0, 0] if res != "D" else [0, 0],), np.array([1.0]))
            self._score_cache[key] = norm
        return self._score_cache[key]

    @staticmethod
    def _norm3(ph, pd_, pa):
        """Renormaliza a suma exacta 1.0 (rng.choice es estricto con el redondeo)."""
        arr = np.array([ph, pd_, pa], dtype=float)
        arr = np.clip(arr, 0, None)
        s = arr.sum()
        return (arr / s) if s > 0 else np.array([1 / 3, 1 / 3, 1 / 3])

    def _sample_match(self, a: str, b: str) -> tuple[int, int, str]:
        """Resultado del ensemble + marcador coherente muestreado del Poisson."""
        p = self._probs(a, b)
        res = self.rng.choice(["H", "D", "A"], p=self._norm3(p["p_home"], p["p_draw"], p["p_away"]))
        scores, ps = self._score_dist(a, b)[res]
        ps = np.array(ps, dtype=float); ps = ps / ps.sum()
        idx = self.rng.choice(len(scores), p=ps)
        gh, ga = scores[idx]
        return int(gh), int(ga), res

    def _knockout_winner(self, a: str, b: str) -> str:
        p = self._probs(a, b)
        ph, pd_, pa = p["p_home"], p["p_draw"], p["p_away"]
        if self.knockout_temp != 1.0:
            arr = np.array([ph, pd_, pa]) ** (1.0 / self.knockout_temp)
            ph, pd_, pa = arr / arr.sum()
        ph, pd_, pa = self._norm3(ph, pd_, pa)
        res = self.rng.choice(["H", "D", "A"], p=[ph, pd_, pa])
        if res == "H":
            return a
        if res == "A":
            return b
        # Empate -> penales, ponderado por fuerza relativa
        pen_home = ph / (ph + pa) if (ph + pa) > 0 else 0.5
        return a if self.rng.random() < pen_home else b

    # ------------------------------------------------------------------ #
    def _simulate_group(self, teams: list[str]) -> pd.DataFrame:
        stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                gh, ga, _ = self._sample_match(a, b)
                stats[a]["gf"] += gh; stats[a]["ga"] += ga
                stats[b]["gf"] += ga; stats[b]["ga"] += gh
                if gh > ga:
                    stats[a]["pts"] += 3
                elif gh < ga:
                    stats[b]["pts"] += 3
                else:
                    stats[a]["pts"] += 1; stats[b]["pts"] += 1
        df = pd.DataFrame([
            {"team": t, "pts": s["pts"], "gd": s["gf"] - s["ga"], "gf": s["gf"],
             "rnd": self.rng.random()}
            for t, s in stats.items()
        ])
        return df.sort_values(["pts", "gd", "gf", "rnd"], ascending=False).reset_index(drop=True)

    @staticmethod
    def _seed_positions(n: int) -> list[int]:
        """Orden de siembra estándar de un cuadro de tamaño n (1 y 2 en extremos
        opuestos: solo se cruzan en la final). Devuelve posiciones 1-indexadas."""
        rounds = [1, 2]
        while len(rounds) < n:
            m = len(rounds) * 2
            rounds = [x for r in rounds for x in (r, m + 1 - r)]
        return rounds

    def _bracket_order(self, qualified: list[str]) -> list[str]:
        """Siembra por Elo en un cuadro estándar: el mejor sembrado enfrenta al
        peor, y los favoritos quedan repartidos en lados opuestos del cuadro.
        Recorta a la mayor potencia de 2 para que el cuadro sea siempre válido."""
        seeded = sorted(qualified, key=lambda t: self.ens.elo.rating(t), reverse=True)
        n = 1 << (len(seeded).bit_length() - 1) if seeded else 0   # mayor potencia de 2 <= len
        seeded = seeded[:n]
        return [seeded[p - 1] for p in self._seed_positions(n)] if n >= 2 else seeded

    def simulate_once(self) -> dict:
        """Una simulación completa. Devuelve hasta dónde llegó cada equipo."""
        firsts, seconds, thirds = [], [], []
        for teams in self.groups.values():
            standing = self._simulate_group(teams)
            firsts.append(standing.iloc[0]["team"])
            seconds.append(standing.iloc[1]["team"])
            row = standing.iloc[2]
            thirds.append((row["team"], row["pts"], row["gd"], row["gf"], self.rng.random()))

        thirds_df = pd.DataFrame(thirds, columns=["team", "pts", "gd", "gf", "rnd"])
        best_thirds = (thirds_df.sort_values(["pts", "gd", "gf", "rnd"], ascending=False)
                       .head(8)["team"].tolist())

        qualified = firsts + seconds + best_thirds  # 32
        reached = {t: "Grupos" for g in self.groups.values() for t in g}
        for t in qualified:
            reached[t] = "16avos"

        round_names = ["16avos", "8vos", "4tos", "Semis", "Final", "Campeón"]
        alive = self._bracket_order(qualified)
        rnd_i = 0
        while len(alive) > 1:
            nxt = []
            for k in range(0, len(alive), 2):
                a, b = alive[k], alive[k + 1]
                winner = self._knockout_winner(a, b)
                nxt.append(winner)
                reached[winner] = round_names[rnd_i + 1]
            alive = nxt
            rnd_i += 1
        return reached

    def projected_bracket(self) -> list[list[dict]]:
        """
        Cuadro eliminatorio MÁS PROBABLE (determinista): clasifican los 2 mejores
        de cada grupo (por puntos actuales y Elo como desempate) + los 8 mejores
        terceros por Elo; en cada cruce avanza el favorito. Devuelve, por ronda,
        una lista de {a, b, winner, pa, pb}.
        """
        from ..data.sources import load_standings
        standings = load_standings()

        def elo(t):
            return self.ens.elo.rating(t)

        firsts, seconds, thirds = [], [], []
        for g, df in standings.items():
            teams = list(df["team"])
            pts = dict(zip(df["team"], df["Pts"]))
            dg = dict(zip(df["team"], df["DG"]))
            teams.sort(key=lambda t: (-pts.get(t, 0), -dg.get(t, 0), -elo(t)))
            if len(teams) >= 3:
                firsts.append(teams[0]); seconds.append(teams[1]); thirds.append(teams[2])
        thirds.sort(key=lambda t: -elo(t))
        qualified = firsts + seconds + thirds[:8]
        if len(qualified) < 2:
            return []

        alive = self._bracket_order(qualified)
        rounds: list[list[dict]] = []
        while len(alive) > 1:
            nxt, matches = [], []
            for i in range(0, len(alive) - 1, 2):
                a, b = alive[i], alive[i + 1]
                p = self._probs(a, b)
                pa, pb = p["p_home"], p["p_away"]
                w = a if pa >= pb else b
                matches.append({"a": a, "b": b, "winner": w,
                                "pa": round(pa, 2), "pb": round(pb, 2)})
                nxt.append(w)
            rounds.append(matches)
            alive = nxt
        return rounds

    def run(self, n_sims: int = 5000) -> pd.DataFrame:
        stage_rank = {"Grupos": 0, "16avos": 1, "8vos": 2, "4tos": 3,
                      "Semis": 4, "Final": 5, "Campeón": 6}
        counts = defaultdict(lambda: defaultdict(int))
        teams = all_teams(self.cfg)

        for _ in range(n_sims):
            reached = self.simulate_once()
            for t, stage in reached.items():
                # cuenta acumulada: alcanzar una ronda cuenta también las anteriores
                # (llegar a la Final suma 16avos..Final; ser Campeón suma todas)
                for st, rank in stage_rank.items():
                    if 1 <= rank <= stage_rank[stage]:
                        counts[t][st] += 1

        rows = []
        for t in teams:
            c = counts[t]
            rows.append({
                "team": t,
                "P_16avos": c["16avos"] / n_sims,
                "P_8vos": c["8vos"] / n_sims,
                "P_4tos": c["4tos"] / n_sims,
                "P_semis": c["Semis"] / n_sims,
                "P_final": c["Final"] / n_sims,
                "P_campeon": c["Campeón"] / n_sims,
            })
        return (pd.DataFrame(rows)
                .sort_values("P_campeon", ascending=False)
                .reset_index(drop=True))
