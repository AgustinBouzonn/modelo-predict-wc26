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
                 knockout_temp: float | None = None, market: dict | None = None,
                 mode: str = "modelo", blend_w: float = 0.5):
        self.ens = ensemble
        self.cfg = cfg or load_config()
        # Fuente de probabilidad: "modelo" | "blend" | "mercado". El mercado
        # (dict {(home,away):(ph,pd,pa)}) solo cubre partidos con cuota; el resto
        # cae al modelo. Permite que TODO el cuadro/simulación use el filtro.
        self.market = market or {}
        self.mode = mode
        self.blend_w = blend_w
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
            out = dict(self.ens.predict(a, b, neutral=True))
            mk = self.market.get((a, b))
            if mk and self.mode != "modelo":
                ph, pd_, pa = mk
                if self.mode == "mercado":
                    out.update(p_home=ph, p_draw=pd_, p_away=pa)
                else:  # blend
                    w = self.blend_w
                    bh = w * out["p_home"] + (1 - w) * ph
                    bd = w * out["p_draw"] + (1 - w) * pd_
                    ba = w * out["p_away"] + (1 - w) * pa
                    s = bh + bd + ba
                    out.update(p_home=bh / s, p_draw=bd / s, p_away=ba / s)
            self._prob_cache[key] = out
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

    def official_bracket(self) -> dict:
        """
        Cuadro eliminatorio OFICIAL del WC2026 (mapa FIFA posición→llave), no
        siembra por Elo. Cada llave de 16avos enfrenta posiciones fijas de grupo
        (1º/2º) y los 8 mejores terceros. Devuelve los dos lados del cuadro (que
        convergen en la final) + el detalle de con quién se cruza cada grupo.

        La asignación de los 8 terceros a sus llaves usa un emparejamiento greedy
        respetando los grupos permitidos por la FIFA en cada slot (aproximación;
        la tabla oficial exacta depende de la combinación de terceros).
        """
        from ..data.sources import load_standings
        # Mapa oficial de 16avos (match 73..88), por par de slots de posición.
        # "1X"=ganador grupo X, "2X"=2º grupo X, "3"=uno de los mejores terceros.
        R32 = [("2A", "2B"), ("1E", "3"), ("1F", "2C"), ("1C", "2F"),
               ("1I", "3"), ("2E", "2I"), ("1A", "3"), ("1L", "3"),
               ("1G", "3"), ("1D", "3"), ("2K", "2L"), ("1H", "2J"),
               ("1B", "3"), ("1J", "2H"), ("1K", "3"), ("2D", "2G")]
        third_allowed = {1: "ABCDF", 4: "CDFGH", 6: "CEFHI", 7: "EHIJK",
                         8: "AEHIJ", 9: "BEFIJ", 12: "EFGIJ", 14: "DEIJL"}

        standings = load_standings()

        def elo(t):
            return self.ens.elo.rating(t)

        # 1º, 2º, 3º de cada grupo (desempate por Pts, DG, GF, Elo)
        pos, thirds = {}, []
        for g, df in standings.items():
            pts = dict(zip(df["team"], df["Pts"]))
            dg = dict(zip(df["team"], df["DG"]))
            gf = dict(zip(df["team"], df["GF"]))
            ordered = sorted(df["team"],
                             key=lambda t: (-pts.get(t, 0), -dg.get(t, 0), -gf.get(t, 0), -elo(t)))
            pos[g] = ordered
            if len(ordered) >= 3:
                thirds.append((ordered[2], g, pts.get(ordered[2], 0), dg.get(ordered[2], 0)))

        # 8 mejores terceros
        thirds.sort(key=lambda x: (-x[2], -x[3], -elo(x[0])))
        best = thirds[:8]

        # Asignar terceros a slots (greedy: slots con menos opciones primero)
        slots = sorted(third_allowed.items(), key=lambda kv: len(kv[1]))
        avail = {grp: team for team, grp, *_ in best}
        third_at = {}
        for mi, allowed in slots:
            cand = [(g, t) for g, t in avail.items() if g in allowed]
            cand.sort(key=lambda gt: -elo(gt[1]))
            if cand:
                g, t = cand[0]
                third_at[mi] = t
                del avail[g]
        # cualquier slot sin asignar (raro): el mejor tercero restante
        for mi in third_allowed:
            if mi not in third_at and avail:
                g = max(avail, key=lambda gg: elo(avail[gg]))
                third_at[mi] = avail.pop(g)

        def resolve(slot, mi):
            if slot == "3":
                return third_at.get(mi, "—")
            p, g = slot[0], slot[1]
            arr = pos.get(g, [])
            idx = 0 if p == "1" else 1
            return arr[idx] if len(arr) > idx else "—"

        labels = {"1": "1º", "2": "2º", "3": "3º"}

        def slot_label(slot):
            return "Mejor 3º" if slot == "3" else f"{labels[slot[0]]} {slot[1]}"

        # Construir los 16 partidos de 16avos con equipos resueltos
        r32 = []
        for mi, (sa, sb) in enumerate(R32):
            a, b = resolve(sa, mi), resolve(sb, mi)
            r32.append({"a": a, "b": b, "la": slot_label(sa), "lb": slot_label(sb)})

        # Simular una ronda (favorito avanza) y devolver matches con ganador
        def play(matches):
            out = []
            for m in matches:
                a, b = m["a"], m["b"]
                if a == "—" or b == "—":
                    w, pa, pb = (a if b == "—" else b), 0.5, 0.5
                else:
                    p = self._probs(a, b)
                    pa, pb = p["p_home"], p["p_away"]
                    w = a if pa >= pb else b
                out.append({**m, "winner": w, "pa": round(pa, 2), "pb": round(pb, 2)})
            return out

        def next_round(prev):
            nxt = []
            for i in range(0, len(prev), 2):
                nxt.append({"a": prev[i]["winner"], "b": prev[i + 1]["winner"],
                            "la": "", "lb": ""})
            return nxt

        # Lado izquierdo = matches 0..7, derecho = 8..15
        def build_side(r32_side):
            rounds = [play(r32_side)]
            for _ in range(3):                       # R16, QF, SF
                rounds.append(play(next_round(rounds[-1])))
            return rounds

        left = build_side(r32[:8])
        right = build_side(r32[8:])
        fin_a, fin_b = left[-1][0]["winner"], right[-1][0]["winner"]
        if fin_a == "—" or fin_b == "—":
            champ = fin_a if fin_b == "—" else fin_b
            pa = pb = 0.5
        else:
            p = self._probs(fin_a, fin_b)
            pa, pb = p["p_home"], p["p_away"]
            champ = fin_a if pa >= pb else fin_b
        final = {"a": fin_a, "b": fin_b, "winner": champ,
                 "pa": round(pa, 2), "pb": round(pb, 2)}

        # Con quién se cruza cada grupo según salga 1º o 2º
        cruces = {}
        for mi, (sa, sb) in enumerate(R32):
            for slot, other in ((sa, sb), (sb, sa)):
                if slot[0] in ("1", "2"):
                    g = slot[1]
                    cruces.setdefault(g, {})[slot[0]] = (slot_label(other), resolve(other, mi))
        return {"left": left, "right": right, "final": final, "champion": champ, "cruces": cruces}

    def title_path(self, team: str, bracket: dict | None = None) -> list[dict] | None:
        """Camino proyectado de un equipo hasta la final: la llave de su rama en
        cada ronda (con los dos equipos proyectados y el favorito)."""
        b = bracket or self.official_bracket()
        names = ["16avos", "8vos", "4tos", "Semis"]
        for side in (b["left"], b["right"]):
            for i, m in enumerate(side[0]):
                if team in (m["a"], m["b"]):
                    path = []
                    for r in range(len(side)):
                        mm = side[r][i >> r]
                        rival = mm["b"] if mm["a"] == team else mm["a"]
                        path.append({"ronda": names[r], "a": mm["a"], "b": mm["b"],
                                     "winner": mm["winner"], "rival_directo": rival if r == 0 else None})
                    fin = b["final"]
                    path.append({"ronda": "Final", "a": fin["a"], "b": fin["b"],
                                 "winner": fin["winner"], "rival_directo": None})
                    return path
        return None

    def group_scenarios(self, group: str) -> list[dict]:
        """Estado de clasificación de cada equipo de un grupo: puntos, partidos
        jugados/restantes y situación (clasificado / depende / eliminado),
        calculado de forma conservadora con los puntos máximos alcanzables."""
        from ..data.sources import load_fixture, load_standings
        st = load_standings().get(group)
        fx = load_fixture()
        if st is None or fx.empty:
            return []
        teams = list(st["team"])
        gfx = fx[fx["group"] == group]
        played = {t: 0 for t in teams}
        for r in gfx[gfx["played"]].itertuples(index=False):
            if r.home in played:
                played[r.home] += 1
            if r.away in played:
                played[r.away] += 1
        pts = dict(zip(st["team"], st["Pts"]))
        rows = []
        for t in teams:
            rem = 3 - played.get(t, 0)
            maxp = pts.get(t, 0) + rem * 3
            rows.append({"team": t, "pts": int(pts.get(t, 0)), "jugados": played.get(t, 0),
                         "restan": rem, "max_pts": maxp})
        rows.sort(key=lambda r: (-r["pts"], -r["max_pts"]))
        # 2º puesto = umbral de clasificación directa
        second_pts = rows[1]["pts"] if len(rows) > 1 else 0
        for i, r in enumerate(rows):
            if r["restan"] == 0:
                r["estado"] = "Clasificado" if i < 2 else "Eliminado (puede ir como 3º)"
            elif r["max_pts"] < second_pts:
                r["estado"] = "Eliminado (puede ir como 3º)"
            elif i < 2 and r["pts"] > (rows[2]["max_pts"] if len(rows) > 2 else 0):
                r["estado"] = "Clasificado"
            else:
                r["estado"] = "Depende"
        return rows

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
