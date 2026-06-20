"""Invariantes de la simulación del torneo."""
from src.simulation.tournament import TournamentSimulator


def test_seed_positions_is_permutation():
    for n in (2, 4, 8, 16, 32):
        pos = TournamentSimulator._seed_positions(n)
        assert sorted(pos) == list(range(1, n + 1)), f"n={n} no es permutación"


def test_seed_positions_top_seeds_opposite_sides():
    # En un cuadro de 32, el sembrado 1 y el 2 deben estar en mitades opuestas
    pos = TournamentSimulator._seed_positions(32)
    half = len(pos) // 2
    assert (pos.index(1) < half) != (pos.index(2) < half)


def test_norm3_sums_to_one():
    arr = TournamentSimulator._norm3(0.3, 0.3, 0.3)
    assert abs(arr.sum() - 1.0) < 1e-9
    arr0 = TournamentSimulator._norm3(0, 0, 0)  # degenerado -> uniforme
    assert abs(arr0.sum() - 1.0) < 1e-9


def test_run_probabilities_monotonic(ensemble):
    """Avanzar de ronda implica haber pasado las anteriores: P decreciente."""
    cfg = {"groups": {chr(65 + i): [f"T{i}", f"U{i}", f"V{i}", f"W{i}"] for i in range(12)},
           "knockout_temperature": 1.0}
    res = TournamentSimulator(ensemble, cfg, seed=1).run(n_sims=200)
    for r in res.itertuples(index=False):
        assert r.P_16avos >= r.P_8vos >= r.P_4tos >= r.P_semis >= r.P_final >= r.P_campeon - 1e-9


def test_run_champion_probs_sum_to_one(ensemble):
    """La suma de P_campeon sobre todos los equipos debe ser ~1 (un solo campeón)."""
    cfg = {"groups": {chr(65 + i): [f"T{i}", f"U{i}", f"V{i}", f"W{i}"] for i in range(12)},
           "knockout_temperature": 1.0}
    res = TournamentSimulator(ensemble, cfg, seed=2).run(n_sims=300)
    assert abs(res["P_campeon"].sum() - 1.0) < 1e-9


def test_bracket_order_handles_non_power_of_two(ensemble):
    """Recorta a la mayor potencia de 2 sin romper."""
    sim = TournamentSimulator(ensemble, {"groups": {}}, seed=3)
    order = sim._bracket_order([f"T{i}" for i in range(20)])  # 20 -> 16
    assert len(order) == 16
