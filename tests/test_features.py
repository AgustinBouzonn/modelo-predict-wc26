"""Tests de feature engineering: estructura y ausencia de fuga de datos."""
from src.features.build_features import (FEATURE_COLS, build_training_table,
                                         current_form)


def test_build_training_table_columns_and_no_nan(synthetic_matches):
    t = build_training_table(synthetic_matches)
    assert all(c in t.columns for c in FEATURE_COLS)
    assert not t[FEATURE_COLS].isna().any().any()
    assert len(t) == len(synthetic_matches)


def test_first_match_elo_diff_is_zero(synthetic_matches):
    # Sin partidos previos, ambos equipos parten del rating base -> elo_diff = 0
    t = build_training_table(synthetic_matches)
    assert abs(t.iloc[0]["elo_diff"]) < 1e-9


def test_current_form_structure(synthetic_matches):
    form = current_form(synthetic_matches)
    assert len(form) > 0
    for v in form.values():
        assert {"form_pts", "form_gf", "form_ga"} <= set(v)
        assert 0.0 <= v["form_pts"] <= 3.0
