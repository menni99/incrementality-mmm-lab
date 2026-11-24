"""Sanity checks for the simulated world. Run with `python test_simulate.py` (or pytest)."""

from functools import partial

import numpy as np
import pandas as pd

from simulate import (
    WorldConfig, analyst_view, hot_markets_scenario, intro_scenario, random_holdout_scenario, simulate_world,
)

SMALL = WorldConfig(n_geos=10, n_days=400)


def test_same_seed_same_world():
    a = simulate_world(SMALL, hot_markets_scenario)[1]
    b = simulate_world(SMALL, hot_markets_scenario)[1]
    pd.testing.assert_frame_equal(a, b)


def test_analyst_view_hides_the_truth():
    _, panel = simulate_world(SMALL, random_holdout_scenario)
    assert not any(col.startswith("true_") for col in analyst_view(panel).columns)


def test_sales_add_up():
    _, panel = simulate_world(SMALL, hot_markets_scenario)
    effects = panel.filter(like="true_effect_").sum(axis=1)
    assert np.allclose(panel["sales"], panel["true_baseline_sales"] + effects)


def test_scenarios_share_one_baseline():
    _, hot = simulate_world(SMALL, hot_markets_scenario)
    _, holdout = simulate_world(SMALL, random_holdout_scenario)
    _, intro = simulate_world(SMALL, partial(intro_scenario, ugc_start="2025-03-01", ugc_end="2025-04-25"))
    for other in (holdout, intro):
        assert np.array_equal(hot["true_baseline_sales"], other["true_baseline_sales"])


def test_one_row_per_region_per_day():
    _, panel = simulate_world(SMALL)
    assert len(panel) == SMALL.n_geos * SMALL.n_days
    assert not panel.duplicated(["geo", "date"]).any()


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok  {name}")
