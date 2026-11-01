"""
simulate.py: builds the fake marketing world for the incrementality study.

We write the "laws of physics" for a pretend brand, so we KNOW the true causal
effect of UGC spend and can grade every measurement method against it.

The world is built in layers, one small function each:
  1. Baseline sales: what each region sells with no marketing at all   <- built
  2. Adstock: ad effects that carry over across days                   <- built
  3. Saturation: diminishing returns on spend                          <- built
  4. Channels + halo: UGC, search, social; UGC lifts branded search    <- built
     plus what each ad platform REPORTS (last-click credit)
  5. Budget plans: who gets UGC, when, and how much                     <- built
     A. hot_markets_plan: UGC goes to regions that look hot (the confounding trap)
     B. holdout_plan:     a coin flip picks the UGC regions (randomized experiment)
     C. test_market_plan: UGC in a few test markets (for synthetic control)
     D. always_on_plan:   nationwide UGC flights of varying size (for the MMM)

Columns starting with "true_" are the hidden truth. A real analyst would only
ever see the date, region, spend, platform-reported revenue and total sales.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Channel:
    """How one marketing channel turns spend into extra sales.

    Sizes are relative to each region's typical daily sales, so a big region
    needs proportionally more spend before it saturates.
    """

    carryover: float   # share of ad memory that survives to the next day (layer 2)
    half_point: float  # ad memory giving half the max effect, as a share of daily sales (layer 3)
    shape: float       # 1 = diminishing returns, above 1 = S-curve (layer 3)
    max_lift: float    # the most extra sales the channel can ever add, as a share of daily sales


def default_channels() -> dict[str, Channel]:
    return {
        # Buyer is already looking: effect is almost immediate.
        "search": Channel(carryover=0.2, half_point=0.05, shape=1.0, max_lift=0.11),
        # In between: some lingering, plain diminishing returns.
        "social": Channel(carryover=0.4, half_point=0.08, shape=1.0, max_lift=0.16),
        # Awareness lingers, and people need a few exposures before it sinks in.
        # Spend sits around the steep middle of its S-curve, so flights of different
        # sizes reach the flat start, the steep part and the plateau.
        "ugc": Channel(carryover=0.7, half_point=0.08, shape=2.0, max_lift=0.045),
    }


@dataclass
class WorldConfig:
    """Every knob of the simulated world, in one place."""

    n_geos: int = 50                 # number of regions (think: US metro areas)
    start_date: str = "2024-01-01"
    n_days: int = 730                # two years of daily data
    seed: int = 42                   # same seed -> same world, every time

    # --- Baseline sales (layer 1) ---
    median_daily_sales: float = 20_000  # a typical region's daily revenue ($)
    size_spread: float = 0.6            # how unequal regions are in size
    mean_growth: float = 0.05           # average yearly growth (+5%)
    growth_spread: float = 0.08         # some regions grow much faster/slower
    weekend_lift: float = 0.15          # Sat/Sun sell ~15% more
    seasonal_amplitude: float = 0.20    # yearly swing of +/-20%, peaking in December
    noise_sd: float = 0.05              # fresh day-to-day randomness (~5%)
    shock_sd: float = 0.04              # "sticky" regional shocks: local events lasting days to weeks
    cluster_shock_sd: float = 0.03      # shocks shared by neighbouring regions (weather, local economy)
    shock_persistence: float = 0.9      # share of yesterday's shock that carries into today
    n_clusters: int = 5                 # groups of neighbouring regions
    growth_drift_sd: float = 0.04       # each region's trend wanders by ~4% a year

    # --- Marketing channels (layer 4) ---
    channels: dict[str, Channel] = field(default_factory=default_channels)
    ugc_halo: float = 0.5               # for every $1 UGC sells directly, another $0.50 arrives via branded search
    ugc_response_spread: float = 0.3    # UGC works better in some regions than others
    ugc_response_growth_link: float = 0.5  # ...and tends to work better in faster-growing regions

    # --- What ad platforms report (last-click attribution) ---
    # Share of each channel's TRUE extra sales that ends with a click on that channel's ad.
    click_share: dict[str, float] = field(default_factory=lambda: {"search": 1.0, "social": 0.5, "ugc": 0.1})
    # Share of sales that would have happened anyway but still end with an ad click
    # (e.g. loyal customers who google the brand and click the top search ad).
    baseline_click_share: dict[str, float] = field(default_factory=lambda: {"search": 0.08, "social": 0.015, "ugc": 0.0})


def make_geos(cfg: WorldConfig, rng: np.random.Generator) -> pd.DataFrame:
    """One row per region: size, hidden growth rate, cluster, and hidden UGC responsiveness.

    Growth rate is the "momentum" a marketer can partly see. Later, budget
    will follow it, which is exactly what creates the confounding trap.
    UGC responsiveness multiplies UGC's effect in that region (average 1). It is
    partly linked to growth, so UGC works a bit better where sales are growing.
    """
    # Lognormal = most regions are mid-sized, a few are huge (like real metros).
    size = cfg.median_daily_sales * np.exp(rng.normal(0, cfg.size_spread, cfg.n_geos))
    growth = rng.normal(cfg.mean_growth, cfg.growth_spread, cfg.n_geos)
    cluster = rng.integers(0, cfg.n_clusters, cfg.n_geos)
    link = cfg.ugc_response_growth_link
    z = link * (growth - cfg.mean_growth) / cfg.growth_spread + np.sqrt(1 - link**2) * rng.normal(0, 1, cfg.n_geos)
    response = np.exp(cfg.ugc_response_spread * z - cfg.ugc_response_spread**2 / 2)
    return pd.DataFrame({
        "geo": [f"geo_{i:02d}" for i in range(cfg.n_geos)],
        "base_sales": size,
        "growth_rate": growth,
        "cluster": cluster,
        "ugc_response": response,
    })


def sticky_shocks(n_series: int, n_days: int, sd: float, persistence: float, rng: np.random.Generator) -> np.ndarray:
    """Random shocks that linger: today's shock = persistence x yesterday's + a fresh bump.

    With persistence 0.9 a shock halves in about a week. sd is the typical size
    of the shock at any moment.
    """
    fresh = rng.normal(0, sd * np.sqrt(1 - persistence**2), (n_series, n_days))
    out = np.empty_like(fresh)
    out[:, 0] = rng.normal(0, sd, n_series)
    for t in range(1, n_days):
        out[:, t] = persistence * out[:, t - 1] + fresh[:, t]
    return out


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Retail spikes: Black Friday weekend, Cyber Monday, the run-up to Christmas."""
    mult = np.ones(len(dates))
    for year in dates.year.unique():
        nov1 = pd.Timestamp(year, 11, 1)
        thanksgiving = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7 + 21)  # 4th Thursday
        black_friday = thanksgiving + pd.Timedelta(days=1)
        bumps = {
            black_friday: 1.6,
            black_friday + pd.Timedelta(days=1): 1.3,
            black_friday + pd.Timedelta(days=2): 1.2,
            black_friday + pd.Timedelta(days=3): 1.4,  # Cyber Monday
        }
        for day, lift in bumps.items():
            mult[dates == day] = lift
        pre_xmas = (dates >= pd.Timestamp(year, 12, 10)) & (dates <= pd.Timestamp(year, 12, 23))
        mult[pre_xmas] = 1.25
    return mult


def baseline_sales(cfg: WorldConfig, geos: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Daily sales per region with zero marketing.

    Each piece is a multiplier, so effects scale with region size:
        sales = size x growth x weekday x season x holiday x noise
    where noise has four parts: fresh daily randomness, sticky regional shocks,
    shocks shared by neighbouring regions, and a slow drift in each region's trend.
    """
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    t_years = np.arange(cfg.n_days) / 365.25

    # Shape (regions, days): each row is one region's two-year history.
    growth = np.exp(np.outer(geos["growth_rate"], t_years))
    weekday = np.where(dates.dayofweek.to_numpy() >= 5, 1 + cfg.weekend_lift, 1.0)
    season = 1 + cfg.seasonal_amplitude * np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 335) / 365.25)
    holiday = holiday_multiplier(dates)
    # All noise is on the log scale, so each part is a percentage wobble.
    fresh = rng.normal(0, cfg.noise_sd, (cfg.n_geos, cfg.n_days))
    regional = sticky_shocks(cfg.n_geos, cfg.n_days, cfg.shock_sd, cfg.shock_persistence, rng)
    shared = sticky_shocks(cfg.n_clusters, cfg.n_days, cfg.cluster_shock_sd, cfg.shock_persistence, rng)
    shared = shared[geos["cluster"].to_numpy()]
    drift = np.cumsum(rng.normal(0, cfg.growth_drift_sd / np.sqrt(365.25), (cfg.n_geos, cfg.n_days)), axis=1)
    noise = np.exp(fresh + regional + shared + drift)

    sales = geos["base_sales"].to_numpy()[:, None] * growth * weekday * season * holiday * noise

    return pd.DataFrame({
        "geo": np.repeat(geos["geo"].to_numpy(), cfg.n_days),
        "date": np.tile(dates, cfg.n_geos),
        "baseline_sales": sales.ravel(),
    })


def adstock(spend: np.ndarray, carryover: float) -> np.ndarray:
    """Ad effects that linger after the day the ad runs.

    Each day: ad memory = today's spend + carryover x yesterday's ad memory.
    With carryover 0.7, $10,000 spent on Monday counts as $10,000 on Monday,
    $7,000 on Tuesday, $4,900 on Wednesday, and so on, fading toward zero.

    Works on one series (days,) or many regions at once (regions, days).
    """
    spend = np.asarray(spend, dtype=float)
    out = np.empty_like(spend)
    memory = np.zeros(spend.shape[:-1])
    for t in range(spend.shape[-1]):
        memory = spend[..., t] + carryover * memory
        out[..., t] = memory
    return out


def saturation(ad_memory: np.ndarray, half_point: float, shape: float = 1.0) -> np.ndarray:
    """Diminishing returns: turns ad memory into a share (0 to 1) of the maximum effect.

    Uses the Hill curve, standard in marketing mix models:
        share = x^shape / (x^shape + half_point^shape)
      - half_point: the ad memory at which you get half of the maximum effect
      - shape = 1: every extra dollar does a bit less than the one before
      - shape > 1: S-curve; small budgets barely register, then the effect
        kicks in, then it flattens out
    """
    x = np.maximum(np.asarray(ad_memory, dtype=float), 0)
    return x**shape / (x**shape + half_point**shape)


def marketing_effects(spend: dict[str, np.ndarray], geos: pd.DataFrame, cfg: WorldConfig) -> dict[str, np.ndarray]:
    """Extra daily sales each channel causes, per region and day.

    For each channel: spend -> adstock (lingering memory) -> saturation (diminishing
    returns) -> extra sales. UGC's effect is scaled by each region's hidden
    responsiveness. UGC also gets a halo: people who see UGC search for the brand
    and buy through a search ad, so those sales LOOK like search sales even
    though UGC caused them.
    """
    size = geos["base_sales"].to_numpy()[:, None]
    effects = {}
    for name, ch in cfg.channels.items():
        memory = adstock(spend[name], ch.carryover)
        share = saturation(memory / size, ch.half_point, ch.shape)
        effects[name] = ch.max_lift * size * share
    effects["ugc"] = effects["ugc"] * geos["ugc_response"].to_numpy()[:, None]
    effects["ugc_halo"] = cfg.ugc_halo * effects["ugc"]
    return effects


def platform_reports(effects: dict[str, np.ndarray], baseline: np.ndarray, cfg: WorldConfig,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Revenue each ad platform claims under last-click attribution.

    A platform gets credit when the last thing a buyer clicked was its ad. That
    includes some sales the ad didn't cause (loyal customers clicking a brand
    search ad), and for search, the UGC halo. UGC videos are rarely clicked
    right before buying, so UGC gets credit for very little.
    """
    reports = {}
    for name in ["search", "social", "ugc"]:
        claimed = cfg.click_share[name] * effects[name] + cfg.baseline_click_share[name] * baseline
        if name == "search":
            claimed = claimed + effects["ugc_halo"]
        reports[name] = claimed * np.exp(rng.normal(0, 0.05, claimed.shape))  # tracking is a bit noisy
    return reports


LEVELS = {"search": 0.05, "social": 0.04, "ugc": 0.02}  # daily spend as a share of region size


def _size(geos: pd.DataFrame) -> np.ndarray:
    return geos["base_sales"].to_numpy()[:, None]


def _between(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= start) & (dates <= end), dtype=float)


def always_on(geos: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator, level: float,
              weekly_swing: float = 0.15, q4_boost: float = 0.30) -> np.ndarray:
    """A channel that runs all the time, the way real budgets do.

    The national budget moves up and down week to week (~15%), and rises ~30%
    in Q4. That Q4 rise is a quiet trap: spend goes up exactly when sales would
    rise anyway.
    """
    week = np.arange(len(dates)) // 7
    swing = np.exp(rng.normal(0, weekly_swing, week.max() + 1))[week]
    q4 = np.where(dates.month.to_numpy() >= 10, 1 + q4_boost, 1.0)
    return level * _size(geos) * (swing * q4)[None, :]


def _with_always_on(geos, dates, rng, ugc: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "search": always_on(geos, dates, rng, LEVELS["search"]),
        "social": always_on(geos, dates, rng, LEVELS["social"]),
        "ugc": ugc,
    }


def simple_plan(geos: pd.DataFrame, dates: pd.DatetimeIndex, rng: np.random.Generator,
                ugc_start: str, ugc_end: str) -> dict[str, np.ndarray]:
    """Flat search and social, UGC on everywhere for one flight. Used to introduce the channels."""
    ones = np.ones((len(geos), len(dates)))
    return {
        "search": LEVELS["search"] * _size(geos) * ones,
        "social": LEVELS["social"] * _size(geos) * ones,
        "ugc": LEVELS["ugc"] * _size(geos) * _between(dates, ugc_start, ugc_end)[None, :],
    }


def hot_markets_plan(geos, dates, rng, n_markets: int = 20, start: str = "2025-03-01",
                     end: str = "2025-11-30", read_noise: float = 0.03) -> dict[str, np.ndarray]:
    """Plan A, the trap: launch UGC in the regions that LOOK like they're growing fastest.

    The marketer can't see true growth, only a noisy read of it (reports, gut
    feel). They pick the top regions on that read, so UGC lands mostly in
    regions that were going to grow anyway. It stops a month before the data
    ends, so the fading tail of its effect is still inside the data.
    """
    perceived = geos["growth_rate"].to_numpy() + rng.normal(0, read_noise, len(geos))
    chosen = np.argsort(perceived)[-n_markets:]
    on = np.zeros((len(geos), len(dates)))
    on[chosen] = _between(dates, start, end)
    return _with_always_on(geos, dates, rng, LEVELS["ugc"] * _size(geos) * on)


def holdout_plan(geos, dates, rng, start: str = "2025-03-01", end: str = "2025-04-25",
                 share: float = 0.5) -> dict[str, np.ndarray]:
    """Plan B, the experiment: a coin flip decides which regions get UGC for an 8-week test."""
    treated = rng.permutation(len(geos))[: round(share * len(geos))]
    on = np.zeros((len(geos), len(dates)))
    on[treated] = _between(dates, start, end)
    return _with_always_on(geos, dates, rng, LEVELS["ugc"] * _size(geos) * on)


def test_market_plan(geos, dates, rng, n_markets: int = 3, start: str = "2025-04-01",
                     end: str = "2025-06-30") -> dict[str, np.ndarray]:
    """Plan C: try UGC in a few typical, mid-sized markets for three months."""
    by_size = np.argsort(geos["base_sales"].to_numpy())
    middle = by_size[len(geos) // 5 : -len(geos) // 5]  # skip the smallest and largest fifth
    chosen = rng.choice(middle, n_markets, replace=False)
    on = np.zeros((len(geos), len(dates)))
    on[chosen] = _between(dates, start, end)
    return _with_always_on(geos, dates, rng, LEVELS["ugc"] * _size(geos) * on)


def always_on_plan(geos, dates, rng, flight_weeks: int = 6, gap_weeks: int = 6,
                   low: float = 0.2, high: float = 2.0, skip_chance: float = 0.15) -> dict[str, np.ndarray]:
    """Plan D: UGC flights everywhere, at different sizes in different regions.

    Six weeks on, six weeks off. In each flight, each region gets its own budget,
    from 0.2x to 2x the usual level, and occasionally sits a flight out. That
    variety lets an MMM see many points on UGC's response curve, from the flat
    start of the S to its plateau.
    """
    week = np.arange(len(dates)) // 7
    cycle = flight_weeks + gap_weeks
    flight_id = week // cycle
    in_flight = (week % cycle) < flight_weeks
    n_flights = flight_id.max() + 1
    intensity = rng.uniform(low, high, (len(geos), n_flights))
    intensity[rng.random((len(geos), n_flights)) < skip_chance] = 0
    ugc = LEVELS["ugc"] * _size(geos) * intensity[:, flight_id] * in_flight[None, :]
    return _with_always_on(geos, dates, rng, ugc)


def simulate_world(cfg: WorldConfig | None = None, plan=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the world. Returns (geos, daily panel).

    plan: optional function (geos, dates, rng) -> {channel: spend array}. Without
    one, the world has baseline sales only. The plan gets its own random stream,
    so every plan runs on exactly the same baseline world.
    """
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    geos = make_geos(cfg, rng)
    panel = baseline_sales(cfg, geos, rng)
    if plan is None:
        return geos, panel

    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    plan_rng = np.random.default_rng([cfg.seed, 1])
    spend = plan(geos, dates, plan_rng)
    effects = marketing_effects(spend, geos, cfg)
    baseline = panel["baseline_sales"].to_numpy().reshape(cfg.n_geos, cfg.n_days)
    reports = platform_reports(effects, baseline, cfg, plan_rng)
    for name, values in spend.items():
        panel[f"spend_{name}"] = values.ravel()
    for name, values in reports.items():
        panel[f"reported_{name}"] = values.ravel()
    for name, values in effects.items():
        panel[f"true_effect_{name}"] = values.ravel()
    panel["sales"] = panel["baseline_sales"] + sum(v.ravel() for v in effects.values())
    panel = panel.rename(columns={"baseline_sales": "true_baseline_sales"})
    return geos, panel


def true_iroas(panel: pd.DataFrame, channel: str) -> float:
    """The real answer: extra revenue caused per $1 of spend (UGC includes its halo)."""
    caused = panel[f"true_effect_{channel}"].sum()
    if channel == "ugc":
        caused += panel["true_effect_ugc_halo"].sum()
    return caused / panel[f"spend_{channel}"].sum()


VISIBLE_COLUMNS = ["geo", "date", "spend_search", "spend_social", "spend_ugc",
                   "reported_search", "reported_social", "reported_ugc", "sales"]


def analyst_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Only what a real analyst would get: no hidden "true_" columns."""
    return panel[VISIBLE_COLUMNS].copy()
