"""Model experiments for player prop predictions (Component 2), on build_features.py outputs.

Compares baselines and several regressors for each target stat, with a time-based split:

- train: previous season before VALID_START
- valid: previous season from VALID_START (tunes settings and the spread of each prediction)
- test:  current season so far (models refit on train + valid first)

Each model's predicted mean becomes a negative binomial distribution, so it can be scored
on what the betting step needs: the chance a stat goes over a line. Lines aren't stored
historically, so the proxy line is the player's 10-game average rounded to the nearest .5
below, about where books open.

Against real lines, models are compared bet by bet (each side of each prop) with the books'
no-vig probabilities, using the historical odds from fetch_historical_odds.py (or the single
odds snapshot if there's no history). Bets come in two splits:

- tune: validation games with odds, priced with the train-only models' predictions
- test: current-season games with odds, priced with the refit models

Raw model probabilities are too confident at the extremes, so for betting the model's
probability is blended with the books' no-vig probability. The blend weight (by Brier
score) and an ev cutoff (by return, with at least MIN_TUNE_BETS bets) are chosen on the
tune split only, then applied unchanged to the test split.

Writes the bets twice: {league}_vegas_bets.csv (raw) and {league}_vegas_bets_eval.csv, which
adds the betting-strategy columns from the manual evaluation workbook.

    python scripts/model_experiments.py [--league nba]
"""
import argparse
import json
import time
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import TARGETS, feature_columns, target_stats
from fetch_player_stats import OUT_DIR, ROOT, name_key

ENSEMBLE = ["Random forest", "Gradient boosting", "Gradient boosting (Poisson)"]  # averaged
MIN_PRIOR = 10                 # same filter as the diagram: 10+ games of history
VALID_START = "2026-02-01"     # within the previous season
ODDS_FILE = ROOT / "data" / "player_props.json"


def odds_history_file(league):
    return OUT_DIR / "odds_history" / f"{league}_props_history.json"
STAT_FOR_MARKET = TARGETS      # odds market -> stat column
ALPHAS = np.r_[0, np.geomspace(1e-3, 2, 60)]  # negative binomial dispersion grid (0 = Poisson)
W_MODEL = 0.5                  # weight on the model when blending with Vegas, if there's no tune split to choose it
BLEND_WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)    # candidates for the blend weight (0 = Vegas only)
EV_CUTOFFS = (0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)       # candidates for the tuned ev cutoff
MIN_TUNE_BETS = 100            # a cutoff needs this many tune bets to be picked
# Strategy columns in the eval file (mirrors nba_vegas_bets_Manual Eval.xlsx), plus the tuned cutoff.
STAKE = 10
BET_MIN_EV = 0.0               # bet when ev is above this
CONFIDENT_MIN_EV = 0.3         # "confident" bet when ev is above this

# A model's predictions for a split: mean and negative binomial dispersion.
Pred = namedtuple("Pred", "mu alpha")


# ---------- data ----------

def load(league):
    df = pd.read_csv(OUT_DIR / f"{league}_features_train.csv", dtype={"player_id": str, "game_id": str})
    df = df[df["prior_games"] >= MIN_PRIOR].reset_index(drop=True)
    feats = feature_columns(df)
    df[feats] = df[feats].astype(float)
    return df, feats


def split(df):
    current = df["season"].max()
    prev = df[df["season"] != current]
    return {
        "train": prev[prev["game_date"] < VALID_START],
        "valid": prev[prev["game_date"] >= VALID_START],
        "test": df[df["season"] == current],
    }


# ---------- models ----------
# Each entry: name -> list of candidate settings; make(settings) builds an unfitted model.
# Baselines aren't fitted: they read a feature column.

BASELINES = {"Last-10 average": "{t}_l10", "Last-20 average": "{t}_l20"}


def _linear(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


MODELS = {
    "Ridge regression": ([{"alpha": a} for a in (10, 100, 300, 1000, 3000)], lambda p: _linear(Ridge(**p))),
    "Poisson regression": ([{"alpha": a} for a in (1e-4, 1e-3, 1e-2)], lambda p: _linear(PoissonRegressor(max_iter=1000, **p))),
    "Random forest": ([{"min_samples_leaf": m} for m in (30, 60, 120)], lambda p: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, max_features=0.33, n_jobs=-1, random_state=0, **p))),
    "Gradient boosting": ([{"learning_rate": 0.05, "max_iter": n, "max_leaf_nodes": l, "min_samples_leaf": 50}
                           for n in (100, 200, 400) for l in (7, 15)],
                          lambda p: HistGradientBoostingRegressor(random_state=0, **p)),
    "Gradient boosting (Poisson)": ([{"learning_rate": 0.05, "max_iter": n, "max_leaf_nodes": l, "min_samples_leaf": 50}
                                     for n in (100, 200, 400) for l in (7, 15)],
                                    lambda p: HistGradientBoostingRegressor(loss="poisson", random_state=0, **p)),
}


def _mae(y, mu):
    return float(np.mean(np.abs(y - mu)))


# ---------- distributions ----------

def nb_logpmf(y, mu, alpha):
    mu = np.clip(mu, 0.05, None)
    if alpha == 0:
        from scipy.stats import poisson
        return poisson.logpmf(y, mu)
    n = 1 / alpha
    return nbinom.logpmf(y, n, n / (n + mu))


def nb_over(line, mu, alpha):
    """P(stat > line) for half-point lines."""
    mu = np.clip(mu, 0.05, None)
    k = np.floor(line)
    if alpha == 0:
        from scipy.stats import poisson
        return poisson.sf(k, mu)
    n = 1 / alpha
    return nbinom.sf(k, n, n / (n + mu))


def nb_under(line, mu, alpha):
    """P(stat < line). On whole-number lines the rest is the chance of a push."""
    mu = np.clip(mu, 0.05, None)
    k = np.ceil(line) - 1
    if alpha == 0:
        from scipy.stats import poisson
        return poisson.cdf(k, mu)
    n = 1 / alpha
    return nbinom.cdf(k, n, n / (n + mu))


def fit_alpha(y, mu):
    """Dispersion that maximises the likelihood of y under NB(mu, alpha)."""
    nll = [-nb_logpmf(y, mu, a).mean() for a in ALPHAS]
    return float(ALPHAS[int(np.argmin(nll))])


def p_over(pred, line, idx=slice(None)):
    return nb_over(line, pred.mu[idx], pred.alpha)


def p_under(pred, line, idx=slice(None)):
    return nb_under(line, pred.mu[idx], pred.alpha)


def blend(p_model, p_vegas, w=W_MODEL):
    """Weighted average in log-odds, so the blend of each side's chances still adds to 1."""
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(w * lo(p_model) + (1 - w) * lo(p_vegas))))


def proxy_line(df, t):
    """Player's 10-game average, rounded down to the nearest .5 (at least 0.5)."""
    return np.maximum(np.floor(df[f"{t}_l10"].values) + 0.5, 0.5)


def score(y, mu, alpha, line):
    p = nb_over(line, mu, alpha)
    over = (y > line).astype(float)
    eps = 1e-6
    return {
        "MAE": _mae(y, mu),
        "Log score": float(-nb_logpmf(y, mu, alpha).mean()),
        "Brier (over proxy line)": float(np.mean((p - over) ** 2)),
        "Log loss (over proxy line)": float(-np.mean(over * np.log(p + eps) + (1 - over) * np.log(1 - p + eps))),
    }


# ---------- experiment ----------

def run(league="nba", report=print):
    df, feats = load(league)
    parts = split(df)

    def add(t, name, mu_va, mu_te, settings, t0):
        va, te = parts["valid"], parts["test"]
        alpha = fit_alpha(va[t].values, mu_va)
        preds[t][name], valid_preds[t][name] = Pred(mu_te, alpha), Pred(mu_va, alpha)
        for part, d, mu in (("valid", va, mu_va), ("test", te, mu_te)):
            rows.append({"target": t, "model": name, "split": part, "settings": settings, "alpha": alpha,
                         **score(d[t].values, mu, alpha, proxy_line(d, t)), "seconds": round(time.time() - t0, 1)})
        report(f"  {t:5s} {name:28s} test MAE {rows[-1]['MAE']:.3f}  log score {rows[-1]['Log score']:.3f}")

    report(", ".join(f"{k}: {len(v):,} rows" for k, v in parts.items()))
    trval = pd.concat([parts["train"], parts["valid"]])

    rows, preds = [], {t: {} for t in target_stats()}
    valid_preds = {t: {} for t in target_stats()}
    for t in target_stats():
        tr, va, te = parts["train"], parts["valid"], parts["test"]
        candidates = {name: (None, None) for name in BASELINES} | MODELS
        for name, (grid, make) in candidates.items():
            t0 = time.time()
            if name in BASELINES:
                col = BASELINES[name].format(t=t)
                mu_va, mu_te, best = va[col].values, te[col].values, "-"
            else:
                # Pick settings on validation, then refit on train + valid for the test season.
                fits = []
                for p in grid:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = make(p).fit(tr[feats], tr[t])
                    mu = np.clip(m.predict(va[feats]), 0, None)
                    fits.append((_mae(va[t].values, mu), p, mu))
                _, best, mu_va = min(fits, key=lambda f: f[0])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = make(best).fit(trval[feats], trval[t])
                mu_te = np.clip(m.predict(te[feats]), 0, None)
            add(t, name, mu_va, mu_te, str(best), t0)
        # Ensemble: average of the tree models' predictions.
        t0 = time.time()
        add(t, "Ensemble (RF + GB)", np.mean([valid_preds[t][n].mu for n in ENSEMBLE], axis=0),
            np.mean([preds[t][n].mu for n in ENSEMBLE], axis=0), " + ".join(ENSEMBLE), t0)
    results = pd.DataFrame(rows)
    bets = vegas_bets(league, {"tune": (parts["valid"], valid_preds), "test": (parts["test"], preds)}, report)
    tuning = tune(bets, report) if bets is not None else None
    return results, bets, parts, preds, tuning


def calibration(parts, preds, model, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1)):
    """Predicted vs actual over rate at the proxy lines on the test season, all targets pooled."""
    te = parts["test"]
    frames = []
    for t in target_stats():
        line = proxy_line(te, t)
        p = p_over(preds[t][model], line)
        frames.append(pd.DataFrame({"p": p, "over": (te[t].values > line).astype(float)}))
    c = pd.concat(frames)
    c["bin"] = pd.cut(c["p"], list(bins))
    return c.groupby("bin", observed=True).agg(props=("p", "size"), predicted=("p", "mean"), actual=("over", "mean"))


# ---------- benchmark against the books ----------

MAIN_MODEL = "Ensemble (RF + GB)"
MARKET_LABELS = {"player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists", "player_threes": "Threes"}
PLACEHOLDER = (1.02, 50)  # decimal prices outside this range are placeholders (e.g. Novig's -10000), not real quotes
MAX_MARGIN = 0.15         # two-sided quotes with a bigger margin are stale, e.g. -3333 on both sides (books run ~6-8%)


def to_us(dec):
    return np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1)).round().astype(int)


def load_odds(league, report=print):
    """Historical odds if they've been pulled, else the single snapshot."""
    path = odds_history_file(league)
    if not path.exists():
        path = ODDS_FILE
    if not path.exists():
        return None
    events = json.loads(path.read_text())
    report(f"Odds: {len(events):,} games from {path.name}")
    return events


def book_quotes(events):
    """One row per book per two-sided prop line, with that book's no-vig P(over). Games are keyed
    by ESPN game id when the odds came from fetch_historical_odds.py, else by UTC date."""
    rows = []
    for ev in events:
        game_key = ev.get("espn_game_id") or ev["commence_time"][:10]
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m["key"] not in STAT_FOR_MARKET:
                    continue
                sides = {}
                for o in m.get("outcomes", []):
                    if o.get("point") is not None and PLACEHOLDER[0] <= o.get("price", 0) <= PLACEHOLDER[1]:
                        sides.setdefault((o["description"], o["point"]), {})[o["name"]] = o["price"]
                for (player, line), q in sides.items():
                    if "Over" in q and "Under" in q:
                        io, iu = 1 / q["Over"], 1 / q["Under"]
                        if io + iu - 1 > MAX_MARGIN:
                            continue
                        rows.append({"game_key": game_key, "name_key": name_key(player), "market": m["key"],
                                     "line": line, "book": bm["title"], "over": q["Over"], "under": q["Under"],
                                     "p_over": io / (io + iu)})
    return pd.DataFrame(rows)


def consensus(quotes):
    """Per prop line: books quoting it, average no-vig P(over), and the best price on each side."""
    keys = ["game_key", "name_key", "market", "line"]
    g = quotes.groupby(keys)
    props = g.agg(books=("book", "count"), vegas_p_over=("p_over", "mean")).reset_index()
    for side in ("over", "under"):
        best = quotes.loc[g[side].idxmax(), keys + [side, "book"]]
        props = props.merge(best.rename(columns={side: f"best_{side}", "book": f"best_{side}_book"}), on=keys)
    # Main line: the one most books quote for that player and market (ties: closest to 50/50).
    props["_tie"] = (props["vegas_p_over"] - 0.5).abs()
    main = props.sort_values(["books", "_tie"], ascending=[False, True]).groupby(keys[:3]).head(1).index
    props["main_line"] = props.index.isin(main)
    return props.drop(columns="_tie")


def vegas_bets(league, splits, report=print):
    """One row per bet (each side of each player prop) in the odds that overlap each split's
    games: best price, the books' no-vig chance of winning next to each model's, the expected
    value at the best price, and what actually happened. `splits` maps a split name to its
    feature rows and those rows' predictions."""
    events = load_odds(league, report)
    if not events:
        return None
    quotes = book_quotes(events)
    if quotes.empty:
        return None
    cons = consensus(quotes)
    by_id = all("espn_game_id" in ev for ev in events)
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv", usecols=["game_id", "player_id", "name_key"], dtype=str)

    frames = []
    for split_name, (rows, preds) in splits.items():
        names = rows.reset_index(drop=True).merge(logs, on=["game_id", "player_id"], how="left").reset_index()
        names["game_key"] = names["game_id"] if by_id else pd.to_datetime(names["start_utc"], utc=True).dt.strftime("%Y-%m-%d")
        for market, t in STAT_FOR_MARKET.items():
            j = names.merge(cons[cons["market"] == market], on=["game_key", "name_key"])
            if j.empty:
                continue
            line, idx, actual = j["line"].values, j["index"].values, j[t].values
            main = preds[t][MAIN_MODEL]
            p_push = 1 - nb_over(line, main.mu[idx], main.alpha) - nb_under(line, main.mu[idx], main.alpha)  # 0 on half-point lines
            for side in ("Over", "Under"):
                over = side == "Over"
                win_prob = p_over if over else p_under
                dec = j["best_over" if over else "best_under"].values
                out = pd.DataFrame({
                    "split": split_name,
                    "game_date": j["game_date"], "player": j["player_name"], "team": j["team"], "opponent": j["opponent"],
                    "home": j["home"].astype(bool), "market": MARKET_LABELS.get(market, market), "line": line, "side": side,
                    "books": j["books"], "main_line": j["main_line"],
                    "best_price": to_us(dec), "best_book": j["best_over_book" if over else "best_under_book"],
                    "breakeven_p": 1 / dec,  # win rate needed at the best price (includes the book's margin)
                    "vegas_p_win": j["vegas_p_over"].values if over else 1 - j["vegas_p_over"].values,
                    "model_mean": main.mu[idx],
                    "model_p_win": win_prob(main, line, idx),
                    "model_p_push": p_push,
                    "final_p_win": np.nan,  # set by apply_blend
                })
                out["edge"] = out["model_p_win"] - out["vegas_p_win"]
                out["ev"] = np.nan  # set by apply_blend
                out["actual"] = actual
                won, lost = (actual > line, actual < line) if over else (actual < line, actual > line)
                out["outcome"] = np.select([won, lost], ["Win", "Loss"], "Push")
                out["blend_weight"] = np.nan  # set by apply_blend
                for model, pred in preds[t].items():
                    out[f"p_win: {model}"] = win_prob(pred, line, idx)
                frames.append(out)
    if not frames:
        report("The odds don't overlap any tune or test games; no Vegas comparison.")
        return None
    bets = (pd.concat(frames, ignore_index=True)
            .sort_values(["split", "game_date", "player", "market", "line", "side"], ascending=[False, True, True, True, True, False])
            .reset_index(drop=True))
    apply_blend(bets, W_MODEL)
    for split_name, b in bets.groupby("split", sort=False):
        report(f"  {split_name}: {len(b):,} bets ({len(b) // 2:,} props, {b['main_line'].sum() // 2:,} main lines) "
               f"from {b['game_date'].nunique()} dates")
    return bets


def apply_blend(bets, w):
    """Sets final_p_win (model blended with Vegas at weight w) and ev at the best price, in place."""
    dec = decimal_odds(bets["best_price"])
    bets["final_p_win"] = blend(bets["model_p_win"].values, bets["vegas_p_win"].values, w)
    bets["ev"] = bets["final_p_win"] * dec + bets["model_p_push"] - 1  # expected profit per $1; pushes refund the stake
    bets["blend_weight"] = w


def _split(bets, split):
    return bets[bets["split"] == split] if split and "split" in bets else bets


def tune(bets, report=print):
    """Picks the blend weight (lowest Brier) and ev cutoff (best return with MIN_TUNE_BETS+ bets)
    on the tune split's main lines, then applies the weight to every bet."""
    if (bets["split"] == "tune").sum() == 0:
        report(f"No tune bets; keeping blend weight {W_MODEL} and ev cutoff {BET_MIN_EV}.")
        return {"blend_weight": W_MODEL, "ev_cutoff": BET_MIN_EV, "weights": None, "cutoffs": None}
    weights = blend_sensitivity(bets, BLEND_WEIGHTS, split="tune")
    w = float(weights.loc[weights["Brier"].idxmin(), "model weight"])
    apply_blend(bets, w)
    cutoffs = ev_cutoffs(bets, split="tune")
    ok = cutoffs[cutoffs["bets"] >= MIN_TUNE_BETS]
    cut = float(ok.loc[ok["return"].idxmax(), "ev cutoff"]) if not ok.empty else BET_MIN_EV
    report(f"Tuned on {bets.loc[bets['split'] == 'tune', 'game_date'].nunique()} tune dates: "
           f"blend weight {w:.1f} on the model, ev cutoff {cut}")
    return {"blend_weight": w, "ev_cutoff": cut, "weights": weights, "cutoffs": cutoffs}


def vegas_summary(bets, main_only=False, min_books=1, split="test"):
    """Brier and log loss of P(win) per stat for the books and each model, over every bet. Pushes are dropped."""
    d = _split(bets, split)
    d = d[(d["outcome"] != "Push") & (d["books"] >= min_books)]
    if main_only:
        d = d[d["main_line"]]
    won = (d["outcome"] == "Win").astype(float)
    cols = {"Vegas consensus (no vig)": "vegas_p_win",
            **{c.split(": ", 1)[1]: c for c in d if c.startswith("p_win: ")},
            f"Blend ({d['blend_weight'].iloc[0]:.0%} model, {1 - d['blend_weight'].iloc[0]:.0%} Vegas)" if len(d) else "Blend": "final_p_win"}
    rows = []
    for market, g in d.groupby("market"):
        w = won[g.index]
        for model, col in cols.items():
            p = g[col].clip(1e-6, 1 - 1e-6)
            rows.append({"market": market, "model": model, "bets": len(g), "Brier": float(np.mean((p - w) ** 2)),
                         "Log loss": float(-np.mean(w * np.log(p) + (1 - w) * np.log(1 - p)))})
    return pd.DataFrame(rows)


# ---------- evaluating bets ----------

def decimal_odds(us):
    us = np.asarray(us, dtype=float)
    return np.where(us > 0, 1 + us / 100, 1 + 100 / -us)


def strategies(tuned_ev=None):
    """Column prefix -> ev cutoff for each flat-stake strategy."""
    out = {"": BET_MIN_EV, "confident_": CONFIDENT_MIN_EV}
    if tuned_ev is not None:
        out["tuned_"] = tuned_ev
    return out


def add_eval_columns(bets, stake=STAKE, tuned_ev=None):
    """The strategy columns from the manual evaluation workbook, as values: bet `stake` on
    every bet with ev above each strategy's cutoff (0 and 0.3, plus the tuned cutoff if given).
    max_win is the payout if it wins (stake included); pushes refund the stake."""
    out = bets.copy()
    dec = decimal_odds(out["best_price"])
    for prefix, cut in strategies(tuned_ev).items():
        bet = out["ev"] > cut
        st = np.where(bet, stake, 0)
        out[f"{prefix}bet"] = np.where(bet, "Yes", "No")
        out[f"{prefix}stake"] = st
        out[f"{prefix}max_win"] = st * dec
        out[f"{prefix}actual_win"] = np.select([out["outcome"] == "Win", out["outcome"] == "Push"], [st * dec, st], 0)
        out[f"{prefix}profit"] = out[f"{prefix}actual_win"] - st
    return out


def strategy_summary(ev_bets, main_only=True, split="test", tuned_ev=None):
    """Bets, staked, profit and return for each strategy in the eval columns."""
    d = _split(ev_bets, split)
    d = d[d["main_line"]] if main_only else d
    rows = []
    for prefix, cut in strategies(tuned_ev).items():
        name = f"ev > {cut}" + (" (tuned)" if prefix == "tuned_" else "")
        staked = d[f"{prefix}stake"].sum()
        profit = d[f"{prefix}profit"].sum()
        rows.append({"strategy": name, "bets": int((d[f"{prefix}stake"] > 0).sum()), "staked": staked,
                     "profit": round(profit, 2), "return": profit / staked if staked else np.nan})
    return pd.DataFrame(rows)


def ev_buckets(bets, main_only=True, split="test", edges=(0, .05, .1, .2, .3, 2)):
    """Positive-ev bets grouped by ev: predicted vs actual win rate and return at a flat stake."""
    d = _split(bets, split)
    d = d[(d["ev"] > 0) & (d["main_line"] if main_only else True)].copy()
    dec = decimal_odds(d["best_price"])
    d["profit"] = np.select([d["outcome"] == "Win", d["outcome"] == "Loss"], [dec - 1, -1], 0)
    d["won"] = (d["outcome"] == "Win").astype(float)
    d["ev_bucket"] = pd.cut(d["ev"], list(edges))
    return d.groupby("ev_bucket", observed=True).agg(
        bets=("ev", "size"), final_p_win=("final_p_win", "mean"), vegas_p_win=("vegas_p_win", "mean"),
        breakeven=("breakeven_p", "mean"), actual_win=("won", "mean"), mean_ev=("ev", "mean"), actual_return=("profit", "mean"))


def blend_sensitivity(bets, weights=BLEND_WEIGHTS, main_only=True, split="test"):
    """Brier score, positive-ev bets and their return at each blend weight."""
    d = _split(bets, split)
    d = d[d["main_line"]] if main_only else d
    d = d[d["outcome"] != "Push"]
    dec = decimal_odds(d["best_price"])
    won = (d["outcome"] == "Win").astype(float).values
    rows = []
    for w in weights:
        p = blend(d["model_p_win"].values, d["vegas_p_win"].values, w)
        ev = p * dec - 1
        pick = ev > 0
        profit = np.where(won[pick] == 1, dec[pick] - 1, -1)
        rows.append({"model weight": w, "Brier": float(np.mean((p - won) ** 2)), "positive-ev bets": int(pick.sum()),
                     "return": float(profit.mean()) if pick.any() else np.nan})
    return pd.DataFrame(rows)


def ev_cutoffs(bets, cutoffs=EV_CUTOFFS, main_only=True, split="test"):
    """Bets and flat-stake return for each ev cutoff, at the current blend."""
    d = _split(bets, split)
    d = d[d["main_line"]] if main_only else d
    dec = decimal_odds(d["best_price"])
    profit = np.select([d["outcome"] == "Win", d["outcome"] == "Loss"], [dec - 1, -1], 0)
    rows = []
    for c in cutoffs:
        pick = (d["ev"] > c).values
        rows.append({"ev cutoff": c, "bets": int(pick.sum()), "return": float(profit[pick].mean()) if pick.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba", choices=["nba", "wnba"])
    args = parser.parse_args()
    results, bets, _, _, tuning = run(args.league)
    results.to_csv(OUT_DIR / f"{args.league}_model_results.csv", index=False)
    test = results[results["split"] == "test"].pivot(index="model", columns="target", values="Log score")
    print("\nTest log score (lower is better):\n", test.round(4).to_string())
    if bets is None:
        return
    bets.to_csv(OUT_DIR / f"{args.league}_vegas_bets.csv", index=False)
    evals = add_eval_columns(bets, tuned_ev=tuning["ev_cutoff"])
    evals.to_csv(OUT_DIR / f"{args.league}_vegas_bets_eval.csv", index=False)
    print(f"\nSaved {args.league}_vegas_bets.csv (one row per bet) and {args.league}_vegas_bets_eval.csv (plus strategy columns)")
    for split in ("tune", "test"):
        print(f"\nStrategies on {split} main lines:\n",
              strategy_summary(evals, split=split, tuned_ev=tuning["ev_cutoff"]).round(3).to_string(index=False))
    summary = vegas_summary(bets, main_only=True, min_books=3).pivot(index="model", columns="market", values="Brier")
    print("\nTest Brier of P(win), main lines with 3+ books (lower is better):\n", summary.round(4).to_string())


if __name__ == "__main__":
    main()
