"""Model experiments for player prop predictions (Component 2), on build_features.py outputs.

Compares baselines and several regressors for each target stat, with a time-based split:

- train: previous season before VALID_START
- valid: previous season from VALID_START (tunes settings and the spread of each prediction)
- test:  current season so far (models refit on train + valid first)

Each model's predicted mean becomes a negative binomial distribution, so it can be scored
on what the betting step needs: the chance a stat goes over a line. Lines aren't stored
historically, so the proxy line is the player's 10-game average rounded to the nearest .5
below, about where books open. Where the odds snapshot overlaps the test games, models are
also compared bet by bet (each side of each prop) with the books' no-vig probabilities on the real lines.

    python scripts/model_experiments.py [--league nba]
"""
import argparse
import json
import time
import warnings

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
STAT_FOR_MARKET = TARGETS      # odds market -> stat column
ALPHAS = np.r_[0, np.geomspace(1e-3, 2, 60)]  # negative binomial dispersion grid (0 = Poisson)


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
        preds[t][name], valid_preds[t][name] = (mu_te, alpha), mu_va
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
        add(t, "Ensemble (RF + GB)", np.mean([valid_preds[t][n] for n in ENSEMBLE], axis=0),
            np.mean([preds[t][n][0] for n in ENSEMBLE], axis=0), " + ".join(ENSEMBLE), t0)
    results = pd.DataFrame(rows)
    bets = vegas_bets(league, parts["test"], preds, report)
    return results, bets, parts, preds


def calibration(parts, preds, model, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1)):
    """Predicted vs actual over rate at the proxy lines on the test season, all targets pooled."""
    te = parts["test"]
    frames = []
    for t in target_stats():
        mu, alpha = preds[t][model]
        line = proxy_line(te, t)
        frames.append(pd.DataFrame({"p": nb_over(line, mu, alpha), "over": (te[t].values > line).astype(float)}))
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


def book_quotes(events):
    """One row per book per two-sided prop line, with that book's no-vig P(over)."""
    rows = []
    for ev in events:
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
                        rows.append({"utc_date": ev["commence_time"][:10], "name_key": name_key(player), "market": m["key"],
                                     "line": line, "book": bm["title"], "over": q["Over"], "under": q["Under"],
                                     "p_over": io / (io + iu)})
    return pd.DataFrame(rows)


def consensus(quotes):
    """Per prop line: books quoting it, average no-vig P(over), and the best price on each side."""
    keys = ["utc_date", "name_key", "market", "line"]
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


def vegas_bets(league, test, preds, report=print):
    """One row per bet (each side of each player prop) in the odds snapshot that overlaps the
    test games: best price, the books' no-vig chance of winning next to each model's, the
    expected value at the best price, and what actually happened."""
    if not ODDS_FILE.exists():
        return None
    quotes = book_quotes(json.loads(ODDS_FILE.read_text()))
    if quotes.empty:
        return None
    cons = consensus(quotes)
    test = test.reset_index(drop=True).assign(utc_date=lambda d: pd.to_datetime(d["start_utc"], utc=True).dt.strftime("%Y-%m-%d"))
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv", usecols=["game_id", "player_id", "name_key"], dtype=str)
    names = test.merge(logs, on=["game_id", "player_id"], how="left").reset_index()  # features don't carry name_key

    frames = []
    for market, t in STAT_FOR_MARKET.items():
        j = names.merge(cons[cons["market"] == market], on=["utc_date", "name_key"])
        if j.empty:
            continue
        line, idx, actual = j["line"].values, j["index"].values, j[t].values
        mu, alpha = preds[t][MAIN_MODEL]
        for side in ("Over", "Under"):
            over = side == "Over"
            win_prob = (lambda m, a: nb_over(line, m, a)) if over else (lambda m, a: nb_under(line, m, a))
            dec = j["best_over" if over else "best_under"].values
            p_win = win_prob(mu[idx], alpha)
            p_push = 1 - nb_over(line, mu[idx], alpha) - nb_under(line, mu[idx], alpha)  # 0 on half-point lines
            out = pd.DataFrame({
                "game_date": j["game_date"], "player": j["player_name"], "team": j["team"], "opponent": j["opponent"],
                "home": j["home"].astype(bool), "market": MARKET_LABELS.get(market, market), "line": line, "side": side,
                "books": j["books"], "main_line": j["main_line"],
                "best_price": to_us(dec), "best_book": j["best_over_book" if over else "best_under_book"],
                "breakeven_p": 1 / dec,  # win rate needed at the best price (includes the book's margin)
                "vegas_p_win": j["vegas_p_over"] if over else 1 - j["vegas_p_over"],
                "model_mean": mu[idx],
                "model_p_win": p_win,
            })
            out["edge"] = out["model_p_win"] - out["vegas_p_win"]
            out["ev"] = p_win * dec + p_push - 1  # expected profit per $1 at the best price; pushes refund the stake
            out["actual"] = actual
            won, lost = (actual > line, actual < line) if over else (actual < line, actual > line)
            out["outcome"] = np.select([won, lost], ["Win", "Loss"], "Push")
            for model, (mu_m, a_m) in preds[t].items():
                out[f"p_win: {model}"] = win_prob(mu_m[idx], a_m)
            frames.append(out)
    if not frames:
        report("Odds snapshot doesn't overlap the test games; no Vegas comparison.")
        return None
    bets = (pd.concat(frames, ignore_index=True)
            .sort_values(["game_date", "player", "market", "line", "side"], ascending=[True, True, True, True, False])
            .reset_index(drop=True))
    report(f"Vegas comparison: {len(bets):,} bets ({len(bets) // 2:,} props, {bets['main_line'].sum() // 2:,} main lines) "
           "from the odds snapshot")
    return bets


def vegas_summary(bets, main_only=False, min_books=1):
    """Brier and log loss of P(win) per stat for the books and each model, over every bet. Pushes are dropped."""
    d = bets[(bets["outcome"] != "Push") & (bets["books"] >= min_books)]
    if main_only:
        d = d[d["main_line"]]
    won = (d["outcome"] == "Win").astype(float)
    cols = {"Vegas consensus (no vig)": "vegas_p_win", **{c.split(": ", 1)[1]: c for c in d if c.startswith("p_win: ")}}
    rows = []
    for market, g in d.groupby("market"):
        w = won[g.index]
        for model, col in cols.items():
            p = g[col].clip(1e-6, 1 - 1e-6)
            rows.append({"market": market, "model": model, "bets": len(g), "Brier": float(np.mean((p - w) ** 2)),
                         "Log loss": float(-np.mean(w * np.log(p) + (1 - w) * np.log(1 - p)))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba", choices=["nba", "wnba"])
    args = parser.parse_args()
    results, bets, _, _ = run(args.league)
    results.to_csv(OUT_DIR / f"{args.league}_model_results.csv", index=False)
    test = results[results["split"] == "test"].pivot(index="model", columns="target", values="Log score")
    print("\nTest log score (lower is better):\n", test.round(4).to_string())
    if bets is not None:
        bets.to_csv(OUT_DIR / f"{args.league}_vegas_bets.csv", index=False)
        print(f"\nSaved {args.league}_vegas_bets.csv: one row per bet (each side of each prop)")
        summary = vegas_summary(bets).pivot(index="model", columns="market", values="Brier")
        print("\nVegas comparison Brier of P(win), all bets (lower is better):\n", summary.round(4).to_string())


if __name__ == "__main__":
    main()
