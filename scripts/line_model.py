"""Win % for every book line (Component 2), from the stats-only model in stat_model.py.

For each player prop line the books posted in the test half of last season (historical odds
from fetch_historical_odds.py), gives each side's chance of winning from the model next to
the books' average odds, the best book's odds, and what actually happened.

The model never sees odds: its win % comes only from its predicted stat. Each prediction
becomes a negative binomial distribution around the predicted mean. By default the stat models
are retrained weekly (see stat_model.py), and the spread is refit at each retraining on the
previous SPREAD_WINDOW_DAYS of out-of-sample predictions, so no game is priced by a model or
spread that saw it. The books' odds are only compared against afterwards. The summaries also
show the model trained once (--frozen uses only that).

Writes {league}_line_bets.csv: one row per side (Over and Under) of each line.

    python scripts/line_model.py [--league nba] [--frozen]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

import stat_model as sm
from build_features import TARGETS
from fetch_player_stats import OUT_DIR, name_key

STAT_FOR_MARKET = TARGETS  # odds market -> stat column
MARKET_LABELS = {"player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists", "player_threes": "Threes"}
PLACEHOLDER = (1.02, 50)  # decimal prices outside this range are placeholders (e.g. Novig's -10000), not real quotes
MAX_MARGIN = 0.15         # two-sided quotes with a bigger margin are stale, e.g. -3333 on both sides (books run ~6-8%)
ALPHAS = np.r_[0, np.geomspace(1e-3, 2, 60)]  # negative binomial dispersion grid (0 = Poisson)
SPREAD_WINDOW_DAYS = 21   # weekly mode: the spread is refit on the out-of-sample predictions from this many days
KEYS = ["game_key", "name_key", "market", "line"]
EV_BUCKETS = [-1, 0, 0.05, 0.1, 0.2, 10]
P_BINS = [0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1]


# ---------- distributions ----------
# Each prediction is a negative binomial with mean mu and dispersion alpha (0 = Poisson).
# alpha is one number, or one per prediction when the spread is refit each week.

def _nb(mu, alpha):
    alpha = np.maximum(alpha, 1e-6)
    n = 1 / alpha
    return nbinom(n, n / (n + mu))


def _poisson(alpha):
    return np.all(np.asarray(alpha) == 0)


def nb_logpmf(y, mu, alpha):
    mu = np.clip(mu, 0.05, None)
    return poisson.logpmf(y, mu) if _poisson(alpha) else _nb(mu, alpha).logpmf(y)


def nb_over(line, mu, alpha):
    """P(stat > line)."""
    mu, k = np.clip(mu, 0.05, None), np.floor(line)
    return poisson.sf(k, mu) if _poisson(alpha) else _nb(mu, alpha).sf(k)


def nb_under(line, mu, alpha):
    """P(stat < line). On whole-number lines the rest is the chance of a push."""
    mu, k = np.clip(mu, 0.05, None), np.ceil(line) - 1
    return poisson.cdf(k, mu) if _poisson(alpha) else _nb(mu, alpha).cdf(k)


def fit_alpha(y, mu):
    """Dispersion that maximises the likelihood of y under NB(mu, alpha)."""
    nll = [-nb_logpmf(y, mu, a).mean() for a in ALPHAS]
    return float(ALPHAS[int(np.argmin(nll))])


def to_us(dec):
    """Decimal odds -> American odds."""
    return np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1)).round().astype(int)


# ---------- odds ----------

def load_odds(league, report=print):
    """Historical odds from fetch_historical_odds.py."""
    path = OUT_DIR / "odds_history" / f"{league}_props_history.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run fetch_historical_odds.py first")
    events = json.loads(path.read_text())
    report(f"Odds: {len(events):,} games from {path.name}")
    return events


def book_quotes(events):
    """One row per book per two-sided prop line, with that book's no-vig P(over).
    Games are keyed by the ESPN game id fetch_historical_odds.py stores with each event."""
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
                        rows.append({"game_key": ev["espn_game_id"], "name_key": name_key(player), "market": m["key"],
                                     "line": line, "book": bm["title"], "over": q["Over"], "under": q["Under"],
                                     "p_over": io / (io + iu)})
    return pd.DataFrame(rows)


def line_consensus(quotes):
    """Per line: books quoting it, their average price and no-vig chance on each side, and the
    best price on each side. Main line = the one most books quote for that player and market."""
    g = quotes.groupby(KEYS)
    lines = g.agg(books=("book", "count"), avg_over=("over", "mean"), avg_under=("under", "mean"),
                  vegas_p_over=("p_over", "mean")).reset_index()
    for side in ("over", "under"):
        best = quotes.loc[g[side].idxmax(), KEYS + [side, "book"]]
        lines = lines.merge(best.rename(columns={side: f"best_{side}", "book": f"best_{side}_book"}), on=KEYS)
    lines["_tie"] = (lines["vegas_p_over"] - 0.5).abs()
    main = lines.sort_values(["books", "_tie"], ascending=[False, True]).groupby(KEYS[:3]).head(1).index
    lines["main_line"] = lines.index.isin(main)
    return lines.drop(columns="_tie")


def fair_odds(p_win, p_push):
    """American odds at which this side breaks even (pushes refund the stake)."""
    p = np.clip(p_win, 0.001, 0.999)
    return to_us((1 - p_push) / p)


def weekly_alphas(stats, t):
    """Spread for each test row, refit at each retraining date on the out-of-sample predictions
    from the SPREAD_WINDOW_DAYS before it: the held-out end of the train half, then the test
    weeks already played (each predicted by models that hadn't seen it)."""
    val, test, preds = stats["val"], stats["split"]["test"], stats["preds"]
    dates = pd.concat([val["game_date"], test["game_date"]]).to_numpy()
    y = np.r_[val[t].to_numpy(), test[t].to_numpy()]
    mu = np.r_[stats["val_preds"][t], preds[f"{t}_pred"].to_numpy()]
    starts = stats["week_starts"]
    alpha = np.empty(len(test))
    for start, end in zip(starts, starts[1:] + [None]):
        rows = ((test["game_date"] >= start) & ((test["game_date"] < end) if end is not None else True)).to_numpy()
        window = (dates >= np.datetime64(start - pd.Timedelta(days=SPREAD_WINDOW_DAYS))) & (dates < np.datetime64(start))
        alpha[rows] = fit_alpha(y[window], mu[window])
    return alpha


def run(league="nba", report=print, stats=None, weekly=True):
    """stats: an existing stat_model.run() result, to skip retraining. weekly: retrain the stat
    models and refit the spread every week (False: train once)."""
    stats = stats or sm.run(league, report, weekly=weekly)
    test, preds, val = stats["split"]["test"], stats["preds"], stats["val"]
    frozen_alphas = {t: fit_alpha(val[t].to_numpy(), stats["val_preds"][t]) for t in STAT_FOR_MARKET.values()}
    if stats["weekly"]:
        alphas = {t: weekly_alphas(stats, t) for t in STAT_FOR_MARKET.values()}
        report("Spread (negative binomial dispersion), first week -> last week: " + ", ".join(
            f"{sm.STAT_NAMES[t]} {a[0]:.3f} -> {a[-1]:.3f}" for t, a in alphas.items()))
    else:
        alphas = {t: np.full(len(test), a) for t, a in frozen_alphas.items()}
        report("Spread (negative binomial dispersion) per stat: " + ", ".join(
            f"{sm.STAT_NAMES[t]} {a:.3f}" for t, a in frozen_alphas.items()))

    events = load_odds(league, report)
    lines = line_consensus(book_quotes(events))
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv", usecols=["game_id", "player_id", "name_key"], dtype=str)
    rows = test[["game_id", "player_id"]].reset_index().merge(logs, on=["game_id", "player_id"], how="left")
    rows = rows.rename(columns={"game_id": "game_key"})

    frames = []
    for market, t in STAT_FOR_MARKET.items():
        j = rows.merge(lines[lines["market"] == market], on=["game_key", "name_key"])
        if j.empty:
            continue
        idx, line = j["index"].to_numpy(), j["line"].to_numpy()
        mu, actual, a = preds[f"{t}_pred"].to_numpy()[idx], preds[t].to_numpy()[idx], alphas[t][idx]
        p_o, p_u = nb_over(line, mu, a), nb_under(line, mu, a)
        p_push = 1 - p_o - p_u  # 0 on half-point lines
        # the frozen model (trained once), for comparison; same as the model when not weekly
        if stats["weekly"]:
            mu_f = preds[f"{t}_pred_Ensemble (frozen)"].to_numpy()[idx]
            p_of, p_uf = nb_over(line, mu_f, frozen_alphas[t]), nb_under(line, mu_f, frozen_alphas[t])
        else:
            p_of, p_uf = p_o, p_u
        info = preds.iloc[idx].reset_index(drop=True)
        for side in ("Over", "Under"):
            s = side.lower()
            p_win = p_o if side == "Over" else p_u
            vegas = j["vegas_p_over"].to_numpy() if side == "Over" else 1 - j["vegas_p_over"].to_numpy()
            best = j[f"best_{s}"].to_numpy()
            out = pd.DataFrame({
                "game_date": info["game_date"], "player": info["player_name"], "team": info["team"],
                "opponent": info["opponent"], "market": MARKET_LABELS[market], "line": line, "side": side,
                "main_line": j["main_line"], "books": j["books"],
                # the books
                "avg_vegas_odds": to_us(j[f"avg_{s}"].to_numpy()),
                "avg_vegas_odds_Decimal": j[f"avg_{s}"].to_numpy(),
                "vegas_implied_p": 1 / j[f"avg_{s}"].to_numpy(),  # with the vig: win rate the average odds need
                "vegas_p_win": vegas,  # vig removed: the books' estimate of the real chance
                # the model (stats only)
                "model_mean": mu,
                "model_p_win": p_win,
                "_p_win_frozen": p_of if side == "Over" else p_uf,  # comparison only; not saved
                "model_odds": fair_odds(p_win, p_push),
                "model_p_push": p_push,
                # best price
                "best_odds": to_us(best),
                "best_odds_Decimal": best,
                "best_book": j[f"best_{s}_book"],
                "breakeven_p": 1 / best,
                # comparison
                "edge": p_win - vegas,
                "ev": p_win * best + p_push - 1,  # expected profit per $1 at the best odds
                "actual": actual,
            })
            won, lost = (actual > line, actual < line) if side == "Over" else (actual < line, actual > line)
            out["outcome"] = np.select([won, lost], ["Win", "Loss"], "Push")
            out["profit_per_1"] = np.select([won, lost], [best - 1, -1.0], 0.0)
            frames.append(out)
    bets = (pd.concat(frames, ignore_index=True)
            .sort_values(["game_date", "player", "market", "line", "side"], ascending=[True, True, True, True, False])
            .reset_index(drop=True))
    report(f"{len(bets):,} bets ({len(bets) // 2:,} lines, {bets['main_line'].sum() // 2:,} main lines) "
           f"on {bets['game_date'].nunique()} dates, {bets['game_date'].min():%Y-%m-%d} to {bets['game_date'].max():%Y-%m-%d}")
    return {"bets": bets, "alphas": alphas, "stats": stats}


# ---------- summaries ----------

def _decided(bets, main_only):
    d = bets[bets["outcome"] != "Push"]
    return d[d["main_line"]] if main_only else d


ACCURACY_COLS = {"Vegas (avg no-vig)": "vegas_p_win", "Model, trained once": "_p_win_frozen",
                 "Model, retrained weekly": "model_p_win"}


def accuracy(bets, main_only=True, by="market"):
    """Brier score of the chance of winning (lower is better): the books' average vs the model,
    trained once and retrained weekly. by: "market" or "month"."""
    d = _decided(bets, main_only)
    won = (d["outcome"] == "Win").astype(float)
    key = d["market"] if by == "market" else d["game_date"].dt.strftime("%Y-%m")
    out = pd.DataFrame({"bets": key.groupby(key).size()})
    for label, col in ACCURACY_COLS.items():
        out[label] = ((d[col] - won) ** 2).groupby(key).mean()
    out.loc["All"] = [len(d), *[((d[col] - won) ** 2).mean() for col in ACCURACY_COLS.values()]]
    return out


def calibration(bets, col="model_p_win", main_only=True):
    """When a source says X% to win, how often did the side win?"""
    d = _decided(bets, main_only)
    b = pd.cut(d[col], P_BINS)
    return d.groupby(b, observed=True).agg(bets=(col, "size"), predicted=(col, "mean"),
                                          actual=("outcome", lambda o: (o == "Win").mean()))


def ev_buckets(bets, main_only=True):
    """Flat $1 on every bet in each model-ev range, at the best odds. Pushes refund the stake."""
    d = bets[bets["main_line"]] if main_only else bets
    b = pd.cut(d["ev"], EV_BUCKETS)
    return d.groupby(b, observed=True).agg(bets=("ev", "size"), model_p_win=("model_p_win", "mean"),
                                          vegas_p_win=("vegas_p_win", "mean"), breakeven=("breakeven_p", "mean"),
                                          win_rate=("outcome", lambda o: (o == "Win").mean()),
                                          mean_ev=("ev", "mean"), return_per_1=("profit_per_1", "mean"))


def save(league, out):
    path = OUT_DIR / f"{league}_line_bets.csv"
    bets = out["bets"]
    bets.drop(columns=[c for c in bets if c.startswith("_")]).to_csv(path, index=False)
    print(f"Saved {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", default="nba")
    ap.add_argument("--frozen", action="store_true", help="train once on the train half instead of weekly")
    args = ap.parse_args()
    out = run(args.league, weekly=not args.frozen)
    b = out["bets"]
    print("\nBrier, main lines (lower is better):")
    print(accuracy(b).round(4).to_string())
    print(accuracy(b, by="month").round(4).to_string())
    print("\nModel calibration, main lines:")
    print(calibration(b).round(3).to_string())
    print("\nReturns by model ev, main lines, $1 flat at best odds:")
    print(ev_buckets(b).round(3).to_string())
    save(args.league, out)


if __name__ == "__main__":
    main()
