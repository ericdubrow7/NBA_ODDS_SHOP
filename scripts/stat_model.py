"""Predict each player's stats for each game (Component 2), on build_features.py outputs.

No odds: this only asks how well we can predict points, rebounds, assists and threes.

Uses last season only, split by date at the midpoint of its regular-season games:

- train: first half of the season. Its last TUNE_SHARE of dates is held out to pick each
  model's settings, then the model is refit on the whole first half.
- test:  second half of the season, never seen while fitting or tuning.

Rows need MIN_PRIOR earlier games so the rolling features mean something.

Writes {league}_stat_model_results.csv (scores by model and stat) and
{league}_stat_predictions.csv (one row per player-game in the test half, actual vs predicted).

    python scripts/stat_model.py [--league nba]
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import feature_columns, target_stats
from fetch_player_stats import OUT_DIR

MIN_PRIOR = 10       # earlier games a row needs
TUNE_SHARE = 0.2     # last share of the train half's dates used to pick settings
ENSEMBLE = ["Ridge regression", "Random forest", "Gradient boosting (Poisson)"]  # averaged into "Ensemble"
STAT_NAMES = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "Threes"}


# ---------- data ----------

def load(league="nba"):
    df = pd.read_csv(OUT_DIR / f"{league}_features_train.csv", dtype={"player_id": str, "game_id": str},
                     parse_dates=["game_date"])
    feats = feature_columns(df)
    df[feats] = df[feats].astype(float)
    return df, feats


def season_split(df):
    """Last completed season, split at the date by which half its games were played."""
    seasons = sorted(df["season"].unique())
    last = seasons[-2] if len(seasons) > 1 else seasons[0]
    s = df[df["season"] == last]
    dates = s.drop_duplicates("game_id")["game_date"].sort_values()
    mid = dates.iloc[len(dates) // 2]
    s = s[s["prior_games"] >= MIN_PRIOR]
    train, test = s[s["game_date"] < mid], s[s["game_date"] >= mid]
    train_dates = np.sort(train["game_date"].unique())
    tune_start = train_dates[int(len(train_dates) * (1 - TUNE_SHARE))]
    return {"season": last, "midpoint": mid, "tune_start": pd.Timestamp(tune_start),
            "train": train.reset_index(drop=True), "test": test.reset_index(drop=True)}


# ---------- models ----------
# Baselines read a feature column. Models: name -> (candidate settings, make(settings)).

BASELINES = {"Last-10 average": "{t}_l10", "Season average": "{t}_season"}

MODELS = {
    "Ridge regression": ([{"alpha": a} for a in (10, 100, 1000)], lambda p: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(**p))),
    "Random forest": ([{"min_samples_leaf": m} for m in (20, 50, 100)], lambda p: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, max_features=0.33, n_jobs=-1, random_state=0, **p))),
    "Gradient boosting (Poisson)": ([{"learning_rate": 0.05, "max_iter": n, "max_leaf_nodes": l, "min_samples_leaf": 50}
                                     for n in (150, 300, 600) for l in (7, 15)],
                                    lambda p: HistGradientBoostingRegressor(loss="poisson", random_state=0, **p)),
}


def scores(y, pred):
    err = pred - y
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "R2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2), "Bias": np.mean(err)}


def tune(make, grid, fit, val, feats, t):
    """Pick the settings with the lowest MAE on the held-out end of the train half.
    Returns them with their predictions for that held-out part."""
    best = None
    for p in grid:
        m = make(p).fit(fit[feats], fit[t])
        pred = np.clip(m.predict(val[feats]), 0, None)
        mae = np.mean(np.abs(pred - val[t]))
        if best is None or mae < best[0]:
            best = (mae, p, pred)
    return best[1], best[2]


def run(league="nba", report=print):
    df, feats = load(league)
    sp = season_split(df)
    train, test = sp["train"], sp["test"]
    fit, val = train[train["game_date"] < sp["tune_start"]], train[train["game_date"] >= sp["tune_start"]]
    report(f"Season {sp['season']}: train {len(train):,} player-games before {sp['midpoint']:%Y-%m-%d} "
           f"(settings tuned on {len(val):,} from {sp['tune_start']:%Y-%m-%d}), test {len(test):,} after")

    preds = test[["game_date", "player_name", "team", "opponent", "home", "pos", "min_l10", *target_stats()]].copy()
    # val_preds: the ensemble's predictions for the held-out end of the train half, from models
    # fit without it. line_model.py uses them to size each prediction's spread.
    rows, settings, fitted, val_preds = [], {}, {}, {}
    for t in target_stats():
        out, val_out = {}, {}
        for name, col in BASELINES.items():
            out[name] = test[col.format(t=t)].fillna(test[f"{t}_l10"]).to_numpy()
        for name, (grid, make) in MODELS.items():
            t0 = time.time()
            p, val_out[name] = tune(make, grid, fit, val, feats, t)
            model = make(p).fit(train[feats], train[t])
            out[name] = np.clip(model.predict(test[feats]), 0, None)
            settings[(t, name)], fitted[(t, name)] = p, model
            report(f"  {STAT_NAMES[t]:<9} {name:<28} {p}  ({time.time() - t0:.0f}s)")
        out["Ensemble"] = np.mean([out[m] for m in ENSEMBLE], axis=0)
        val_preds[t] = np.mean([val_out[m] for m in ENSEMBLE], axis=0)
        for name, pred in out.items():
            rows.append({"stat": STAT_NAMES[t], "model": name, **scores(test[t].to_numpy(), pred)})
            preds[f"{t}_pred_{name}"] = pred
        preds[f"{t}_pred"] = out["Ensemble"]

    results = pd.DataFrame(rows)
    return {"results": results, "preds": preds, "split": sp, "feats": feats,
            "settings": settings, "models": fitted, "val": val, "val_preds": val_preds}


# ---------- summaries ----------

def score_table(results, metric="MAE"):
    """Models x stats for one metric, best first."""
    tab = results.pivot(index="model", columns="stat", values=metric)[list(STAT_NAMES.values())]
    return tab.sort_values("Points", ascending=metric not in ("R2",))


def by_minutes(preds, model="Ensemble"):
    """MAE by the player's recent minutes: are role players or starters harder?"""
    bins = pd.cut(preds["min_l10"], [0, 15, 25, 32, 48], labels=["<15", "15-25", "25-32", "32+"])
    out = {}
    for t, label in STAT_NAMES.items():
        err = (preds[f"{t}_pred_{model}"] - preds[t]).abs()
        out[label] = err.groupby(bins, observed=True).mean()
    tab = pd.DataFrame(out)
    tab.insert(0, "player-games", bins.value_counts().reindex(tab.index))
    return tab


def importance(run_out, stat="pts", model="Gradient boosting (Poisson)", top=15, sample=4000):
    """Permutation importance on the test half: how much MAE rises when a feature is shuffled."""
    test, feats = run_out["split"]["test"], run_out["feats"]
    d = test.sample(min(sample, len(test)), random_state=0)
    r = permutation_importance(run_out["models"][(stat, model)], d[feats], d[stat],
                               scoring="neg_mean_absolute_error", n_repeats=3, random_state=0)
    return pd.Series(r.importances_mean, index=feats).sort_values(ascending=False).head(top).rename("MAE increase")


def save(league, run_out):
    run_out["results"].to_csv(OUT_DIR / f"{league}_stat_model_results.csv", index=False)
    run_out["preds"].to_csv(OUT_DIR / f"{league}_stat_predictions.csv", index=False)
    print(f"Saved {league}_stat_model_results.csv and {league}_stat_predictions.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", default="nba")
    args = ap.parse_args()
    out = run(args.league)
    for metric in ("MAE", "R2", "Bias"):
        print(f"\n{metric} on test half:")
        print(score_table(out["results"], metric).round(3).to_string())
    print("\nEnsemble MAE by recent minutes:")
    print(by_minutes(out["preds"]).round(3).to_string())
    save(args.league, out)


if __name__ == "__main__":
    main()
