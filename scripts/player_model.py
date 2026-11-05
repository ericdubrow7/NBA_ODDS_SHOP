"""Per-player stat predictions (Component 2): a separate model for each player, compared with
the pooled model in stat_model.py, which learns one set of rules from every player.

Same data and split as stat_model.py (last season, train first half, test second half). For
each player and stat, a ridge regression on that player's own games only, with a dozen
features (the player's form, minutes, home/away, rest, teammates out, opponent). Two versions:

- Player only:              predicts the stat from the player's own games alone.
- Pooled + player adjustment: the pooled ridge model's prediction plus a per-player
                              correction, fitted on that player's misses from the pooled model.

Ridge shrinks each player's coefficients toward zero, so with little data a player-only model
falls back to the player's average, and an adjustment falls back to no adjustment. One
shrinkage strength per stat and version is picked on the held-out end of the train half.
Players with fewer than MIN_PLAYER_GAMES training games get the pooled prediction.

run() returns a stat_model.run()-style result, so line_model.run(stats=...) can price the
book lines with any version.

    python scripts/player_model.py [--league nba]
"""
import argparse

import numpy as np
import pandas as pd

import stat_model as sm
from build_features import target_stats

MIN_PLAYER_GAMES = 15            # training games a player needs for their own model
RIDGE_ALPHAS = (1, 3, 10, 30, 100, 300)
VERSIONS = {"player": "Player only", "adjusted": "Pooled + player adjustment"}


def player_features(t):
    return [f"{t}_l5", f"{t}_l10", f"{t}_season", "min_l3", "min_l10", "starter_rate_l10", "home",
            "back_to_back", "days_rest", "out_min_l10", f"opp_allowed_{t}_vs_lg", "opp_pace_l10"]


# ---------- per-player ridge ----------

def _ridge(X, y, alpha):
    """Ridge on standardized columns with an unpenalized intercept. Returns a predict function."""
    center, scale = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    scale[~(scale > 0)] = 1.0
    center = np.nan_to_num(center)
    Z = np.nan_to_num((X - center) / scale)
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: y.mean() + np.nan_to_num((Xn - center) / scale) @ b


def fit_predict(fit, target, new, t, alpha, base_fit=None, base_new=None):
    """Per player: fit on `fit` rows, predict `new` rows. With base_* given, the model learns the
    player's miss (target - base) and adds it back. NaN where the player has too few games."""
    cols = player_features(t)
    y_all = target - (base_fit if base_fit is not None else 0)
    pred = np.full(len(new), np.nan)
    fit_groups = fit.groupby("player_id").indices
    for pid, new_idx in new.groupby("player_id").indices.items():
        idx = fit_groups.get(pid)
        if idx is None or len(idx) < MIN_PLAYER_GAMES:
            continue
        f = _ridge(fit[cols].to_numpy(dtype=float)[idx], y_all[idx], alpha)
        pred[new_idx] = f(new[cols].to_numpy(dtype=float)[new_idx]) + (base_new[new_idx] if base_new is not None else 0)
    return np.clip(pred, 0, None)


def pooled_ridge(stats, t, fit, new):
    """The pooled ridge model (stat_model's settings) fit on `fit`, predicting `fit` and `new`."""
    grid, make = sm.MODELS["Ridge regression"]
    m = make(stats["settings"][(t, "Ridge regression")]).fit(fit[stats["feats"]], fit[t])
    return np.clip(m.predict(fit[stats["feats"]]), 0, None), np.clip(m.predict(new[stats["feats"]]), 0, None)


def run(league="nba", report=print, stats=None, use="adjusted"):
    """stats: an existing stat_model.run() result. use: which version becomes {stat}_pred
    ("player", "adjusted", or "pooled")."""
    stats = stats or sm.run(league, report, weekly=False)  # per-player models are trained once, so compare frozen to frozen
    sp = stats["split"]
    train, test, val = sp["train"], sp["test"], stats["val"]
    fit = train[train["game_date"] < sp["tune_start"]]
    preds, val_preds = stats["preds"].copy(), {}
    rows, chosen = [], {}
    report(f"Players with {MIN_PLAYER_GAMES}+ training games: {(train['player_id'].value_counts() >= MIN_PLAYER_GAMES).sum()}"
           f" of {train['player_id'].nunique()}; test player-games they cover: "
           f"{test['player_id'].map(train['player_id'].value_counts()).ge(MIN_PLAYER_GAMES).mean():.0%}")

    for t in target_stats():
        pooled_test, pooled_val = preds[f"{t}_pred"].to_numpy(), stats["val_preds"][t]
        base = {"tune": pooled_ridge(stats, t, fit, val), "final": pooled_ridge(stats, t, train, test)}
        val_by_version = {"pooled": pooled_val}
        for key, label in VERSIONS.items():
            def predict(a, fit_rows, new_rows, b):
                return fit_predict(fit_rows, fit_rows[t].to_numpy(), new_rows, t, a,
                                   *(b if key == "adjusted" else (None, None)))
            # shrinkage strength: lowest MAE on the held-out end of the train half (pooled where no model)
            tried = {a: np.where(np.isnan(p := predict(a, fit, val, base["tune"])), pooled_val, p) for a in RIDGE_ALPHAS}
            a = min(tried, key=lambda a: np.mean(np.abs(tried[a] - val[t].to_numpy())))
            chosen[(t, key)], val_by_version[key] = a, tried[a]
            p = predict(a, train, test, base["final"])
            preds[f"{t}_pred_{label}"] = np.where(np.isnan(p), pooled_test, p)
        preds[f"{t}_pred_Pooled (ensemble)"] = pooled_test
        for label in ["Pooled (ensemble)", "Last-10 average", *VERSIONS.values()]:
            rows.append({"stat": sm.STAT_NAMES[t], "model": label,
                         **sm.scores(test[t].to_numpy(), preds[f"{t}_pred_{label}"].to_numpy())})
        if use != "pooled":
            preds[f"{t}_pred"] = preds[f"{t}_pred_{VERSIONS[use]}"]
        val_preds[t] = val_by_version[use]
        report(f"  {sm.STAT_NAMES[t]:<9} shrinkage: player only {chosen[(t, 'player')]}, "
               f"adjustment {chosen[(t, 'adjusted')]}")

    return {**stats, "results": pd.DataFrame(rows), "preds": preds, "val_preds": val_preds,
            "player_alphas": chosen, "use": use}


# ---------- summaries ----------

def by_player(out, stat="pts", min_games=20):
    """Test-half MAE per player: pooled vs each per-player version. Negative 'gain' = per-player worse."""
    p = out["preds"]
    d = pd.DataFrame({"player": p["player_name"], "y": p[stat]})
    for label in ["Pooled (ensemble)", *VERSIONS.values()]:
        d[label] = (p[f"{stat}_pred_{label}"] - p[stat]).abs()
    g = d.groupby("player").agg(games=("y", "size"), **{c: (c, "mean") for c in ["Pooled (ensemble)", *VERSIONS.values()]})
    g = g[g["games"] >= min_games]
    for label in VERSIONS.values():
        g[f"gain: {label}"] = g["Pooled (ensemble)"] - g[label]
    return g.sort_values(f"gain: {VERSIONS['adjusted']}", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", default="nba")
    args = ap.parse_args()
    out = run(args.league)
    for metric in ("MAE", "R2"):
        print(f"\n{metric} on test half:")
        print(sm.score_table(out["results"], metric).round(3).to_string())


if __name__ == "__main__":
    main()
