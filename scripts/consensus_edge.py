"""Consensus edge (Component 2): bet when one book's price beats the fair price implied by all
the books together. No stats model: the market's own consensus is the model.

For every player prop line in the historical odds (fetch_historical_odds.py), the fair chance
of each side is the books' average no-vig probability. A side is a bet when the best price
across books pays more than that fair chance is worth:

    edge = consensus_p_win * best_decimal_odds - 1     (expected profit per $1 if the consensus is right)

Results come from the box scores; a player who didn't play voids the bet, as books do.

The rule was found on last season's odds (Jan-Apr) and fixed before looking at this season:
    RULE:          main line, MIN_BOOKS+ books quoting, edge > MIN_EDGE
    RULE_NO_NOVIG: the same, skipping bets where Novig has the best price
This season's games are the out-of-sample test.

Writes {league}_consensus_bets.csv: one row per side of each line, with rule flags.

    python scripts/consensus_edge.py [--league nba]
"""
import argparse

import numpy as np
import pandas as pd

from fetch_player_stats import OUT_DIR
from line_model import MARKET_LABELS, STAT_FOR_MARKET, book_quotes, line_consensus, load_odds, to_us

MIN_EDGE = 0.02
MIN_BOOKS = 4
RULES = {"Rule: edge > 2%, 4+ books, main line": "rule",
         "Same, Novig not the best price": "rule_no_novig"}
EDGE_BUCKETS = [-1, 0, 0.02, 0.04, 0.06, 1]
BOOTSTRAPS = 500


def season_label(logs):
    """The season each game belongs to, from the game logs."""
    return logs.drop_duplicates("game_id").set_index("game_id")["season"]


def run(league="nba", report=print):
    events = load_odds(league, report)
    lines = line_consensus(book_quotes(events))
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv", dtype={"game_id": str, "player_id": str},
                       parse_dates=["game_date"])
    logs = logs[logs["season_type"] == 2] if "season_type" in logs else logs
    keep = ["game_id", "name_key", "game_date", "season", "player_name", "team", "opponent", *STAT_FOR_MARKET.values()]
    played = logs[keep].rename(columns={"game_id": "game_key"})
    games_with_logs = set(played["game_key"])
    lines = lines[lines["game_key"].isin(games_with_logs)]  # games not in the logs yet can't be settled
    j = lines.merge(played, on=["game_key", "name_key"], how="left")
    void = j["player_name"].isna()
    report(f"{len(lines):,} lines on {lines['game_key'].nunique():,} games with box scores; "
           f"{void.sum():,} voided (player didn't play)")
    j = j[~void].reset_index(drop=True)

    frames = []
    for side in ("Over", "Under"):
        s = side.lower()
        p_win = j["vegas_p_over"] if side == "Over" else 1 - j["vegas_p_over"]
        best = j[f"best_{s}"]
        actual = np.select([j["market"] == m for m in STAT_FOR_MARKET], [j[t] for t in STAT_FOR_MARKET.values()])
        out = pd.DataFrame({
            "season": j["season"], "game_date": j["game_date"], "player": j["player_name"], "team": j["team"],
            "opponent": j["opponent"], "market": j["market"].map(MARKET_LABELS), "line": j["line"], "side": side,
            "main_line": j["main_line"], "books": j["books"],
            "avg_odds": to_us(j[f"avg_{s}"].to_numpy()),
            "consensus_p_win": p_win,                    # books' average no-vig chance
            "fair_odds": to_us(1 / p_win.clip(0.001, 0.999).to_numpy()),
            "best_odds": to_us(best.to_numpy()), "best_odds_Decimal": best, "best_book": j[f"best_{s}_book"],
            "breakeven_p": 1 / best,
            "edge": p_win * best - 1,                    # expected profit per $1 at the best price if the consensus is right
            "actual": actual,
        })
        line = out["line"].to_numpy()
        won, lost = (actual > line, actual < line) if side == "Over" else (actual < line, actual > line)
        out["outcome"] = np.select([won, lost], ["Win", "Loss"], "Push")
        out["profit_per_1"] = np.select([won, lost], [best - 1, -1.0], 0.0)
        frames.append(out)
    bets = (pd.concat(frames, ignore_index=True)
            .sort_values(["game_date", "player", "market", "line", "side"], ascending=[True, True, True, True, False])
            .reset_index(drop=True))
    bets["rule"] = bets["main_line"] & (bets["books"] >= MIN_BOOKS) & (bets["edge"] > MIN_EDGE)
    bets["rule_no_novig"] = bets["rule"] & (bets["best_book"] != "Novig")
    seasons = sorted(bets["season"].unique())
    bets["sample"] = np.where(bets["season"] == seasons[-1], "This season (test)", "Last season (rule found here)")
    for name, b in bets.groupby("sample"):
        report(f"  {name}: {b['game_date'].min():%Y-%m-%d} to {b['game_date'].max():%Y-%m-%d}, "
               f"{b['game_date'].nunique()} dates, {len(b) // 2:,} lines, {b['rule'].sum():,} rule bets")
    return bets


# ---------- summaries ----------

def _ci(b, seed=0):
    """95% range of the return, resampling whole game dates (bets on the same night are related)."""
    if b.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    groups = [g.to_numpy() for _, g in b.groupby("game_date")["profit_per_1"]]
    means = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(len(groups), size=len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def rule_results(bets):
    """Each rule, last season vs this season: $1 flat at the best price."""
    rows = []
    for label, col in RULES.items():
        for sample, b in bets[bets[col]].groupby("sample", sort=False):
            lo, hi = _ci(b)
            rows.append({"rule": label, "sample": sample, "bets": len(b), "bets per day": len(b) / b["game_date"].nunique(),
                         "win rate": (b["outcome"] == "Win").mean(), "avg edge": b["edge"].mean(),
                         "profit per $1": b["profit_per_1"].sum(), "return": b["profit_per_1"].mean(),
                         "return 95% low": lo, "return 95% high": hi})
    return pd.DataFrame(rows).set_index(["rule", "sample"])


def edge_buckets(bets, sample="This season (test)"):
    """Main lines with 4+ books, grouped by edge: does a bigger gap mean a better return?"""
    d = bets[(bets["sample"] == sample) & bets["main_line"] & (bets["books"] >= MIN_BOOKS)]
    g = d.groupby(pd.cut(d["edge"], EDGE_BUCKETS), observed=True)
    return g.agg(bets=("edge", "size"), avg_edge=("edge", "mean"), win_rate=("outcome", lambda o: (o == "Win").mean()),
                 breakeven=("breakeven_p", "mean"), return_per_1=("profit_per_1", "mean"))


def by_group(bets, col, rule="rule", sample="This season (test)"):
    """Rule bets split by best_book, market or side."""
    d = bets[bets[rule] & (bets["sample"] == sample)]
    return d.groupby(col).agg(bets=("edge", "size"), avg_edge=("edge", "mean"),
                              win_rate=("outcome", lambda o: (o == "Win").mean()),
                              return_per_1=("profit_per_1", "mean")).sort_values("bets", ascending=False)


def daily(bets, rule="rule", sample="This season (test)"):
    """Rule bets per day with running profit ($1 flat)."""
    d = bets[bets[rule] & (bets["sample"] == sample)]
    out = d.groupby("game_date").agg(bets=("edge", "size"), profit=("profit_per_1", "sum"))
    out["running profit"] = out["profit"].cumsum()
    return out


def save(league, bets):
    path = OUT_DIR / f"{league}_consensus_bets.csv"
    bets.to_csv(path, index=False)
    print(f"Saved {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", default="nba")
    args = ap.parse_args()
    bets = run(args.league)
    pd.set_option("display.width", 200)
    print("\nRules, $1 flat at the best price:")
    print(rule_results(bets).round(3).to_string())
    for sample in ("Last season (rule found here)", "This season (test)"):
        print(f"\nEdge buckets, main lines with {MIN_BOOKS}+ books, {sample}:")
        print(edge_buckets(bets, sample).round(3).to_string())
    print("\nThis season's rule bets by best book:")
    print(by_group(bets, "best_book").round(3).to_string())
    save(args.league, bets)


if __name__ == "__main__":
    main()
