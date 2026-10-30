"""Feature engineering for the prediction model (Component 2), from fetch_player_stats.py outputs.

Builds one row per player-game with the features known before tip-off, for every stat
we get prop lines for:

- model_data/{league}_features_train.csv: past games, with the actual stats as targets.
- model_data/{league}_features_upcoming.csv: upcoming games, same features, no targets.

Every row's features come from the player's and teams' state after their most recent
earlier game, so a row never sees its own result, and training and upcoming rows are
built the same way. The one exception is teammates out: past games use the box score
(who didn't play), upcoming games use the injury report (who is listed Out).
No modelling happens here.

    python scripts/build_features.py [--league nba]
"""
import argparse

import pandas as pd

from fetch_player_stats import OUT_DIR, season_label, season_year

# Odds API market -> stat column to predict. Only markets the site fetches (see common.js).
TARGETS = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
}
# Combo markets are a sum of stat columns, e.g. "player_points_rebounds_assists": ("pts", "reb", "ast").
COMBOS = {}

VOLUME = ["fga", "fg3a", "fta", "oreb", "dreb", "tov"]  # shot and rebound chances behind the targets
WINDOWS = (5, 10, 20)
POSITIONS = ("G", "F", "C")
OUT_STATS = ["min_l10", "pts_l10", "reb_l10", "ast_l10", "fga_l10"]  # summed over absent teammates
ROTATION_GAMES = 10   # absent teammates count if they played in one of the team's last N games this season
ROTATION_MIN = 15     # ...and averaged this many minutes over their last 10, so benchwarmers who sit don't count
LEAGUE_WINDOW = 300   # team-games in the league average (about 10 per team)
ROW_KEYS = ["player_id", "player_name", "team", "opponent", "home", "game_id", "game_date", "start_utc", "season", "pos"]


def target_stats():
    return list(TARGETS.values()) + list(COMBOS)


def pos_group(p):
    """ESPN positions (G, F, C, PG, SF, G-F...) -> G, F or C by the primary position."""
    p = str(p).split("-")[0].upper() if pd.notna(p) else ""
    return "C" if p == "C" else "F" if p.endswith("F") else "G" if p.endswith("G") else None


def load(league):
    ids = {"player_id": str, "game_id": str}
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv", dtype=ids)
    upcoming = pd.read_csv(OUT_DIR / f"{league}_upcoming_players.csv", dtype=ids)
    for df in (logs, upcoming):
        df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
        df["pos"] = df["position"].map(pos_group)
    for name, cols in COMBOS.items():
        logs[name] = logs[list(cols)].sum(axis=1)
    return logs, upcoming


def _roll(g, col, w, fn="mean", min_periods=1):
    """Rolling stat per group over the last w games, including the current one."""
    levels = list(range(len(g.keys))) if isinstance(g.keys, list) else 0
    return getattr(g[col].rolling(w, min_periods=min_periods), fn)().reset_index(level=levels, drop=True)


def _season_mean(gs, col):
    return gs[col].expanding().mean().reset_index(level=[0, 1], drop=True)


# ---------- player ----------

def player_snapshots(logs):
    """Each player's state after each game (that game included)."""
    df = logs.sort_values(["player_id", "start_utc"]).reset_index(drop=True)
    df["starter"] = df["starter"].astype(float)
    # Usage: the player's share of team plays (shots, free-throw trips, turnovers) while on court.
    tm = df.groupby(["game_id", "team"])[["fga", "fta", "tov", "minutes"]].transform("sum")
    df["plays"] = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["team_plays_on_court"] = df["minutes"] * (tm["fga"] + 0.44 * tm["fta"] + tm["tov"]) / (tm["minutes"] / 5)

    g = df.groupby("player_id")
    gs = df.groupby(["player_id", "season"])
    snap = df[["player_id", "start_utc", "season", "game_date", "team"]].rename(
        columns={"season": "snap_season", "game_date": "last_game_date", "team": "snap_team"})

    for s in target_stats():
        for w in WINDOWS:
            snap[f"{s}_l{w}"] = _roll(g, s, w)
        snap[f"{s}_std_l10"] = _roll(g, s, 10, "std", min_periods=3)
        snap[f"{s}_per36_l10"] = _roll(g, s, 10, "sum") / _roll(g, "minutes", 10, "sum") * 36
        snap[f"{s}_season"] = _season_mean(gs, s)

    for w in (3, *WINDOWS):
        snap[f"min_l{w}"] = _roll(g, "minutes", w)
    snap["min_season"] = _season_mean(gs, "minutes")
    for s in VOLUME:
        snap[f"{s}_l10"] = _roll(g, s, 10)
    snap["fg_pct_l20"] = _roll(g, "fgm", 20, "sum") / _roll(g, "fga", 20, "sum")
    snap["fg3_pct_l20"] = _roll(g, "fg3m", 20, "sum") / _roll(g, "fg3a", 20, "sum")
    snap["usage_l10"] = 100 * _roll(g, "plays", 10, "sum") / _roll(g, "team_plays_on_court", 10, "sum")
    snap["starter_rate_l10"] = _roll(g, "starter", 10)
    snap["games_total"] = g.cumcount() + 1
    snap["games_season"] = gs.cumcount() + 1
    return snap


# ---------- team, league, position ----------

def team_games(logs):
    """One row per team per game: totals, possessions, and what the opponent scored."""
    stats = target_stats() + ["fga", "fta", "oreb", "tov"]
    tg = logs.groupby(["game_id", "team", "start_utc", "season"], as_index=False)[stats].sum()
    tg["poss"] = tg["fga"] - tg["oreb"] + tg["tov"] + 0.44 * tg["fta"]
    other = tg[["game_id", "team", "poss", *target_stats()]].rename(
        columns={"team": "opp", "poss": "opp_poss", **{s: f"allowed_{s}" for s in target_stats()}})
    tg = tg.merge(other, on="game_id")
    tg = tg[tg["team"] != tg["opp"]].sort_values(["team", "start_utc"]).reset_index(drop=True)
    tg["pace"] = (tg["poss"] + tg["opp_poss"]) / 2
    return tg


def team_snapshots(tg):
    """Each team's state after each game: pace, and what its opponents scored against it."""
    g = tg.groupby("team")
    gs = tg.groupby(["team", "season"])
    snap = tg[["team", "start_utc", "season"]].rename(columns={"season": "team_snap_season"})
    snap["pace_l10"] = _roll(g, "pace", 10)
    for s in target_stats():
        snap[f"allowed_{s}_l10"] = _roll(g, f"allowed_{s}", 10)
        snap[f"allowed_{s}_season"] = _season_mean(gs, f"allowed_{s}")
    return snap


def league_snapshots(tg):
    """League-wide average allowed per team-game over the most recent team-games."""
    lg = tg.sort_values("start_utc").reset_index(drop=True)
    snap = lg[["start_utc"]].copy()
    for s in target_stats():
        snap[f"lg_allowed_{s}"] = lg[f"allowed_{s}"].rolling(LEAGUE_WINDOW, min_periods=30).mean()
    return snap


def position_snapshots(logs):
    """What each team allowed to guards, forwards and centers: per-36 rates over its last 20 games."""
    d = (logs.dropna(subset=["pos"])
         .groupby(["opponent", "game_id", "start_utc", "pos"], as_index=False)[target_stats() + ["minutes"]].sum()
         .rename(columns={"opponent": "def_team"})
         .sort_values(["def_team", "pos", "start_utc"]).reset_index(drop=True))
    g = d.groupby(["def_team", "pos"])
    snap = d[["def_team", "pos", "start_utc"]].copy()
    mins = _roll(g, "minutes", 20, "sum", min_periods=5)
    for s in target_stats():
        snap[f"opp_pos_{s}_per36_l20"] = _roll(g, s, 20, "sum", min_periods=5) / mins * 36
    return snap


# ---------- teammates out ----------

def _team_game_index(logs):
    """Team games numbered within each season, so last season's players don't carry into a new one."""
    tgames = logs[["team", "season", "game_id", "start_utc"]].drop_duplicates().sort_values(["team", "start_utc"])
    tgames["k"] = tgames.groupby(["team", "season"]).cumcount()
    return tgames


def _absent_totals(cands, psnap):
    """Sums recent production of absent rotation teammates whose last game was for that team."""
    cols = ["player_id", "start_utc", "snap_team", "starter_rate_l10", *OUT_STATS]
    v = pd.merge_asof(cands.sort_values("start_utc"), psnap[cols].sort_values("start_utc"), on="start_utc",
                      by="player_id", allow_exact_matches=False, direction="backward")
    v = v[(v["snap_team"] == v["team"]) & (v["min_l10"] >= ROTATION_MIN)]
    v["starter"] = v["starter_rate_l10"] >= 0.5
    agg = v.groupby(["team", "game_id"]).agg(
        **{f"out_{c}": (c, "sum") for c in OUT_STATS}, out_starters=("starter", "sum"), out_players=("player_id", "count"))
    return agg.reset_index()


def teammates_out_train(logs, psnap):
    """Past games: rotation players (played in one of the team's last N games) missing from the box score."""
    tgames = _team_game_index(logs)
    app = logs[["team", "game_id", "player_id"]].merge(tgames[["team", "season", "game_id", "k"]], on=["team", "game_id"])
    pool = (pd.concat([app.assign(k=app["k"] + i) for i in range(1, ROTATION_GAMES + 1)])[["team", "season", "k", "player_id"]]
            .drop_duplicates()
            .merge(tgames, on=["team", "season", "k"]))
    played = app[["team", "game_id", "player_id"]].assign(played=True)
    cands = pool.merge(played, on=["team", "game_id", "player_id"], how="left")
    return _absent_totals(cands[cands["played"].isna()][["team", "game_id", "start_utc", "player_id"]], psnap)


def teammates_out_upcoming(logs, upcoming, psnap):
    """Upcoming games: rotation players listed Out on the injury report. Day-to-day counts as playing."""
    tgames = _team_game_index(logs)
    tgames = tgames[tgames["season"].isin(upcoming["season"].unique())]
    recent = (logs.merge(tgames.groupby("team").tail(ROTATION_GAMES)[["team", "game_id"]])
              [["team", "player_id"]].drop_duplicates())
    out = upcoming[upcoming["injury_status"] == "Out"][["team", "game_id", "start_utc", "player_id"]]
    return _absent_totals(out.merge(recent, on=["team", "player_id"]), psnap)


# ---------- assemble ----------

def attach_features(rows, t, absent):
    """Joins each row to the latest snapshots strictly before its start time. `t` holds the snapshot tables."""
    asof = dict(on="start_utc", allow_exact_matches=False, direction="backward")
    rows = rows.sort_values("start_utc").reset_index(drop=True)
    out = pd.merge_asof(rows, t["player"].sort_values("start_utc"), by="player_id", **asof)

    # Season averages and counts only carry within a season; at a new season they start empty.
    new_season = out["snap_season"] != out["season"]
    season_cols = [c for c in t["player"].columns if c.endswith("_season") and c != "snap_season"]
    out.loc[new_season, season_cols] = float("nan")
    out["games_season"] = out["games_season"].where(~new_season, 0)
    out["prior_games"] = out.pop("games_total").fillna(0).astype(int)
    out["prior_games_season"] = out.pop("games_season").fillna(0).astype(int)

    days = (pd.to_datetime(out["game_date"]) - pd.to_datetime(out.pop("last_game_date"))).dt.days
    out["days_rest"] = days.clip(upper=10)  # off-season gaps read as fully rested
    out["back_to_back"] = days == 1
    out["home"] = out["home"].astype(bool)
    for p in POSITIONS:
        out[f"pos_{p}"] = out["pos"] == p
    out = out.drop(columns=["snap_season", "snap_team"])

    team = t["team"].sort_values("start_utc")
    own = team[["team", "start_utc", "pace_l10"]].rename(columns={"pace_l10": "team_pace_l10"})
    out = pd.merge_asof(out.sort_values("start_utc"), own, by="team", **asof)
    opp = team.rename(columns={c: f"opp_{c}" for c in team.columns if c not in ("team", "start_utc")})
    out = pd.merge_asof(out.sort_values("start_utc"), opp, left_by="opponent", right_by="team",
                        suffixes=("", "_opp_snap"), **asof).drop(columns="team_opp_snap")
    new_opp_season = out.pop("opp_team_snap_season") != out["season"]
    out.loc[new_opp_season, [c for c in out.columns if c.startswith("opp_allowed_") and c.endswith("_season")]] = float("nan")

    # Opponent defense relative to the league: above 1 means it allows more than average.
    out = pd.merge_asof(out.sort_values("start_utc"), t["league"].sort_values("start_utc"), **asof)
    for s in target_stats():
        out[f"opp_allowed_{s}_vs_lg"] = out[f"opp_allowed_{s}_l10"] / out.pop(f"lg_allowed_{s}")

    out = pd.merge_asof(out.sort_values("start_utc"), t["position"].sort_values("start_utc"),
                        left_by=["opponent", "pos"], right_by=["def_team", "pos"], **asof).drop(columns="def_team")

    out = out.merge(absent, on=["team", "game_id"], how="left")
    out_cols = [c for c in absent.columns if c.startswith("out_")]
    out[out_cols] = out[out_cols].fillna(0)
    return out.sort_values(["start_utc", "team", "player_name"]).reset_index(drop=True)


def feature_columns(df):
    """Model inputs: everything that isn't an identifier, a target, or upcoming-only info."""
    skip = set(ROW_KEYS) | set(target_stats()) | {"injury_status", "eligible"}
    return [c for c in df.columns if c not in skip]


def build(league="nba", report=print):
    logs, upcoming = load(league)
    tg = team_games(logs)
    psnap = player_snapshots(logs)
    tables = {"player": psnap, "team": team_snapshots(tg), "league": league_snapshots(tg),
              "position": position_snapshots(logs)}

    train = attach_features(logs[ROW_KEYS + target_stats()], tables, teammates_out_train(logs, psnap))

    up = upcoming.copy()
    up["game_date"] = up["start_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    up["season"] = [season_label(league, season_year(league, pd.Timestamp(d).date())) for d in up["game_date"]]
    up = attach_features(up[ROW_KEYS + ["injury_status", "eligible"]], tables, teammates_out_upcoming(logs, up, psnap))

    feats = feature_columns(train)
    report(f"{len(feats)} features for targets {target_stats()}")
    report(f"Train: {len(train):,} player-games ({(train['prior_games'] >= 10).sum():,} with 10+ prior games)")
    report(f"Upcoming: {len(up):,} player-games ({up['eligible'].sum():,} eligible)")
    return train, up, feats


def save(league, train, up, report=print):
    for name, df in (("features_train", train), ("features_upcoming", up)):
        path = OUT_DIR / f"{league}_{name}.csv"
        df.to_csv(path, index=False)
        report(f"Saved {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba", choices=["nba", "wnba"])
    args = parser.parse_args()
    train, up, _ = build(args.league)
    save(args.league, train, up)


if __name__ == "__main__":
    main()
