"""Historical player prop odds from The Odds API, for backtesting the model on many nights.

Pulls one snapshot per completed game, LEAD_MINUTES before tip-off (close to the closing
line), for the last quarter of last regular season and all of this season so far. Games
come from fetch_player_stats.py's game logs, so run that first to include the latest games.

Each game is cached under model_data/odds_history/{league}/ and never fetched again, so an
interrupted run can be restarted without paying twice. All cached games are combined into
model_data/odds_history/{league}_props_history.json, in the same format as
data/player_props.json.

Historical endpoints need a paid plan. Cost per The Odds API docs:
  events lookup: 1 credit per date (0 if no events)
  event odds:    10 x markets returned x regions (BOOKMAKERS bills as 1 region per 10 books)

    python scripts/fetch_historical_odds.py --dry-run         # counts and cost only, no requests
    ODDS_API_KEY=xxxx python scripts/fetch_historical_odds.py [--max-credits 25000] [--reserve 500]
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fetch_odds import BOOKMAKERS, LEAGUES, MARKETS
from fetch_player_stats import OUT_DIR

API = "https://api.the-odds-api.com/v4/historical/sports"
LEAD_MINUTES = 30        # snapshot this long before tip-off
LAST_SEASON_SHARE = 0.25  # last quarter of last season's regular-season games
MATCH_WINDOW = timedelta(hours=3)  # ESPN and Odds API start times must agree within this
REQUEST_SLEEP = 0.3


def odds_cost():
    """Credits per game snapshot, if every market comes back."""
    regions = math.ceil(len(BOOKMAKERS.split(",")) / 10)
    return 10 * len(MARKETS.split(",")) * regions


def nickname(team):
    """'LA Clippers' and 'Los Angeles Clippers' -> 'clippers'; 'Portland Trail Blazers' -> 'blazers'."""
    return team.split()[-1].lower()


# ---------- which games ----------

def target_games(league):
    """Completed games to backtest: last quarter of last regular season, plus this season so far."""
    logs = pd.read_csv(OUT_DIR / f"{league}_player_game_logs.csv",
                       usecols=["game_id", "season", "season_type", "start_utc", "game_date", "team_name", "opponent_name", "home"],
                       dtype={"game_id": str})
    g = logs[logs["home"]].drop_duplicates("game_id").rename(columns={"team_name": "home_team", "opponent_name": "away_team"})
    g = g[g["season_type"] == 2]
    g["start_utc"] = pd.to_datetime(g["start_utc"], utc=True)
    seasons = sorted(g["season"].unique())
    current, previous = seasons[-1], seasons[-2] if len(seasons) > 1 else None
    parts = [g[g["season"] == current]]
    if previous:
        prev = g[g["season"] == previous].sort_values("start_utc")
        parts.insert(0, prev.iloc[int(len(prev) * (1 - LAST_SEASON_SHARE)):])
    games = pd.concat(parts).sort_values("start_utc").reset_index(drop=True)
    return games[["game_id", "season", "game_date", "start_utc", "home_team", "away_team"]]


# ---------- requests ----------

class Client:
    def __init__(self, api_key, sport, max_credits, reserve):
        self.key, self.sport = api_key, sport
        self.max_credits, self.reserve = max_credits, reserve
        self.spent, self.remaining = 0, None
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def can_spend(self, cost):
        if self.spent + cost > self.max_credits:
            return False
        return self.remaining is None or self.remaining - cost >= self.reserve

    def get(self, path, params):
        r = self.session.get(f"{API}/{self.sport}/{path}", params={"apiKey": self.key, **params}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        self.spent += int(float(r.headers.get("x-requests-last", 0) or 0))
        if r.headers.get("x-requests-remaining") is not None:
            self.remaining = int(float(r.headers["x-requests-remaining"]))
        time.sleep(REQUEST_SLEEP)
        return r.json()


def iso(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def match_events(day_games, events):
    """ESPN game_id -> Odds API event, by team nicknames and start time."""
    by_teams = {}
    for ev in events:
        by_teams.setdefault((nickname(ev["home_team"]), nickname(ev["away_team"])), []).append(ev)
    out = {}
    for g in day_games.itertuples():
        for ev in by_teams.get((nickname(g.home_team), nickname(g.away_team)), []):
            if abs(pd.Timestamp(ev["commence_time"]) - g.start_utc) <= MATCH_WINDOW:
                out[g.game_id] = ev
                break
    return out


# ---------- run ----------

def run(league, games, client, cache, report=print):
    """Fetches every uncached game; returns counts. Stops early if a credit limit would be crossed."""
    done = {p.stem for p in cache.glob("*.json")}
    todo = games[~games["game_id"].isin(done)]
    fetched, unmatched, failed = 0, [], []
    for date, day in todo.groupby("game_date", sort=True):
        if not client.can_spend(1 + odds_cost()):
            report(f"Stopping before {date}: credit limit reached (spent {client.spent:,}).")
            return fetched, unmatched, failed, True
        # Events list as of just before the day's first tip-off includes every game that day.
        snap = day["start_utc"].min() - timedelta(minutes=60)
        events = client.get("events", {"date": iso(snap)}).get("data", [])
        matched = match_events(day, events)
        unmatched += [g for g in day["game_id"] if g not in matched]
        for g in day.itertuples():
            ev = matched.get(g.game_id)
            if ev is None:
                continue
            if not client.can_spend(odds_cost()):
                report(f"Stopping at {date}: credit limit reached (spent {client.spent:,}).")
                return fetched, unmatched, failed, True
            when = pd.Timestamp(ev["commence_time"]) - timedelta(minutes=LEAD_MINUTES)
            try:
                res = client.get(f"events/{ev['id']}/odds", {
                    "date": iso(when), "bookmakers": BOOKMAKERS, "markets": MARKETS,
                    "oddsFormat": "decimal", "dateFormat": "iso",
                })
            except RuntimeError as err:
                failed.append((g.game_id, str(err)))
                continue
            data = res.get("data") or {}
            data["snapshot_timestamp"] = res.get("timestamp")
            data["espn_game_id"] = g.game_id
            (cache / f"{g.game_id}.json").write_text(json.dumps(data))
            fetched += 1
        remaining = f", {client.remaining:,} left on key" if client.remaining is not None else ""
        report(f"{date}: {len(matched)}/{len(day)} games matched, {fetched} fetched so far, {client.spent:,} credits spent{remaining}")
    return fetched, unmatched, failed, False


def combine(league, cache):
    """All cached games in one file, same format as data/player_props.json."""
    events = [json.loads(p.read_text()) for p in sorted(cache.glob("*.json"))]
    out = OUT_DIR / "odds_history" / f"{league}_props_history.json"
    out.write_text(json.dumps(events))
    return out, len(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba", choices=list(LEAGUES))
    parser.add_argument("--dry-run", action="store_true", help="count games and estimate credits; makes no requests")
    parser.add_argument("--max-credits", type=int, default=30000, help="stop before spending more than this in one run")
    parser.add_argument("--reserve", type=int, default=500, help="stop before the key's remaining credits drop below this")
    args = parser.parse_args()

    cache = OUT_DIR / "odds_history" / args.league
    cache.mkdir(parents=True, exist_ok=True)
    games = target_games(args.league)
    done = {p.stem for p in cache.glob("*.json")}
    todo = games[~games["game_id"].isin(done)]

    print(f"Snapshot {LEAD_MINUTES} min before tip-off | markets: {MARKETS} | books: {BOOKMAKERS}")
    print(f"Cost per game: {odds_cost()} credits, plus 1 per date for the events lookup")
    for season, s in games.groupby("season"):
        t = todo[todo["season"] == season]
        print(f"  {season}: {len(s):,} games, {s['game_date'].min()} to {s['game_date'].max()} | "
              f"to fetch: {len(t):,} games on {t['game_date'].nunique()} dates")
    estimate = len(todo) * odds_cost() + todo["game_date"].nunique()
    print(f"Already cached: {len(games) - len(todo):,} games")
    print(f"Estimated cost: {estimate:,} credits (less if some markets aren't offered for a game)")
    print(f"Latest game in the logs: {games['game_date'].max()} (ET). Rerun fetch_player_stats.py first to include newer games.")
    if args.dry_run:
        return

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY (historical odds need a paid plan).")
    client = Client(key, LEAGUES[args.league], args.max_credits, args.reserve)
    fetched, unmatched, failed, stopped = run(args.league, games, client, cache)
    out, n = combine(args.league, cache)
    print(f"\nFetched {fetched:,} games for {client.spent:,} credits. Cached total: {n:,} games -> {out.name}")
    if unmatched:
        print(f"{len(unmatched)} games had no matching Odds API event, e.g. {unmatched[:5]}")
    if failed:
        print(f"{len(failed)} requests failed, e.g. {failed[:3]}")
    if stopped:
        print("Stopped early at a credit limit; rerun to continue (cached games are skipped).")


if __name__ == "__main__":
    main()
