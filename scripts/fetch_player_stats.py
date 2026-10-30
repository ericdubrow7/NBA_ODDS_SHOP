"""Player stats ingestion for the prediction model (Component 2 in the architecture diagram).

Pulls player game logs, upcoming games and the players in them from ESPN's public
scoreboard and box-score API, and writes CSVs to model_data/ (gitignored). No modelling.

ESPN's API is unofficial and undocumented, so it could change without notice. NBA.com's
stats site and CDN block scripted requests, so they aren't used.

Finished days and box scores are cached under model_data/cache/ and never refetched. The
first run downloads about two seasons (~1,500 requests, 10-15 minutes).

    python scripts/fetch_player_stats.py [--league nba] [--seasons-back 1] [--min-games 10] [--hours 48]
"""
import argparse
import json
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "model_data"
ODDS_FILE = ROOT / "data" / "player_props.json"
ET = ZoneInfo("America/New_York")


def name_key(name):
    """Normalised player name for matching across sources: no accents, punctuation or suffixes."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_made(v):
    """'11-18' -> (11, 18)."""
    made, _, att = (v or "").partition("-")
    return (int(made), int(att)) if made.isdigit() and att.isdigit() else (None, None)


def to_num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Ingest:
    def __init__(self, league="nba", seasons_back=1, season_types=(2,), request_sleep=0.2):
        """season_types are ESPN's: 2 = regular season, 3 = playoffs."""
        self.league = league
        self.season_types = set(season_types)
        self.request_sleep = request_sleep
        self.espn = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{league}"
        self.cache = OUT_DIR / "cache" / "espn" / league
        for sub in ("scoreboard", "summary"):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # Retry dropped connections and server hiccups with backoff (1s, 2s, 4s, 8s).
        retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.today = datetime.now(ET).date()
        self.seasons = [self.current_season_year() - i for i in range(seasons_back + 1)]

    # ---------- seasons ----------

    def season_label(self, year):
        """ESPN names a season by its end year: NBA 2027 = 2026-27; WNBA seasons sit in one year."""
        return f"{year - 1}-{str(year)[2:]}" if self.league == "nba" else str(year)

    def season_window(self, year):
        """Calendar dates that can hold games for an ESPN season year."""
        if self.league == "nba":
            return date(year - 1, 10, 1), date(year, 6, 30)
        return date(year, 5, 1), date(year, 10, 31)

    def current_season_year(self):
        t = self.today
        if self.league == "nba":
            return t.year + 1 if t.month >= 8 else t.year
        return t.year if t.month >= 5 else t.year - 1

    # ---------- requests ----------

    def get_json(self, path, params):
        r = self.session.get(f"{self.espn}/{path}", params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.request_sleep)
        return r.json()

    def scoreboard(self, day):
        """Events on an ET calendar day (ESPN rejects date ranges), cached once every game that day is final."""
        f = self.cache / "scoreboard" / f"{day:%Y%m%d}.json"
        if f.exists():
            return json.loads(f.read_text())
        events = self.get_json("scoreboard", {"dates": f"{day:%Y%m%d}"}).get("events", [])
        if day < self.today and all(e["status"]["type"]["completed"] for e in events):
            f.write_text(json.dumps(events))
        return events

    def summary(self, event_id):
        f = self.cache / "summary" / f"{event_id}.json"
        if f.exists():
            return json.loads(f.read_text())
        data = self.get_json("summary", {"event": event_id})
        box = {"boxscore": data.get("boxscore", {})}  # keep only what's used; full summaries are large
        f.write_text(json.dumps(box))
        return box

    # ---------- steps ----------

    def completed_games(self, report=print):
        completed = []
        for year in self.seasons:
            start, end = self.season_window(year)
            end = min(end, self.today - timedelta(days=1))
            day, n = start, 0
            while day <= end:
                for e in self.scoreboard(day):
                    if e["season"]["year"] == year and e["season"]["type"] in self.season_types and e["status"]["type"]["completed"]:
                        completed.append(e)
                        n += 1
                day += timedelta(days=1)
            report(f"{self.season_label(year)}: {n} completed games")
        return completed

    def player_rows(self, event):
        comp = event["competitions"][0]
        sides = {t["team"]["id"]: t for t in comp["competitors"]}
        start = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        rows = []
        for team in self.summary(event["id"])["boxscore"].get("players", []):
            tid = team["team"]["id"]
            opp = next(t for k, t in sides.items() if k != tid)
            for block in team["statistics"]:
                labels = block["labels"]
                for a in block["athletes"]:
                    if a.get("didNotPlay") or not a.get("stats"):
                        continue
                    s = dict(zip(labels, a["stats"]))
                    minutes = to_num(s.get("MIN"))
                    if not minutes:
                        continue
                    fgm, fga = split_made(s.get("FG"))
                    fg3m, fg3a = split_made(s.get("3PT"))
                    ftm, fta = split_made(s.get("FT"))
                    rows.append({
                        "league": self.league.upper(),
                        "season": self.season_label(event["season"]["year"]),
                        "season_type": event["season"]["type"],
                        "game_id": event["id"],
                        "game_date": start.astimezone(ET).date().isoformat(),
                        "start_utc": start.isoformat(),
                        "team": team["team"]["abbreviation"],
                        "team_name": team["team"]["displayName"],
                        "opponent": opp["team"]["abbreviation"],
                        "opponent_name": opp["team"]["displayName"],
                        "home": sides[tid]["homeAway"] == "home",
                        "player_id": a["athlete"]["id"],
                        "player_name": a["athlete"]["displayName"],
                        "name_key": name_key(a["athlete"]["displayName"]),
                        "position": (a["athlete"].get("position") or {}).get("abbreviation"),
                        "starter": a.get("starter", False),
                        "minutes": minutes,
                        "pts": to_num(s.get("PTS")),
                        "reb": to_num(s.get("REB")),
                        "ast": to_num(s.get("AST")),
                        "fg3m": fg3m, "fg3a": fg3a,
                        "fgm": fgm, "fga": fga,
                        "ftm": ftm, "fta": fta,
                        "oreb": to_num(s.get("OREB")),
                        "dreb": to_num(s.get("DREB")),
                        "stl": to_num(s.get("STL")),
                        "blk": to_num(s.get("BLK")),
                        "tov": to_num(s.get("TO")),
                        "pf": to_num(s.get("PF")),
                        "plus_minus": to_num((s.get("+/-") or "").replace("+", "")),
                    })
        return rows

    def game_logs(self, completed, report=print):
        """One row per player per game; players who didn't play are dropped."""
        rows, failed = [], []
        for i, e in enumerate(completed, 1):
            try:
                rows.extend(self.player_rows(e))
            except Exception as err:  # one bad box score shouldn't stop the run
                failed.append((e["id"], repr(err)))
            if i % 200 == 0:
                report(f"  {i}/{len(completed)} games")
        logs = (pd.DataFrame(rows)
                .drop_duplicates(["game_id", "player_id"])
                .sort_values(["game_date", "game_id", "team", "player_name"])
                .reset_index(drop=True))
        report(f"{len(logs):,} player-games from {logs['game_id'].nunique():,} games, {logs['player_id'].nunique():,} players")
        if failed:
            report(f"{len(failed)} box scores failed, e.g. {failed[:3]}")
        return logs

    def upcoming_games(self, lookahead_hours=48, report=print):
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=lookahead_hours)
        upcoming = []
        day = self.today
        while day <= cutoff.astimezone(ET).date():
            for e in self.scoreboard(day):
                start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
                if e["status"]["type"]["state"] != "pre" or not (now <= start <= cutoff):
                    continue
                teams = {t["homeAway"]: t["team"] for t in e["competitions"][0]["competitors"]}
                upcoming.append({
                    "game_id": e["id"],
                    "start_utc": start.isoformat(),
                    "start_et": start.astimezone(ET).strftime("%a %b %d %I:%M %p"),
                    "home": teams["home"]["abbreviation"], "home_name": teams["home"]["displayName"],
                    "away": teams["away"]["abbreviation"], "away_name": teams["away"]["displayName"],
                })
            day += timedelta(days=1)
        games = pd.DataFrame(upcoming, columns=["game_id", "start_utc", "start_et", "home", "home_name", "away", "away_name"])
        report(f"{len(games)} games in the next {lookahead_hours} hours")
        return games

    def rosters(self, teams, report=print):
        """Current roster and injury status for each team abbreviation. Not cached: rosters change daily."""
        ids = {t["team"]["abbreviation"]: t["team"]["id"]
               for t in self.get_json("teams", {})["sports"][0]["leagues"][0]["teams"]}
        rows, missing = [], []
        for abbr in sorted(teams):
            try:
                athletes = self.get_json(f"teams/{ids[abbr]}/roster", {})["athletes"]
            except (KeyError, requests.RequestException):
                missing.append(abbr)
                continue
            for a in athletes:
                injuries = a.get("injuries") or []
                rows.append({
                    "player_id": a["id"],
                    "player_name": a["displayName"],
                    "position": (a.get("position") or {}).get("abbreviation"),
                    "team": abbr,
                    "injury_status": injuries[0].get("status") if injuries else None,
                })
        if missing:
            report(f"No roster for {missing}; using each player's most recent team from the game logs instead")
        return pd.DataFrame(rows, columns=["player_id", "player_name", "position", "team", "injury_status"]), missing

    def upcoming_players(self, logs, games, min_games=10, report=print):
        """Players on the current rosters of teams in upcoming games, with their games played.
        Injured players are listed with injury_status rather than removed."""
        current = self.season_label(self.seasons[0])
        counts = logs.groupby("player_id").agg(games=("game_id", "nunique"), last_game=("game_date", "max"))
        counts["games_this_season"] = logs[logs["season"] == current].groupby("player_id")["game_id"].nunique()

        side = pd.concat([
            games.rename(columns={"home": "team", "away": "opponent"}).assign(home=True),
            games.rename(columns={"away": "team", "home": "opponent"}).assign(home=False),
        ])[["game_id", "start_utc", "start_et", "team", "opponent", "home"]]

        roster, missing = self.rosters(set(side["team"]), report)
        if missing:  # fall back to the team from each player's latest game
            latest = logs.sort_values("start_utc").groupby("player_id").tail(1)
            fallback = latest[latest["team"].isin(missing)][["player_id", "player_name", "position", "team"]]
            roster = pd.concat([roster, fallback.assign(injury_status=None)], ignore_index=True)

        roster["name_key"] = roster["player_name"].map(name_key)
        players = (roster.join(counts, on="player_id")
                   .merge(side, on="team", how="inner"))
        players[["games", "games_this_season"]] = players[["games", "games_this_season"]].fillna(0).astype(int)
        players["eligible"] = players["games"] >= min_games
        players = players[["player_id", "player_name", "name_key", "position", "team", "injury_status", "games",
                           "last_game", "games_this_season", "game_id", "start_utc", "start_et", "opponent", "home", "eligible"]]
        players = players.sort_values(["start_utc", "team", "eligible", "games"], ascending=[True, True, False, False]).reset_index(drop=True)

        report(f"{len(players)} players on teams playing soon; {players['eligible'].sum()} have {min_games}+ games "
               f"({len(self.seasons)} season(s) counted)")
        report(f"Only {(players['games_this_season'] >= min_games).sum()} have {min_games}+ games this season alone")
        return players

    def odds_name_check(self, logs, odds_file=ODDS_FILE, report=print):
        """Odds-snapshot player names with no match in the stats; those need an alias before joining."""
        if not Path(odds_file).exists():
            report(f"No odds snapshot at {odds_file}; skipping.")
            return []
        sport_key = f"basketball_{self.league}"
        events = json.loads(Path(odds_file).read_text())
        odds_names = {o["description"] for ev in events if ev.get("sport_key", "basketball_nba") == sport_key
                      for bm in ev.get("bookmakers", []) for m in bm.get("markets", []) for o in m.get("outcomes", [])
                      if o.get("description")}
        known = set(logs["name_key"])
        unmatched = sorted(n for n in odds_names if name_key(n) not in known)
        report(f"{len(odds_names) - len(unmatched)} of {len(odds_names)} odds player names match the stats")
        if unmatched:
            report(f"Unmatched: {unmatched}")
        return unmatched

    def save(self, logs, games, players, report=print):
        for name, df in (("player_game_logs", logs), ("upcoming_games", games), ("upcoming_players", players)):
            path = OUT_DIR / f"{self.league}_{name}.csv"
            df.to_csv(path, index=False)
            report(f"Saved {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba", choices=["nba", "wnba"])
    parser.add_argument("--seasons-back", type=int, default=1, help="previous seasons to include alongside the current one")
    parser.add_argument("--playoffs", action="store_true", help="include playoff games")
    parser.add_argument("--min-games", type=int, default=10, help="games needed to get a prediction")
    parser.add_argument("--hours", type=float, default=48, help="look-ahead window for upcoming games")
    args = parser.parse_args()

    ing = Ingest(args.league, args.seasons_back, (2, 3) if args.playoffs else (2,))
    print("League:", args.league.upper(), "| Seasons:", [ing.season_label(y) for y in ing.seasons], "| Today (ET):", ing.today)
    logs = ing.game_logs(ing.completed_games())
    games = ing.upcoming_games(args.hours)
    players = ing.upcoming_players(logs, games, args.min_games)
    ing.odds_name_check(logs)
    ing.save(logs, games, players)


if __name__ == "__main__":
    main()
