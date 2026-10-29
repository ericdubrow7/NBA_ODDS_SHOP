"""Refresh data/player_props.json from The Odds API.

Same requests as the NBA player-props notebook, for NBA and WNBA. Reads the key
from the ODDS_API_KEY environment variable so it never ships with the static site.

    ODDS_API_KEY=xxxx python scripts/fetch_odds.py [--hours 48] [--leagues nba,wnba]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Same as LEAGUES in common.js.
LEAGUES = {"nba": "basketball_nba", "wnba": "basketball_wnba"}
MARKETS = "player_points,player_rebounds,player_assists,player_threes"
# Same list as BOOKS in common.js. Billed as one region per 10 books.
BOOKMAKERS = "fanduel,draftkings,betmgm,williamhill_us,betrivers,ballybet,espnbet,hardrockbet,novig"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "player_props.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=48, help="look-ahead window; 0 = all upcoming games")
    parser.add_argument("--leagues", default="nba,wnba", help="comma-separated: nba, wnba")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("Set ODDS_API_KEY")

    sports = [LEAGUES[name.strip().lower()] for name in args.leagues.split(",") if name.strip()]
    events = []
    for sport in sports:
        resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/events", params={"apiKey": api_key})
        if resp.status_code != 200:
            print(f"No events for {sport}: {resp.status_code} {resp.text[:200]}")  # e.g. off-season
            continue
        events += resp.json()
    if args.hours:
        cutoff = datetime.now(timezone.utc) + timedelta(hours=args.hours)
        events = [e for e in events if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= cutoff]
    print(f"Events in window: {len(events)}")

    games_data = []
    for i, event in enumerate(events, start=1):
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{event['sport_key']}/events/{event['id']}/odds",
            params={"apiKey": api_key, "bookmakers": BOOKMAKERS, "markets": MARKETS, "oddsFormat": "decimal", "dateFormat": "iso", "includeLinks": "true"},
        )
        if r.status_code != 200:
            print(f"Failed {event['id']}: {r.status_code} {r.text[:200]}")
        else:
            games_data.append(r.json())
            print(f"[{i}/{len(events)}] {event['id']} remaining={r.headers.get('x-requests-remaining')}")
        time.sleep(1.2)

    if not games_data:
        sys.exit("No odds returned; keeping the existing snapshot.")
    OUT_FILE.write_text(json.dumps(games_data))
    print(f"Saved {len(games_data)} events to {OUT_FILE}")


if __name__ == "__main__":
    main()
