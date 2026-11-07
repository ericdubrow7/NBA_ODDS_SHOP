# NBA & WNBA Prop Line Shopper

A static web version of `NBA Odds Analysis - Player Props Avg vs Best Book.ipynb`.

For every NBA and WNBA player-prop line (Points, Rebounds, Assists, Threes), it:

1. Collects Over/Under prices from FanDuel, DraftKings, BetMGM, Caesars, BetRivers, Bally Bet, theScore Bet (formerly ESPN BET), Hard Rock Bet and Novig.
2. Averages the decimal odds across books offering the same player, market, side and line.
3. Picks the book with the best price and shows how far it sits above the average.

The **Arbitrage** page (`arbitrage.html`) pairs the best Over with the best Under for each line. It lists the pairs where betting both sides locks in a profit, and shows how to split a stake.

The **Picks** page (`picks.html`) lists bets where a book's price beats the fair price implied by all the books together (their average no-vig chance), the same logic as `scripts/consensus_edge.py`. Every row shows whether it meets each rule: edge over 2%, 4+ books quoting, main line, and Novig not the best price. You can filter by minimum edge and by which rules a pick meets. The rule isn't a proven edge: it lost money in this season's test so far, which the page says up front.

Pick NBA, WNBA or both under *Data source & settings*. The choice applies to every page and decides which leagues a live fetch requests.

There's no server and no build step. It's just HTML, CSS and JavaScript files plus a data snapshot.

## Files

| Path | Purpose |
| --- | --- |
| `index.html`, `app.js` | Line Shopper page (all analysis runs in the browser) |
| `arbitrage.html`, `arbitrage.js` | Arbitrage page |
| `picks.html`, `picks.js` | Picks page: bets that beat the books' consensus fair price, with a check per rule |
| `common.js`, `styles.css` | Shared by every page: sportsbook list, data loading, settings, styles |
| `data/player_props.json` | Snapshot loaded on page open, both leagues (same format the notebook saves) |
| `scripts/fetch_odds.py` | Refreshes the snapshot from The Odds API (key from `ODDS_API_KEY`) |
| `.github/workflows/refresh-odds.yml` | Optional GitHub Action that runs the script and commits the new snapshot |
| `scripts/fetch_player_stats.py` | Player stats ingestion for the prediction model (see below) |
| `notebooks/Component 2 - Player Stats Ingestion.ipynb` | Runs the stats ingestion step by step for inspection |
| `scripts/build_features.py` | Feature engineering on the ingested stats, for every prop stat |
| `notebooks/Component 2 - Feature Engineering.ipynb` | Runs the feature build and shows the features |
| `scripts/stat_model.py` | Predicts each player's points, rebounds, assists and threes per game (last season: train first half, test second half; no odds) |
| `notebooks/Component 2 - Stat Predictions.ipynb` | Runs the stat predictions and shows scores, drivers and player-by-player results |
| `scripts/player_model.py` | Reference: a separate model per player, compared with the pooled model (it predicted worse) |
| `scripts/line_model.py` | Win % for every historical book line from the stats-only model, next to average Vegas odds, best book odds and the result |
| `notebooks/Component 2 - Line Win Pct.ipynb` | Runs the line pricing and shows accuracy vs Vegas, calibration and returns by ev |
| `scripts/consensus_edge.py` | No stats model: bets when one book's price beats the books' consensus fair price; backtested on last season, tested on this season |
| `notebooks/Component 2 - Consensus Edge.ipynb` | Runs the consensus-edge backtest and shows how the fixed rules did on each season |
| `scripts/fetch_historical_odds.py` | Pulls historical prop odds (one snapshot per game, 30 min before tip) for backtesting; paid Odds API plan |

## Getting data into the app

- **Snapshot (default):** the page loads `data/player_props.json` automatically. To update it, replace the file with the notebook's `nba_player_props.json`, run `scripts/fetch_odds.py` (both leagues by default; `--leagues nba` or `--leagues wnba` for one), or use the GitHub Action.
- **Live:** open *Data source & settings*, paste your Odds API key, and click *Fetch live odds*. The key is stored only in your browser's localStorage, and only if you tick "Remember". It's sent only to `api.the-odds-api.com`. Each game costs about 4 credits, and only the checked leagues are fetched.
- **File:** *Open JSON file* loads any JSON saved by the notebook.

> Don't hard-code your API key into `app.js`. On a public static site anyone can read it.

## Player stats for the prediction model

`scripts/fetch_player_stats.py` is the data-ingestion step for the planned prediction model. It pulls player game logs (current and previous season), upcoming games, and the players in those games from ESPN's public API. It also checks that player names in the odds snapshot match the stats.

```bash
pip install pandas requests
python scripts/fetch_player_stats.py --league nba   # or --league wnba
```

Then build model features for points, rebounds, assists and threes: rolling form, minutes and usage, shot volume, rest, teammates out, and opponent pace and defense (overall, vs league average, and by position):

```bash
python scripts/build_features.py --league nba
```

Then predict each player's stats per game, with no odds involved (`pip install scikit-learn`; about 5 minutes). It uses last season only, trains on the first half and tests on the second half, retraining the models weekly through the second half the way they'd be used for real (`--frozen` trains once instead), and writes scores to `nba_stat_model_results.csv` and test-half predictions to `nba_stat_predictions.csv`:

```bash
python scripts/stat_model.py --league nba
```

Then give every book line in the test half a win % from that model (needs the historical odds below; `pip install scipy`). The model still uses stats only, and odds are just compared against afterwards. It writes one row per side of each line to `nba_line_bets.csv`, with the books' average odds and no-vig chance, the model's chance and fair odds, the best book's odds, and the actual result:

```bash
python scripts/line_model.py --league nba
```

To test line shopping without any stats model, compare each book's price with the fair price implied by all the books (their average no-vig probability). It flags bets where the best price beats that by more than 2% on main lines quoted by 4+ books (a rule found on last season's odds), settles them from the box scores, and reports last season and this season separately, this season being the real test. Writes `nba_consensus_bets.csv`:

```bash
python scripts/consensus_edge.py --league nba
```

For backtesting on many nights, pull historical prop odds for the second half of last season and this season so far (paid Odds API plan; about 40 credits per game). `--dry-run` shows the game count and cost without making requests:

```bash
python scripts/fetch_historical_odds.py --dry-run
ODDS_API_KEY=xxxx python scripts/fetch_historical_odds.py
```

All of these write to `model_data/`, which is gitignored along with its ESPN cache (about 60 MB per league). The first run takes 10-15 minutes, and later runs take seconds because finished games are cached. ESPN's API is unofficial and could change without notice.

## Run locally

```bash
cd NBA_Odds_Web
python -m http.server 8000
# open http://localhost:8000
```

(Opening `index.html` directly works too, but browsers block loading the snapshot over `file://`. Use *Open JSON file* or a live fetch in that case.)

## Deploy for free

Pick one:

**GitHub Pages**
1. Create a new GitHub repo and push the *contents* of this folder to it (so `index.html` is at the repo root).
2. Repo → Settings → Pages → Source: *Deploy from a branch*, Branch: `main`, folder `/ (root)`.
3. Your site will be at `https://<user>.github.io/<repo>/`.
4. Optional auto-refresh: Settings → Secrets and variables → Actions → add `ODDS_API_KEY`. Then run *Refresh odds snapshot* from the Actions tab, or uncomment the `schedule` in the workflow. Watch your quota: one run uses about 50 credits on a 12-game slate.

**Netlify Drop**: drag this folder onto https://app.netlify.com/drop.

**Cloudflare Pages**: Create project → *Direct upload* → upload this folder. If you connect the Git repo instead, leave the build command empty and set the output directory to `/`.
