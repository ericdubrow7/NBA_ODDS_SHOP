# NBA Prop Line Shopper

A static web version of `NBA Odds Analysis - Player Props Avg vs Best Book.ipynb`.

For every NBA player-prop line (Points, Rebounds, Assists, Threes), it:

1. Collects Over/Under prices from FanDuel, DraftKings, BetMGM, Caesars and BetRivers.
2. Averages the decimal odds across books offering the same player, market, side and line.
3. Picks the book with the best price and shows how far it sits above the average.

There's no server and no build step. It's just `index.html`, `styles.css`, `app.js` and a data snapshot.

## Files

| Path | Purpose |
| --- | --- |
| `index.html`, `styles.css`, `app.js` | The app (all analysis runs in the browser) |
| `data/nba_player_props.json` | Snapshot loaded on page open (same format the notebook saves) |
| `scripts/fetch_odds.py` | Refreshes the snapshot from The Odds API (key from `ODDS_API_KEY`) |
| `.github/workflows/refresh-odds.yml` | Optional GitHub Action that runs the script and commits the new snapshot |

## Getting data into the app

- **Snapshot (default):** the page loads `data/nba_player_props.json` automatically. To update it, replace the file with the notebook's `nba_player_props.json`, run `scripts/fetch_odds.py`, or use the GitHub Action.
- **Live:** open *Data source & settings*, paste your Odds API key, and click *Fetch live odds*. The key is stored only in your browser's localStorage, and only if you tick "Remember". It's sent only to `api.the-odds-api.com`. Each game costs about 4 credits.
- **File:** *Open JSON file* loads any JSON saved by the notebook.

> Don't hard-code your API key into `app.js`. On a public static site anyone can read it.

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
