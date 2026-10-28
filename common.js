// Shared by every page: books, storage, data loading, formatting, theme.
// Load before the page script (app.js, arbitrage.js).

const API_BASE = "https://api.the-odds-api.com/v4";
const SPORT = "basketball_nba";
const MARKETS = {
  player_points: "Points",
  player_rebounds: "Rebounds",
  player_assists: "Assists",
  player_threes: "Threes",
};
// Odds API bookmaker key -> display name. Matched by key because titles change
// (espnbet now reports as "theScore Bet"). Requested via `bookmakers`, which bills
// one region per 10 books, so these span us/us2/us_ex at the cost of one region.
const BOOKS = {
  fanduel: "FanDuel",
  draftkings: "DraftKings",
  betmgm: "BetMGM",
  williamhill_us: "Caesars",
  betrivers: "BetRivers",
  ballybet: "Bally Bet",
  espnbet: "theScore Bet",
  hardrockbet: "Hard Rock Bet",
  novig: "Novig",
};
const ALL_BOOKS = Object.values(BOOKS);
// Fallback when the API returns no deep link (it doesn't return homepages).
const BOOK_HOMEPAGES = {
  FanDuel: "https://sportsbook.fanduel.com/",
  DraftKings: "https://sportsbook.draftkings.com/",
  BetMGM: "https://sports.betmgm.com/",
  Caesars: "https://sportsbook.caesars.com/",
  BetRivers: "https://www.betrivers.com/",
  "Bally Bet": "https://www.ballybet.com/",
  "theScore Bet": "https://thescore.bet/",
  "Hard Rock Bet": "https://www.hardrock.bet/",
  Novig: "https://novig.us/",
};
// Books that existed before a saved selection; anything newer starts checked.
const ORIGINAL_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "BetRivers"];
const SNAPSHOT_URL = "data/nba_player_props.json";
const REQUEST_SLEEP_MS = 300;

const store = {
  get(k, fallback = null) {
    try { const v = localStorage.getItem(k); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

function initialBooks() {
  const saved = store.get("books");
  const known = store.get("knownBooks", ORIGINAL_BOOKS);
  store.set("knownBooks", ALL_BOOKS);
  if (!saved) return ALL_BOOKS.slice();
  return ALL_BOOKS.filter((b) => saved.includes(b) || !known.includes(b));
}

const $ = (id) => document.getElementById(id);

function decimalToUs(d) {
  if (d == null || Number.isNaN(d) || d === 1) return 0;
  return d >= 2 ? (d - 1) * 100 : -100 / (d - 1);
}

// Walks every usable price: selected books only, skipping Novig's 1.0 "no price".
function forEachPrice(events, books, fn) {
  for (const ev of events) {
    for (const bm of ev.bookmakers || []) {
      const book = BOOKS[bm.key];
      if (!book || !books.includes(book)) continue;
      for (const m of bm.markets || []) {
        if (!(m.key in MARKETS)) continue;
        for (const o of m.outcomes || []) {
          if (o.point == null || o.price == null || o.price <= 1) continue;
          fn({ ev, book, market: m, outcome: o, updated: m.last_update || bm.last_update, link: deepLink(bm, m, o) });
        }
      }
    }
  }
}

// Deepest link the API gave (fetched with includeLinks): the bet slip, then the
// market, then the game. Only https URLs are used since they become hrefs.
function deepLink(bm, m, o) {
  const pick = [[o.link, "slip"], [m.link, "market"], [bm.link, "game"]]
    .find(([url]) => typeof url === "string" && url.startsWith("https://"));
  return pick ? { url: pick[0], kind: pick[1] } : null;
}

// ---------- data loading ----------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function apiGet(path, params) {
  const url = new URL(API_BASE + path);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return { json: await res.json(), remaining: res.headers.get("x-requests-remaining"), used: res.headers.get("x-requests-used") };
}

// Returns { data, total, remaining }, or null if there was nothing to fetch or the user cancelled.
async function fetchLiveEvents(key, hours, report) {
  report("Looking up upcoming NBA games…");
  const { json: allEvents } = await apiGet(`/sports/${SPORT}/events`, { apiKey: key });
  const cutoff = Number.isFinite(hours) ? Date.now() + hours * 3600e3 : Infinity;
  const events = allEvents.filter((e) => new Date(e.commence_time).getTime() <= cutoff);
  if (!events.length) {
    report("No NBA games in that window.", "warn");
    return null;
  }

  const cost = events.length * Object.keys(MARKETS).length;
  if (!confirm(`Fetch player props for ${events.length} game(s)? This uses about ${cost} API credits.`)) {
    report("Cancelled.");
    return null;
  }

  const data = [];
  let remaining = null;
  for (let i = 0; i < events.length; i++) {
    report(`Fetching game ${i + 1} of ${events.length}…`);
    try {
      const r = await apiGet(`/sports/${SPORT}/events/${events[i].id}/odds`, {
        apiKey: key,
        bookmakers: Object.keys(BOOKS).join(","),
        markets: Object.keys(MARKETS).join(","),
        oddsFormat: "decimal",
        dateFormat: "iso",
        includeLinks: "true",
      });
      data.push(r.json);
      remaining = r.remaining;
    } catch (err) {
      console.warn("Event failed", events[i].id, err);
    }
    if (i < events.length - 1) await sleep(REQUEST_SLEEP_MS);
  }
  return { data, total: events.length, remaining };
}

async function loadSnapshot(setData) {
  try {
    setStatus("Loading saved snapshot…");
    const res = await fetch(SNAPSHOT_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const stamp = latestUpdate(data);
    setData(data, `Snapshot${stamp ? " · odds as of " + fmtClock(stamp) : ""}`);
    setStatus(`Loaded snapshot: ${data.length} games${stamp ? ", odds as of " + fmtClock(stamp) : ""}.`);
  } catch (err) {
    setStatus(`No snapshot available (${err.message}). Enter an API key and fetch live odds.`, "warn");
    $("settings").open = true;
  }
}

function loadFile(file, setData) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      if (!Array.isArray(data)) throw new Error("expected an array of events");
      setData(data, `File · ${file.name}`);
      setStatus(`Loaded ${data.length} games from ${file.name}.`);
    } catch (err) {
      setStatus(`Could not read ${file.name}: ${err.message}`, "err");
    }
  };
  reader.readAsText(file);
}

function latestUpdate(data) {
  let max = 0;
  for (const ev of data) for (const bm of ev.bookmakers || []) for (const m of bm.markets || []) {
    const t = Date.parse(m.last_update || bm.last_update || "");
    if (t > max) max = t;
  }
  return max ? new Date(max) : null;
}

// Keeps the loaded odds for this tab so switching pages doesn't spend credits on a refetch.
function rememberData(events, source) {
  try { sessionStorage.setItem("events", JSON.stringify({ events, source })); } catch {}
}

function recallData() {
  try { return JSON.parse(sessionStorage.getItem("events")); } catch { return null; }
}

// Fetches live odds with the key and look-ahead fields in the settings panel.
async function fetchLive(setData) {
  const key = $("apiKey").value.trim();
  if (!key) {
    setStatus("Enter your Odds API key first.", "warn");
    $("settings").open = true;
    $("apiKey").focus();
    return;
  }
  if ($("rememberKey").checked) store.set("apiKey", key); else store.del("apiKey");

  const btn = $("fetchLive");
  btn.disabled = true;
  try {
    const res = await fetchLiveEvents(key, parseFloat($("lookahead").value), setStatus);
    if (!res) return;
    setData(res.data, `Live · fetched ${fmtClock(new Date())}`);
    const quota = res.remaining != null ? ` · ${res.remaining} API credits left` : "";
    setStatus(`Loaded ${res.data.length} of ${res.total} games${quota}.`);
  } catch (err) {
    setStatus(`Fetch failed: ${err.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

// Wires the shared data-source controls (key, look-ahead, fetch/snapshot/file buttons).
function initDataSource(setData) {
  $("apiKey").value = store.get("apiKey", "");
  $("lookahead").value = store.get("lookahead", 48) ?? "";
  $("lookahead").addEventListener("change", () => {
    const v = parseFloat($("lookahead").value);
    store.set("lookahead", Number.isFinite(v) ? v : null);
  });
  $("rememberKey").addEventListener("change", (e) => { if (!e.target.checked) store.del("apiKey"); });
  $("fetchLive").addEventListener("click", () => fetchLive(setData));
  $("loadSnapshot").addEventListener("click", () => loadSnapshot(setData));
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0], setData); e.target.value = ""; });

  const saved = recallData();
  if (saved && Array.isArray(saved.events)) {
    setData(saved.events, saved.source);
    setStatus(`Loaded ${saved.events.length} games · ${saved.source}.`);
  } else {
    loadSnapshot(setData);
  }
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- formatting helpers ----------

const gameTimeFmt = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
});
const fmtGameTime = (iso) => `${gameTimeFmt.format(new Date(iso))} ET`;
const fmtClock = (d) => d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const dateStamp = () => new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
const signed = (n) => (n > 0 ? `+${n}` : String(n));

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setStatus(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

function initTheme() {
  const saved = store.get("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
  });
}
