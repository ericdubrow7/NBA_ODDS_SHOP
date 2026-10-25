// NBA Player Props: Average Odds vs Most Favorable Book
// Browser port of "NBA Odds Analysis - Player Props Avg vs Best Book.ipynb".

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
// Books that existed before a saved selection; anything newer starts checked.
const ORIGINAL_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "BetRivers"];
const SNAPSHOT_URL = "data/nba_player_props.json";
const REQUEST_SLEEP_MS = 300;
const PAGE_SIZE = 150;

const store = {
  get(k, fallback = null) {
    try { const v = localStorage.getItem(k); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

const state = {
  events: [],
  source: "",
  rows: [],
  filtered: [],
  books: initialBooks(),
  minBooks: store.get("minBooks", 2),
  lookahead: store.get("lookahead", 48),
  fmt: store.get("fmt", "us"),
  sort: { key: "pct", dir: -1 },
  shown: PAGE_SIZE,
};

function initialBooks() {
  const saved = store.get("books");
  const known = store.get("knownBooks", ORIGINAL_BOOKS);
  store.set("knownBooks", ALL_BOOKS);
  if (!saved) return ALL_BOOKS.slice();
  return ALL_BOOKS.filter((b) => saved.includes(b) || !known.includes(b));
}

const $ = (id) => document.getElementById(id);

// ---------- odds math (matches the notebook) ----------

function decimalToUs(d) {
  if (d == null || Number.isNaN(d) || d === 1) return 0;
  return d >= 2 ? (d - 1) * 100 : -100 / (d - 1);
}

function buildComparison(events, books, minBooks) {
  const lines = new Map();
  for (const ev of events) {
    for (const bm of ev.bookmakers || []) {
      const name = BOOKS[bm.key];
      if (!name || !books.includes(name)) continue;
      for (const m of bm.markets || []) {
        if (!(m.key in MARKETS)) continue;
        for (const o of m.outcomes || []) {
          // Novig posts 1.0 (shown as 0) when it has no price; treat as not offered.
          if (o.point == null || o.price == null || o.price <= 1) continue;
          const key = [ev.id, m.key, o.description, o.name, o.point].join("|");
          let line = lines.get(key);
          if (!line) {
            line = {
              id: ev.id,
              start: ev.commence_time,
              home: ev.home_team,
              away: ev.away_team,
              market: MARKETS[m.key],
              player: o.description,
              side: o.name,
              line: o.point,
              prices: {},
            };
            lines.set(key, line);
          }
          if (!(name in line.prices)) line.prices[name] = o.price;
        }
      }
    }
  }

  const rows = [];
  for (const r of lines.values()) {
    const offered = books.filter((b) => r.prices[b] != null);
    if (offered.length < minBooks) continue;
    const vals = offered.map((b) => r.prices[b]);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const best = Math.max(...vals);
    const bestBook = offered.find((b) => r.prices[b] === best); // first max, like idxmax
    const avgUs = Math.round(decimalToUs(avg));
    const bestUs = Math.round(decimalToUs(best));
    rows.push({
      ...r,
      booksOffering: offered.length,
      avg,
      best,
      bestBook,
      avgUs,
      bestUs,
      priceVsAvg: best - avg,
      pct: (best / avg - 1) * 100,
      usVsAvg: bestUs - avgUs,
    });
  }
  return rows;
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

async function fetchLive() {
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
    setStatus("Looking up upcoming NBA games…");
    const { json: allEvents } = await apiGet(`/sports/${SPORT}/events`, { apiKey: key });
    const hours = parseFloat($("lookahead").value);
    const cutoff = Number.isFinite(hours) ? Date.now() + hours * 3600e3 : Infinity;
    const events = allEvents.filter((e) => new Date(e.commence_time).getTime() <= cutoff);
    if (!events.length) {
      setStatus("No NBA games in that window.", "warn");
      return;
    }

    const cost = events.length * Object.keys(MARKETS).length;
    if (!confirm(`Fetch player props for ${events.length} game(s)? This uses about ${cost} API credits.`)) {
      setStatus("Cancelled.");
      return;
    }

    const data = [];
    let remaining = null;
    for (let i = 0; i < events.length; i++) {
      setStatus(`Fetching game ${i + 1} of ${events.length}…`);
      try {
        const r = await apiGet(`/sports/${SPORT}/events/${events[i].id}/odds`, {
          apiKey: key,
          bookmakers: Object.keys(BOOKS).join(","),
          markets: Object.keys(MARKETS).join(","),
          oddsFormat: "decimal",
          dateFormat: "iso",
        });
        data.push(r.json);
        remaining = r.remaining;
      } catch (err) {
        console.warn("Event failed", events[i].id, err);
      }
      if (i < events.length - 1) await sleep(REQUEST_SLEEP_MS);
    }
    setData(data, `Live · fetched ${fmtClock(new Date())}`);
    const quota = remaining != null ? ` · ${remaining} API credits left` : "";
    setStatus(`Loaded ${data.length} of ${events.length} games${quota}.`);
  } catch (err) {
    setStatus(`Fetch failed: ${err.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

async function loadSnapshot() {
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

function loadFile(file) {
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

function setData(events, source) {
  state.events = events;
  state.source = source;
  recompute();
}

function recompute() {
  state.rows = buildComparison(state.events, state.books, state.minBooks);
  populateFilterOptions();
  applyFilters();
}

// ---------- filtering / sorting ----------

const SORTERS = {
  start: (r) => r.start,
  player: (r) => r.player,
  market: (r) => r.market,
  side: (r) => r.side,
  line: (r) => r.line,
  booksOffering: (r) => r.booksOffering,
  avg: (r) => r.avg,
  bestBook: (r) => r.bestBook,
  best: (r) => r.best,
  usVsAvg: (r) => r.usVsAvg,
  pct: (r) => r.pct,
};

function applyFilters() {
  const q = $("search").value.trim().toLowerCase();
  const game = $("fGame").value;
  const market = $("fMarket").value;
  const side = $("fSide").value;
  const book = $("fBook").value;
  const minEdge = parseFloat($("fEdge").value);

  let rows = state.rows.filter((r) =>
    (!q || r.player.toLowerCase().includes(q) || r.home.toLowerCase().includes(q) || r.away.toLowerCase().includes(q)) &&
    (!game || r.id === game) &&
    (!market || r.market === market) &&
    (!side || r.side === side) &&
    (!book || r.bestBook === book) &&
    (!Number.isFinite(minEdge) || r.pct >= minEdge)
  );

  const { key, dir } = state.sort;
  const get = SORTERS[key] || ((r) => r.prices[key] ?? -Infinity);
  rows.sort((a, b) => {
    const x = get(a), y = get(b);
    if (x === y) return (b.pct - a.pct) || (b.usVsAvg - a.usVsAvg);
    return (x > y ? 1 : -1) * dir;
  });

  state.filtered = rows;
  state.shown = PAGE_SIZE;
  renderSummary();
  renderTable();
}

function populateFilterOptions() {
  const games = new Map();
  for (const r of state.rows) if (!games.has(r.id)) games.set(r.id, r);
  const gameOpts = [...games.values()].sort((a, b) => a.start.localeCompare(b.start))
    .map((r) => [r.id, `${r.away} @ ${r.home} · ${fmtGameTime(r.start)}`]);
  fillSelect($("fGame"), "All games", gameOpts);
  fillSelect($("fMarket"), "All markets", Object.values(MARKETS).map((m) => [m, m]));
  fillSelect($("fBook"), "Any best book", state.books.map((b) => [b, b]));
}

function fillSelect(sel, allLabel, opts) {
  const prev = sel.value;
  sel.innerHTML = `<option value="">${esc(allLabel)}</option>` +
    opts.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  if (opts.some(([v]) => v === prev)) sel.value = prev;
}

// ---------- rendering ----------

function renderSummary() {
  const rows = state.filtered;
  const games = new Set(rows.map((r) => r.id)).size;
  const players = new Set(rows.map((r) => r.player)).size;
  const top = rows.reduce((m, r) => (!m || r.pct > m.pct ? r : m), null);
  const avgEdge = rows.length ? rows.reduce((s, r) => s + r.pct, 0) / rows.length : 0;

  $("tiles").innerHTML = [
    tile("Comparable lines", rows.length.toLocaleString(), `${state.minBooks}+ books · ${state.source || "no data"}`),
    tile("Games", games, ""),
    tile("Players", players, ""),
    tile("Avg best-vs-avg edge", `${avgEdge.toFixed(2)}%`, "decimal price"),
    tile("Top edge", top ? `${top.pct.toFixed(2)}%` : "–",
      top ? `${top.player} ${top.side} ${top.line} ${top.market} @ ${top.bestBook}` : ""),
  ].join("");

  const bookCounts = countBy(rows, (r) => r.bestBook);
  renderBars($("bookBars"), state.books.map((b) => [b, bookCounts[b] || 0]).sort((a, b) => b[1] - a[1]));
  const marketCounts = countBy(rows, (r) => r.market);
  renderBars($("marketBars"), Object.values(MARKETS).map((m) => [m, marketCounts[m] || 0]).sort((a, b) => b[1] - a[1]));
}

function tile(label, value, note) {
  return `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(String(value))}</div>` +
    `<div class="note" title="${esc(note)}">${esc(note) || "&nbsp;"}</div></div>`;
}

function renderBars(el, entries) {
  const max = Math.max(1, ...entries.map((e) => e[1]));
  el.innerHTML = entries.map(([label, n]) =>
    `<div class="bar-row"><span>${esc(label)}</span>` +
    `<div class="bar-track"><div class="bar-fill" style="width:${(n / max) * 100}%"></div></div>` +
    `<span class="n">${n}</span></div>`).join("");
}

function columns() {
  const isUs = state.fmt === "us";
  return [
    { key: "start", label: "Game", cls: "l" },
    { key: "player", label: "Player", cls: "l" },
    { key: "market", label: "Market" },
    { key: "side", label: "Side" },
    { key: "line", label: "Line" },
    ...state.books.map((b) => ({ key: b, label: b, book: true })),
    { key: "booksOffering", label: "Books" },
    { key: "avg", label: isUs ? "Avg Odds" : "Avg Decimal" },
    { key: "bestBook", label: "Best Book", cls: "l" },
    { key: "best", label: isUs ? "Best Odds" : "Best Decimal" },
    { key: "usVsAvg", label: isUs ? "Odds vs Avg" : "Decimal vs Avg" },
    { key: "pct", label: "% vs Avg" },
  ];
}

function renderHead() {
  $("headRow").innerHTML = columns().map((c) => {
    const sort = state.sort.key === c.key ? (state.sort.dir > 0 ? "ascending" : "descending") : "none";
    return `<th class="${c.cls || ""}" data-key="${esc(c.key)}" aria-sort="${sort}">${esc(c.label)}</th>`;
  }).join("");
}

function renderTable() {
  renderHead();
  const cols = columns();
  const rows = state.filtered.slice(0, state.shown);
  const maxPct = Math.max(1, ...state.filtered.map((r) => r.pct));

  if (!rows.length) {
    $("body").innerHTML = `<tr><td class="empty" colspan="${cols.length}">${
      state.events.length ? "No lines match these filters." : "No data loaded yet."}</td></tr>`;
  } else {
    $("body").innerHTML = rows.map((r) => {
      const cells = cols.map((c) => {
        switch (c.key) {
          case "start":
            return `<td class="l game">${esc(r.away)} @ ${esc(r.home)}<br>${esc(fmtGameTime(r.start))}</td>`;
          case "player": return `<td class="l player">${esc(r.player)}</td>`;
          case "market": return `<td>${esc(r.market)}</td>`;
          case "side": return `<td>${esc(r.side)}</td>`;
          case "line": return `<td>${r.line.toFixed(1)}</td>`;
          case "booksOffering": return `<td>${r.booksOffering}</td>`;
          case "avg": return `<td>${fmtOdds(r.avg, r.avgUs)}</td>`;
          case "bestBook": return `<td class="l best">${esc(r.bestBook)}</td>`;
          case "best": return `<td class="best">${fmtOdds(r.best, r.bestUs)}</td>`;
          case "usVsAvg":
            return `<td>${state.fmt === "us" ? signed(r.usVsAvg) : "+" + r.priceVsAvg.toFixed(3)}</td>`;
          case "pct": return `<td style="${edgeStyle(r.pct, maxPct)}">${r.pct.toFixed(2)}%</td>`;
          default: {
            const p = r.prices[c.key];
            if (p == null) return `<td class="na">–</td>`;
            const cls = p === r.best ? "best" : "";
            return `<td class="${cls}">${fmtOdds(p, Math.round(decimalToUs(p)))}</td>`;
          }
        }
      });
      return `<tr>${cells.join("")}</tr>`;
    }).join("");
  }

  const total = state.filtered.length;
  $("rowCount").textContent = total ? `Showing ${rows.length.toLocaleString()} of ${total.toLocaleString()} lines` : "";
  $("more").hidden = rows.length >= total;
}

function edgeStyle(pct, maxPct) {
  if (pct <= 0) return "";
  const a = Math.min(1, pct / maxPct) * 0.55 + 0.05;
  return `background: rgba(46, 160, 90, ${a.toFixed(3)});`;
}

function exportCsv() {
  const header = ["Start (ET)", "Home", "Away", "Market", "Player", "Side", "Line",
    ...state.books.map((b) => `${b} (US)`), ...state.books.map((b) => `${b} (Dec)`),
    "Books", "Avg Decimal", "Avg US", "Best Book", "Best Decimal", "Best US", "Decimal vs Avg", "US vs Avg", "% vs Avg"];
  const lines = [header];
  for (const r of state.filtered) {
    lines.push([
      fmtGameTime(r.start), r.home, r.away, r.market, r.player, r.side, r.line,
      ...state.books.map((b) => (r.prices[b] != null ? Math.round(decimalToUs(r.prices[b])) : "")),
      ...state.books.map((b) => r.prices[b] ?? ""),
      r.booksOffering, r.avg.toFixed(3), r.avgUs, r.bestBook, r.best.toFixed(3), r.bestUs,
      r.priceVsAvg.toFixed(3), r.usVsAvg, r.pct.toFixed(2),
    ]);
  }
  const csv = lines.map((row) => row.map(csvCell).join(",")).join("\r\n");
  download(new Blob([csv], { type: "text/csv" }), `nba_player_props_best_book_${dateStamp()}.csv`);
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
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
const fmtOdds = (dec, us) => (state.fmt === "us" ? signed(us) : dec.toFixed(2));

function countBy(rows, fn) {
  const out = {};
  for (const r of rows) { const k = fn(r); out[k] = (out[k] || 0) + 1; }
  return out;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setStatus(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

// ---------- wiring ----------

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

function initSettings() {
  $("apiKey").value = store.get("apiKey", "");
  $("lookahead").value = state.lookahead ?? "";
  $("minBooks").value = state.minBooks;

  $("bookChecks").innerHTML = ALL_BOOKS.map((b) =>
    `<label><input type="checkbox" value="${esc(b)}" ${state.books.includes(b) ? "checked" : ""}> ${esc(b)}</label>`).join("");
  $("bookChecks").addEventListener("change", () => {
    state.books = [...$("bookChecks").querySelectorAll("input:checked")].map((i) => i.value);
    store.set("books", state.books);
    recompute();
  });
  $("minBooks").addEventListener("change", () => {
    state.minBooks = Math.max(1, parseInt($("minBooks").value, 10) || 1);
    store.set("minBooks", state.minBooks);
    recompute();
  });
  $("lookahead").addEventListener("change", () => {
    const v = parseFloat($("lookahead").value);
    state.lookahead = Number.isFinite(v) ? v : null;
    store.set("lookahead", state.lookahead);
  });
  $("rememberKey").addEventListener("change", (e) => { if (!e.target.checked) store.del("apiKey"); });

  $("fetchLive").addEventListener("click", fetchLive);
  $("loadSnapshot").addEventListener("click", loadSnapshot);
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ""; });
  $("downloadJson").addEventListener("click", () => {
    if (!state.events.length) return setStatus("Nothing to save yet.", "warn");
    download(new Blob([JSON.stringify(state.events)], { type: "application/json" }), `nba_player_props_${dateStamp()}.json`);
  });
}

function initTable() {
  for (const id of ["search", "fGame", "fMarket", "fSide", "fBook", "fEdge"]) {
    $(id).addEventListener(id === "search" || id === "fEdge" ? "input" : "change", applyFilters);
  }
  document.querySelectorAll(".seg button").forEach((b) => {
    b.classList.toggle("on", b.dataset.fmt === state.fmt);
    b.addEventListener("click", () => {
      state.fmt = b.dataset.fmt;
      store.set("fmt", state.fmt);
      document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
      renderTable();
    });
  });
  $("headRow").addEventListener("click", (e) => {
    const th = e.target.closest("th");
    if (!th) return;
    const key = th.dataset.key;
    const textCol = ["start", "player", "market", "side", "bestBook"].includes(key);
    state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: textCol ? 1 : -1 };
    applyFilters();
  });
  $("more").addEventListener("click", () => { state.shown += PAGE_SIZE; renderTable(); });
  $("exportCsv").addEventListener("click", exportCsv);
}

initTheme();
initSettings();
initTable();
renderSummary();
renderTable();
loadSnapshot();
