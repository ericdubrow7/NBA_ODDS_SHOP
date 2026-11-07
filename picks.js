// NBA/WNBA Player Props: recommended bets from the books' consensus.
// Same logic as scripts/consensus_edge.py: the fair chance of each side is the books' average
// no-vig probability, and a side is a pick when the best price pays more than that is worth.
// The consensus uses every book (as in the backtest); the best price uses the selected books.

const state = {
  events: [],
  source: "",
  rows: [],
  filtered: [],
  books: initialBooks(),
  fmt: store.get("fmt", "us"),
  sort: "edge",
};

const MIN_EDGE = 0.02;
const MIN_BOOKS = 4;
const PLACEHOLDER = [1.02, 50]; // decimal prices outside this are placeholders, not real quotes
const MAX_MARGIN = 0.15;        // two-sided quotes with a bigger margin are stale

// Rule checks shown on every row, in column order.
const RULES = [
  { key: "edgeOk", label: "Edge > 2%" },
  { key: "booksOk", label: "4+ books" },
  { key: "main", label: "Main line" },
  { key: "notNovig", label: "Novig not best" },
];
// "Show" filter: which rules a pick has to meet.
const SHOW = {
  rule: (r) => r.rule,
  ruleNoNovig: (r) => r.rule && r.notNovig,
  edgeBooks: (r) => r.edgeOk && r.booksOk,
  positive: (r) => r.edge > 0,
  all: () => true,
};

// ---------- consensus math ----------

function buildPicks(events, books) {
  // Every book's two-sided quote per line, from all books.
  const lines = new Map();
  forEachPrice(events, ALL_BOOKS, ({ ev, book, market: m, outcome: o, link }) => {
    if (o.name !== "Over" && o.name !== "Under") return;
    if (o.price < PLACEHOLDER[0] || o.price > PLACEHOLDER[1]) return;
    const key = [ev.id, m.key, o.description, o.point].join("|");
    let g = lines.get(key);
    if (!g) {
      g = { id: ev.id, league: leagueOf(ev), start: ev.commence_time, home: ev.home_team, away: ev.away_team,
        marketKey: m.key, market: MARKETS[m.key], player: o.description, line: o.point, quotes: new Map() };
      lines.set(key, g);
    }
    const q = g.quotes.get(book) || {};
    q[o.name] = { price: o.price, link };
    g.quotes.set(book, q);
  });

  const props = [];
  for (const g of lines.values()) {
    const two = [...g.quotes].filter(([, q]) => q.Over && q.Under &&
      1 / q.Over.price + 1 / q.Under.price - 1 <= MAX_MARGIN);
    if (!two.length) continue;
    const pOver = two.reduce((s, [, q]) => s + (1 / q.Over.price) / (1 / q.Over.price + 1 / q.Under.price), 0) / two.length;
    const avg = (side) => two.reduce((s, [, q]) => s + q[side].price, 0) / two.length;
    const best = (side) => two.filter(([b]) => books.includes(b))
      .reduce((m, [b, q]) => (!m || q[side].price > m.price ? { book: b, price: q[side].price, link: q[side].link } : m), null);
    props.push({ ...g, books: two.length, pOver, avgOver: avg("Over"), avgUnder: avg("Under"), bestOver: best("Over"), bestUnder: best("Under") });
  }

  // Main line: the one most books quote for that player and market (ties: closest to 50/50).
  const mainKey = new Map();
  for (const p of props) {
    const k = [p.id, p.marketKey, p.player].join("|");
    const cur = mainKey.get(k);
    if (!cur || p.books > cur.books || (p.books === cur.books && Math.abs(p.pOver - 0.5) < Math.abs(cur.pOver - 0.5))) mainKey.set(k, p);
  }

  const rows = [];
  for (const p of props) {
    const main = mainKey.get([p.id, p.marketKey, p.player].join("|")) === p;
    for (const side of ["Over", "Under"]) {
      const best = side === "Over" ? p.bestOver : p.bestUnder;
      if (!best) continue; // none of the selected books quote this side
      const pWin = side === "Over" ? p.pOver : 1 - p.pOver;
      const edge = pWin * best.price - 1;
      const r = {
        id: p.id, league: p.league, start: p.start, home: p.home, away: p.away, market: p.market, player: p.player,
        line: p.line, side, books: p.books, main, pWin, avg: side === "Over" ? p.avgOver : p.avgUnder, best, edge,
        edgeOk: edge > MIN_EDGE, booksOk: p.books >= MIN_BOOKS, notNovig: best.book !== "Novig",
      };
      r.rule = r.edgeOk && r.booksOk && r.main;
      rows.push(r);
    }
  }
  return rows;
}

// ---------- data ----------

function setData(events, source) {
  state.events = events;
  state.source = source;
  rememberData(events, source);
  recompute();
}

function recompute() {
  state.rows = buildPicks(state.events, state.books);
  populateFilterOptions();
  applyFilters();
}

// ---------- filtering / sorting ----------

const SORTS = {
  edge: (a, b) => b.edge - a.edge,
  start: (a, b) => a.start.localeCompare(b.start) || b.edge - a.edge,
  player: (a, b) => a.player.localeCompare(b.player) || b.edge - a.edge,
};

function applyFilters() {
  const q = $("search").value.trim().toLowerCase();
  const game = $("fGame").value;
  const market = $("fMarket").value;
  const show = SHOW[$("fShow").value];
  const minEdge = parseFloat($("fEdge").value);
  const floor = Number.isFinite(minEdge) ? minEdge / 100 : -Infinity;

  state.filtered = state.rows.filter((r) =>
    (!q || r.player.toLowerCase().includes(q) || r.home.toLowerCase().includes(q) || r.away.toLowerCase().includes(q)) &&
    (!game || r.id === game) &&
    (!market || r.market === market) &&
    show(r) && r.edge >= floor
  ).sort(SORTS[state.sort]);
  render();
}

function populateFilterOptions() {
  const games = new Map();
  for (const r of state.rows) if (!games.has(r.id)) games.set(r.id, r);
  const gameOpts = [...games.values()].sort((a, b) => a.start.localeCompare(b.start))
    .map((r) => [r.id, `${leagueTag(r)}${r.away} @ ${r.home} · ${fmtGameTime(r.start)}`]);
  fillSelect($("fGame"), "All games", gameOpts);
  fillSelect($("fMarket"), "All markets", Object.values(MARKETS).map((m) => [m, m]));
}

function fillSelect(sel, allLabel, opts) {
  const prev = sel.value;
  sel.innerHTML = `<option value="">${esc(allLabel)}</option>` +
    opts.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  if (opts.some(([v]) => v === prev)) sel.value = prev;
}

// ---------- rendering ----------

const fmtOdds = (dec) => (state.fmt === "us" ? signed(Math.round(decimalToUs(dec))) : dec.toFixed(2));
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const flag = (ok) => `<td class="flag ${ok ? "yes" : "no"}" title="${ok ? "Meets" : "Doesn't meet"} this rule">${ok ? "✓" : "✗"}</td>`;

function render() {
  const rows = state.filtered;
  const picks = state.rows.filter((r) => r.rule);
  const noNovig = picks.filter((r) => r.notNovig);
  const top = picks.reduce((m, r) => (!m || r.edge > m.edge ? r : m), null);
  $("tiles").innerHTML = [
    tile("Recommended (all rules)", picks.length, state.source || "no data"),
    tile("…and Novig not best", noNovig.length, "the version to watch"),
    tile("Biggest edge", top ? pct(top.edge) : "–", top ? `${top.player} ${top.side} ${top.line} ${top.market}` : ""),
    tile("Sides checked", state.rows.length.toLocaleString(), `best price from ${state.books.length} books`),
  ].join("");

  const head = ["Game", "Player", "Prop", "Side", "Books", "Fair %", "Fair odds", "Avg odds", "Best odds", "Edge",
    ...RULES.map((r) => r.label)];
  $("headRow").innerHTML = head.map((h, i) => `<th class="${i < 3 ? "l" : ""}">${esc(h)}</th>`).join("");

  if (!rows.length) {
    const msg = !state.events.length ? "No data loaded yet."
      : noLeagueGames(state.events) ? noLeagueGames(state.events)
      : "No picks match these filters. Try a lower minimum edge or Show: Any line.";
    $("body").innerHTML = `<tr><td class="empty" colspan="${head.length}">${esc(msg)}</td></tr>`;
  } else {
    $("body").innerHTML = rows.map(row).join("");
  }
  $("rowCount").textContent = rows.length ? `${rows.length.toLocaleString()} pick${rows.length === 1 ? "" : "s"}` : "";
}

function row(r) {
  return `<tr class="${r.rule ? (r.notNovig ? "pick" : "pick-novig") : ""}">
    <td class="l game">${esc(leagueTag(r))}${esc(r.away)} @ ${esc(r.home)}<br>${esc(fmtGameTime(r.start))}</td>
    <td class="l player">${esc(r.player)}</td>
    <td class="l">${r.line.toFixed(1)} ${esc(r.market)}</td>
    <td>${r.side}</td>
    <td>${r.books}</td>
    <td>${pct(r.pWin)}</td>
    <td>${fmtOdds(1 / r.pWin)}</td>
    <td>${fmtOdds(r.avg)}</td>
    <td class="best"><b>${fmtOdds(r.best.price)}</b> ${openButton(r.best.book, r.best.link, true)}</td>
    <td class="${r.edge > 0 ? "edge-pos" : "edge-neg"}">${r.edge > 0 ? "+" : ""}${pct(r.edge)}</td>
    ${RULES.map((rule) => flag(r[rule.key])).join("")}
  </tr>`;
}

function tile(label, value, note) {
  return `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(String(value))}</div>` +
    `<div class="note" title="${esc(note)}">${esc(note) || "&nbsp;"}</div></div>`;
}

function exportCsv() {
  const head = ["League", "Game", "Start", "Player", "Market", "Line", "Side", "Books", "Fair %", "Avg odds", "Best odds",
    "Best book", "Edge %", ...RULES.map((r) => r.label), "Recommended"];
  const cell = (v) => `"${String(v).replace(/"/g, '""')}"`;
  const lines = state.filtered.map((r) => [r.league, `${r.away} @ ${r.home}`, r.start, r.player, r.market, r.line, r.side,
    r.books, (r.pWin * 100).toFixed(1), Math.round(decimalToUs(r.avg)), Math.round(decimalToUs(r.best.price)), r.best.book,
    (r.edge * 100).toFixed(2), ...RULES.map((rule) => (r[rule.key] ? "Yes" : "No")), r.rule ? "Yes" : "No"].map(cell).join(","));
  download(new Blob([[head.map(cell).join(","), ...lines].join("\n")], { type: "text/csv" }), `recommended_bets_${dateStamp()}.csv`);
}

// ---------- wiring ----------

function initPage() {
  $("bookChecks").innerHTML = ALL_BOOKS.map((b) =>
    `<label><input type="checkbox" value="${esc(b)}" ${state.books.includes(b) ? "checked" : ""}> ${esc(b)}</label>`).join("");
  $("bookChecks").addEventListener("change", () => {
    state.books = [...$("bookChecks").querySelectorAll("input:checked")].map((i) => i.value);
    store.set("books", state.books);
    recompute();
  });

  $("fShow").value = store.get("picksShow", "rule");
  $("fEdge").value = store.get("picksMinEdge", 2);
  $("fShow").addEventListener("change", () => { store.set("picksShow", $("fShow").value); applyFilters(); });
  $("fEdge").addEventListener("input", () => { store.set("picksMinEdge", $("fEdge").value); applyFilters(); });
  $("search").addEventListener("input", applyFilters);
  for (const id of ["fGame", "fMarket"]) $(id).addEventListener("change", applyFilters);
  $("sortSel").addEventListener("change", (e) => { state.sort = e.target.value; applyFilters(); });
  $("exportCsv").addEventListener("click", exportCsv);

  document.querySelectorAll(".seg button[data-fmt]").forEach((b) => {
    b.classList.toggle("on", b.dataset.fmt === state.fmt);
    b.addEventListener("click", () => {
      state.fmt = b.dataset.fmt;
      store.set("fmt", state.fmt);
      document.querySelectorAll(".seg button[data-fmt]").forEach((x) => x.classList.toggle("on", x === b));
      render();
    });
  });
}

initTheme();
initLeagues(recompute);
initPage();
render();
initDataSource(setData);
