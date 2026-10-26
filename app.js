// NBA Player Props: Average Odds vs Most Favorable Book
// Browser port of "NBA Odds Analysis - Player Props Avg vs Best Book.ipynb".

const PAGE_SIZE = 150;

const state = {
  events: [],
  source: "",
  rows: [],
  filtered: [],
  books: initialBooks(),
  minBooks: store.get("minBooks", 2),
  fmt: store.get("fmt", "us"),
  view: store.get("view", null),
  sort: { key: "pct", dir: -1 },
  shown: PAGE_SIZE,
};

// ---------- odds math (matches the notebook) ----------

function buildComparison(events, books, minBooks) {
  const lines = new Map();
  forEachPrice(events, books, ({ ev, book, market: m, outcome: o }) => {
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
    if (!(book in line.prices)) line.prices[book] = o.price;
  });

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

function setData(events, source) {
  state.events = events;
  state.source = source;
  rememberData(events, source);
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
  const nBooks = parseInt($("fBooks").value, 10);
  const minEdge = parseFloat($("fEdge").value);

  let rows = state.rows.filter((r) =>
    (!q || r.player.toLowerCase().includes(q) || r.home.toLowerCase().includes(q) || r.away.toLowerCase().includes(q)) &&
    (!game || r.id === game) &&
    (!market || r.market === market) &&
    (!side || r.side === side) &&
    (!book || r.bestBook === book) &&
    (!nBooks || r.booksOffering >= nBooks) &&
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
  // "N+ books" from just above the settings minimum up to the most any line has.
  const maxBooks = state.rows.reduce((m, r) => Math.max(m, r.booksOffering), 0);
  const countOpts = [];
  for (let n = state.minBooks + 1; n <= maxBooks; n++) countOpts.push([String(n), n === maxBooks ? `${n} books` : `${n}+ books`]);
  fillSelect($("fBooks"), "Any # of books", countOpts);
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

// Matches the max-width breakpoint in styles.css.
const mobileMq = matchMedia("(max-width: 700px)");
// Saved "table"/"cards" choice; until one is made, phones get cards.
const cardView = () => (state.view ? state.view === "cards" : mobileMq.matches);

function renderTable() {
  const cols = columns();
  const rows = state.filtered.slice(0, state.shown);
  const maxPct = Math.max(1, ...state.filtered.map((r) => r.pct));
  const cards = cardView();
  $("grid").parentElement.hidden = cards;
  $("cards").hidden = !cards;
  $("sortSel").hidden = !cards;
  document.querySelectorAll(".seg button[data-view]").forEach((b) =>
    b.classList.toggle("on", b.dataset.view === (cards ? "cards" : "table")));
  const sortVal = `${state.sort.key}:${state.sort.dir}`;
  if ([...$("sortSel").options].some((o) => o.value === sortVal)) $("sortSel").value = sortVal;

  if (cards) {
    renderCards(rows, maxPct);
  } else if (!rows.length) {
    renderHead();
    $("body").innerHTML = `<tr><td class="empty" colspan="${cols.length}">${
      state.events.length ? "No lines match these filters." : "No data loaded yet."}</td></tr>`;
  } else {
    renderHead();
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

function renderCards(rows, maxPct) {
  if (!rows.length) {
    $("cards").innerHTML = `<div class="empty">${state.events.length ? "No lines match these filters." : "No data loaded yet."}</div>`;
    return;
  }
  const isUs = state.fmt === "us";
  $("cards").innerHTML = rows.map((r) => {
    const chips = state.books.filter((b) => r.prices[b] != null).map((b) => {
      const p = r.prices[b];
      return `<span class="chip${p === r.best ? " best" : ""}">${esc(b)} <b>${fmtOdds(p, Math.round(decimalToUs(p)))}</b></span>`;
    }).join("");
    return `<article class="card">
      <div class="card-top">
        <div>
          <div class="card-player">${esc(r.player)}</div>
          <div class="card-prop">${esc(r.side)} ${r.line.toFixed(1)} ${esc(r.market)}</div>
          <div class="card-game">${esc(r.away)} @ ${esc(r.home)} · ${esc(fmtGameTime(r.start))}</div>
        </div>
        <div class="card-edge" style="${edgeStyle(r.pct, maxPct)}">${r.pct.toFixed(2)}%</div>
      </div>
      <div class="card-best">
        <span class="best-pick">Best <b>${esc(r.bestBook)} ${fmtOdds(r.best, r.bestUs)}</b></span>
        <span>Avg <b>${fmtOdds(r.avg, r.avgUs)}</b></span>
        <span>vs avg <b>${isUs ? signed(r.usVsAvg) : "+" + r.priceVsAvg.toFixed(3)}</b></span>
        <span><b>${r.booksOffering}</b> books</span>
      </div>
      <div class="card-books">${chips}</div>
    </article>`;
  }).join("");
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

// ---------- formatting helpers ----------

const fmtOdds = (dec, us) => (state.fmt === "us" ? signed(us) : dec.toFixed(2));

function countBy(rows, fn) {
  const out = {};
  for (const r of rows) { const k = fn(r); out[k] = (out[k] || 0) + 1; }
  return out;
}

// ---------- wiring ----------

function initSettings() {
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
  $("downloadJson").addEventListener("click", () => {
    if (!state.events.length) return setStatus("Nothing to save yet.", "warn");
    download(new Blob([JSON.stringify(state.events)], { type: "application/json" }), `nba_player_props_${dateStamp()}.json`);
  });
}

function initTable() {
  for (const id of ["search", "fGame", "fMarket", "fSide", "fBook", "fBooks", "fEdge"]) {
    $(id).addEventListener(id === "search" || id === "fEdge" ? "input" : "change", applyFilters);
  }
  document.querySelectorAll(".seg button[data-fmt]").forEach((b) => {
    b.classList.toggle("on", b.dataset.fmt === state.fmt);
    b.addEventListener("click", () => {
      state.fmt = b.dataset.fmt;
      store.set("fmt", state.fmt);
      document.querySelectorAll(".seg button[data-fmt]").forEach((x) => x.classList.toggle("on", x === b));
      renderTable();
    });
  });
  document.querySelectorAll(".seg button[data-view]").forEach((b) => {
    b.addEventListener("click", () => {
      state.view = b.dataset.view;
      store.set("view", state.view);
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
  $("sortSel").addEventListener("change", (e) => {
    const [key, dir] = e.target.value.split(":");
    state.sort = { key, dir: Number(dir) };
    applyFilters();
  });
  mobileMq.addEventListener("change", renderTable);
  $("more").addEventListener("click", () => { state.shown += PAGE_SIZE; renderTable(); });
  $("exportCsv").addEventListener("click", exportCsv);
}

initTheme();
initSettings();
initTable();
renderSummary();
renderTable();
initDataSource(setData);
