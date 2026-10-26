// NBA Player Props: arbitrage finder.
// For each player/market/line, pairs the best Over with the best Under across the
// selected books. If 1/over + 1/under < 1, backing both sides locks in a profit.

const state = {
  events: [],
  source: "",
  rows: [],
  filtered: [],
  books: initialBooks(),
  fmt: store.get("fmt", "us"),
  stake: store.get("arbStake", 100),
  sort: "roi",
};

const STALE_MIN = 15;

// ---------- arbitrage math ----------

function findArbs(events, books) {
  const groups = new Map();
  forEachPrice(events, books, ({ ev, book, market: m, outcome: o, updated }) => {
    if (o.name !== "Over" && o.name !== "Under") return;
    const key = [ev.id, m.key, o.description, o.point].join("|");
    let g = groups.get(key);
    if (!g) {
      g = {
        id: ev.id,
        start: ev.commence_time,
        home: ev.home_team,
        away: ev.away_team,
        market: MARKETS[m.key],
        player: o.description,
        line: o.point,
        Over: null,
        Under: null,
        counts: { Over: 0, Under: 0 },
      };
      groups.set(key, g);
    }
    g.counts[o.name]++;
    if (!g[o.name] || o.price > g[o.name].price) g[o.name] = { price: o.price, book, updated };
  });

  const rows = [];
  for (const g of groups.values()) {
    if (!g.Over || !g.Under) continue;
    const implied = 1 / g.Over.price + 1 / g.Under.price;
    rows.push({ ...g, implied, roi: (1 / implied - 1) * 100 });
  }
  return rows;
}

// Splits `total` so both legs pay the same: stake_i = total * (1/price_i) / implied.
function stakes(r, total) {
  const over = (total / r.Over.price) / r.implied;
  const under = (total / r.Under.price) / r.implied;
  const payout = total / r.implied;
  return { over, under, payout, profit: payout - total };
}

// ---------- data ----------

function setData(events, source) {
  state.events = events;
  state.source = source;
  rememberData(events, source);
  recompute();
}

function recompute() {
  state.rows = findArbs(state.events, state.books);
  populateFilterOptions();
  applyFilters();
}

// ---------- filtering / sorting ----------

const SORTS = {
  roi: (a, b) => b.roi - a.roi,
  start: (a, b) => a.start.localeCompare(b.start) || b.roi - a.roi,
  player: (a, b) => a.player.localeCompare(b.player) || b.roi - a.roi,
};

function applyFilters() {
  const q = $("search").value.trim().toLowerCase();
  const game = $("fGame").value;
  const market = $("fMarket").value;
  const minRoi = parseFloat($("fRoi").value);
  const floor = Number.isFinite(minRoi) ? minRoi : 0;

  state.filtered = state.rows.filter((r) =>
    (!q || r.player.toLowerCase().includes(q) || r.home.toLowerCase().includes(q) || r.away.toLowerCase().includes(q)) &&
    (!game || r.id === game) &&
    (!market || r.market === market) &&
    // At 0%, require a real edge so float noise on an exact break-even doesn't show as an arb.
    (floor === 0 ? r.roi > 1e-9 : r.roi >= floor)
  ).sort(SORTS[state.sort]);
  render();
}

function populateFilterOptions() {
  const games = new Map();
  for (const r of state.rows) if (!games.has(r.id)) games.set(r.id, r);
  const gameOpts = [...games.values()].sort((a, b) => a.start.localeCompare(b.start))
    .map((r) => [r.id, `${r.away} @ ${r.home} · ${fmtGameTime(r.start)}`]);
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
const money = (n) => `$${n.toFixed(2)}`;

function ago(iso) {
  const t = Date.parse(iso || "");
  if (!t) return { text: "", stale: false };
  const min = Math.max(0, Math.round((Date.now() - t) / 60000));
  const text = min < 1 ? "just now" : min < 60 ? `${min} min ago` : min < 1440 ? `${Math.round(min / 60)} hr ago` : `${Math.round(min / 1440)} d ago`;
  return { text, stale: min > STALE_MIN };
}

function render() {
  const rows = state.filtered;
  const arbs = state.rows.filter((r) => r.roi > 1e-9);
  const best = arbs.reduce((m, r) => (!m || r.roi > m.roi ? r : m), null);
  $("tiles").innerHTML = [
    tile("Arbitrage spots", arbs.length, state.source || "no data"),
    tile("Best return", best ? `${best.roi.toFixed(2)}%` : "–", best ? `${best.player} ${best.line} ${best.market}` : ""),
    tile(`Profit on ${money(state.stake)}`, best ? money(stakes(best, state.stake).profit) : "–", best ? "best spot" : ""),
    tile("Over/Under pairs checked", state.rows.length.toLocaleString(), `${state.books.length} books`),
  ].join("");

  if (!rows.length) {
    const msg = !state.events.length ? "No data loaded yet."
      : state.rows.some((r) => r.roi > 1e-9) ? "No spots match these filters."
      : "No arbitrage in this data. Fetch live odds, or set Min return below 0% to see the closest near-misses.";
    $("arbs").innerHTML = `<div class="empty">${esc(msg)}</div>`;
  } else {
    $("arbs").innerHTML = rows.map(card).join("");
  }
  $("rowCount").textContent = rows.length ? `${rows.length.toLocaleString()} spot${rows.length === 1 ? "" : "s"}` : "";
}

function tile(label, value, note) {
  return `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(String(value))}</div>` +
    `<div class="note" title="${esc(note)}">${esc(note) || "&nbsp;"}</div></div>`;
}

function card(r) {
  const s = stakes(r, state.stake);
  const leg = (side, q, stake) => {
    const age = ago(q.updated);
    return `<div class="leg">
      <div class="leg-side">${side} ${r.line.toFixed(1)}</div>
      <div class="leg-book"><b>${esc(q.book)}</b> ${fmtOdds(q.price)}</div>
      <div class="leg-stake">Bet <b>${money(stake)}</b></div>
      <div class="leg-age${age.stale ? " stale" : ""}">${esc(age.text)}</div>
    </div>`;
  };
  const positive = r.roi > 1e-9;
  return `<article class="card">
    <div class="card-top">
      <div>
        <div class="card-player">${esc(r.player)}</div>
        <div class="card-prop">${r.line.toFixed(1)} ${esc(r.market)}</div>
        <div class="card-game">${esc(r.away)} @ ${esc(r.home)} · ${esc(fmtGameTime(r.start))}</div>
      </div>
      <div class="card-edge ${positive ? "arb-pos" : "arb-neg"}">${positive ? "+" : ""}${r.roi.toFixed(2)}%</div>
    </div>
    <div class="legs">${leg("Over", r.Over, s.over)}${leg("Under", r.Under, s.under)}</div>
    <div class="card-best">
      <span>Total <b>${money(state.stake)}</b></span>
      <span>Payout <b>${money(s.payout)}</b></span>
      <span class="${positive ? "best-pick" : ""}">Profit <b>${s.profit >= 0 ? "+" : "−"}${money(Math.abs(s.profit))}</b></span>
      <span>${r.counts.Over} / ${r.counts.Under} books quoting</span>
    </div>
  </article>`;
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

  $("stake").value = state.stake;
  $("stake").addEventListener("input", () => {
    const v = parseFloat($("stake").value);
    if (!(v > 0)) return;
    state.stake = v;
    store.set("arbStake", v);
    render();
  });
  for (const id of ["search", "fRoi"]) $(id).addEventListener("input", applyFilters);
  for (const id of ["fGame", "fMarket"]) $(id).addEventListener("change", applyFilters);
  $("sortSel").addEventListener("change", (e) => { state.sort = e.target.value; applyFilters(); });

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
initPage();
render();
initDataSource(setData);
