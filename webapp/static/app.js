/* FootyIntel — frontend logic
 * Talks to the FastAPI backend (app/main.py) and renders Chart.js visualizations.
 */

const state = {
  league: "EPL",
  leagueMeta: { label: "Premier League", code: "E0", country: "England" },
  teams: [],
  charts: {},
  pendingPredict: null, // {home, away} — applied by loadTeams when the dropdowns fill
};

const $ = (sel) => document.querySelector(sel);

const LEAGUE_LABELS = {
  EPL: "Premier League",
  LALIGA: "La Liga",
  SERIEA: "Serie A",
};

/* ---------- Helpers ---------- */

async function api(path) {
  const res = await fetch("/api" + path);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

async function apiRaw(path, options = {}) {
  const res = await fetch("/api" + path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function fmtPct(x) {
  return x == null ? "—" : (x * 100).toFixed(1) + "%";
}

function fmtNum(x, d = 2) {
  return x == null ? "—" : Number(x).toFixed(d);
}

function destroyChart(id) {
  if (state.charts[id]) {
    state.charts[id].destroy();
    delete state.charts[id];
  }
}

function makeChart(id, config) {
  destroyChart(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (typeof Chart === "undefined") {
    // Chart.js failed to load (offline/CDN blocked) — degrade gracefully.
    console.warn("Chart.js not available; skipping chart", id);
    ctx.parentElement &&
      (ctx.parentElement.innerHTML =
        '<p class="muted small">Chart unavailable (Chart.js not loaded).</p>');
    return;
  }
  try {
    state.charts[id] = new Chart(ctx, config);
  } catch (err) {
    console.error("Failed to render chart", id, err);
    ctx.parentElement &&
      (ctx.parentElement.innerHTML = `<p class="muted small">Chart failed to render: ${err.message}</p>`);
  }
}

const PALETTE = {
  green: "rgba(52,211,153,1)",
  amber: "rgba(251,191,36,1)",
  red: "rgba(248,113,113,1)",
  blue: "rgba(56,189,248,1)",
  purple: "rgba(167,139,250,1)",
  gray: "rgba(148,163,184,1)",
};

/* ---------- Navigation ---------- */

const VIEWS = {
  dashboard: { title: "Dashboard", sub: "League overview & model insights" },
  fixtures: { title: "Fixtures", sub: "Who's playing next — run the model on any match" },
  teams: { title: "Team Intelligence", sub: "Ratings, strength & form per team" },
  predictor: { title: "Match Predictor", sub: "H/D/A, expected goals & score matrix" },
  backtest: { title: "Backtest Lab", sub: "Walk-forward validation vs baselines" },
  research: { title: "Research", sub: "Edge test verdict, power & control analysis" },
  compare: { title: "League Compare", sub: "Model quality across leagues" },
};

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    switchView(btn.dataset.view);
  });
});

function switchView(view) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById("view-" + view).classList.add("active");
  $("#pageTitle").textContent = VIEWS[view].title;
  $("#pageSubtitle").textContent = VIEWS[view].sub;

  // Per-view loading message so a slow first load never looks like a hang.
  const msgs = {
    teams: "Fitting model & computing ratings… (first load ~15s)",
    predictor: "Fitting model… (first load ~15s)",
    backtest: "Computing walk-forward backtest… (first load ~1-2 min)",
    research: "Computing control, power & holdout verdict…",
    compare: "Running backtests across all leagues… (first load ~2-5 min)",
  };
  const t = $("#loadingText");
  if (t) t.textContent = msgs[view] || "Loading intelligence…";

  if (view === "dashboard") loadDashboard();
  if (view === "fixtures") loadFixtures();
  if (view === "teams") loadTeams();
  if (view === "predictor") loadTeams();
  if (view === "backtest") loadBacktest();
  if (view === "research") loadResearch();
  if (view === "compare") loadCompare();
}

$("#leagueSelect").addEventListener("change", (e) => {
  state.league = e.target.value;
  state.leagueMeta.label = LEAGUE_LABELS[state.league];
  const activeView = document.querySelector(".nav-item.active");
  switchView(activeView.dataset.view);
});

$("#refreshBtn").addEventListener("click", () => {
  const activeView = document.querySelector(".nav-item.active");
  switchView(activeView.dataset.view);
});

/* ---------- Loading overlay & errors ---------- */

let loadingCount = 0;
function showLoading(msg) {
  loadingCount++;
  const t = $("#loadingText");
  if (t && msg) t.textContent = msg;
  $("#loadingOverlay").classList.add("visible");
}
function hideLoading() {
  loadingCount = Math.max(0, loadingCount - 1);
  if (loadingCount === 0) $("#loadingOverlay").classList.remove("visible");
}

let errorDismissed = false;
function showError(msg) {
  const el = $("#errorBanner");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("success");
  el.classList.add("visible");
  errorDismissed = false;
}

function showNotice(msg) {
  // Green success-flavored banner, auto-dismissed after a few seconds.
  const el = $("#errorBanner");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("success");
  el.classList.add("visible");
  errorDismissed = true;
  clearTimeout(showNotice._t);
  showNotice._t = setTimeout(() => {
    el.classList.remove("visible");
  }, 6000);
}

async function withLoading(fn, msg) {
  showLoading(msg);
  try {
    return await fn();
  } catch (err) {
    // Never leave the page blank: surface the failure instead.
    if (!errorDismissed) {
      showError(`⚠️ ${err && err.message ? err.message : err}`);
    }
    console.error(err);
  } finally {
    hideLoading();
  }
}

/* ---------- Dashboard ---------- */

async function loadDashboard() {
  await withLoading(async () => {
    const d = await api(`/dashboard?league=${state.league}`);
    $("#leagueMeta").textContent = `${d.league_meta.label} · ${d.n_matches.toLocaleString()} matches`;

    $("#dashStats").innerHTML = [
      { label: "Matches", value: d.n_matches.toLocaleString(), icon: "📅" },
      { label: "Goals / Match", value: fmtNum(d.averages.total_goals_per_match), icon: "⚽" },
      { label: "Home Win Rate", value: fmtPct(d.outcome_distribution.home_win), icon: "🏠" },
      { label: "Draw Rate", value: fmtPct(d.outcome_distribution.draw), icon: "🤝" },
    ]
      .map(
        (s) =>
          `<div class="stat-card"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
      )
      .join("");

    $("#standingsSeason").textContent =
      `${d.latest_date.slice(0, 4)}/${(Number(d.latest_date.slice(0, 4)) + 1) % 100}`;
    $("#standingsTable").innerHTML =
      `<thead><tr><th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GD</th><th>Pts</th></tr></thead><tbody>` +
      d.standings
        .slice(0, 10)
        .map(
          (r) =>
            `<tr><td>${r.rank}</td><td class="team-cell">${r.team}</td><td>${r.played}</td><td>${r.won}</td><td>${r.draw}</td><td>${r.lost}</td><td>${r.goal_diff > 0 ? "+" : ""}${r.goal_diff}</td><td class="pts">${r.points}</td></tr>`
        )
        .join("") +
      `</tbody>`;

    makeChart("outcomeChart", {
      type: "doughnut",
      data: {
        labels: ["Home Win", "Draw", "Away Win"],
        datasets: [
          {
            data: [
              d.outcome_distribution.home_win,
              d.outcome_distribution.draw,
              d.outcome_distribution.away_win,
            ],
            backgroundColor: [PALETTE.green, PALETTE.amber, PALETTE.red],
            borderWidth: 0,
          },
        ],
      },
      options: {
        plugins: {
          legend: { position: "bottom", labels: { color: "#cbd5e1" } },
          tooltip: { callbacks: { label: (c) => fmtPct(c.parsed) } },
        },
        cutout: "62%",
      },
    });

    makeChart("goalsChart", {
      type: "line",
      data: {
        labels: d.goals_per_game_over_time.months,
        datasets: [
          {
            label: "Goals per game",
            data: d.goals_per_game_over_time.goals_per_game,
            borderColor: PALETTE.blue,
            backgroundColor: "rgba(56,189,248,0.12)",
            fill: true,
            tension: 0.35,
            pointRadius: 2,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#94a3b8", maxRotation: 45, maxTicksLimit: 10 }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
      },
    });

    $("#formGrid").innerHTML = d.form
      ? Object.entries(d.form)
          .map(
            ([team, form]) =>
              `<div class="form-item"><span class="form-team">${team}</span><span class="form-dots">${form
                .split("")
                .map((c) => `<i class="fd ${c}">${c}</i>`)
                .join("")}</span></div>`
          )
          .join("")
      : "<p>No form data</p>";

    const recentRows = d.recent
      .map((r) => {
        const homeScore = r.home_goals;
        const awayScore = r.away_goals;
        const cls =
          r.result === "H" ? "res-home" : r.result === "A" ? "res-away" : "res-draw";
        return `<tr class="${cls}"><td>${r.date}</td><td>${r.home_team}</td><td class="score">${homeScore} - ${awayScore}</td><td>${r.away_team}</td></tr>`;
      })
      .join("");
    $("#recentTable").innerHTML =
      `<thead><tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th></tr></thead><tbody>${recentRows}</tbody>`;
  });
}

/* ---------- Team Intelligence ---------- */

async function loadTeams() {
  await withLoading(async () => {
    const d = await api(`/teams?league=${state.league}`);
    state.teams = d.teams;
    const teams = [...d.teams].sort((a, b) => (b.elo_rating || 0) - (a.elo_rating || 0));

    makeChart("eloChart", {
      type: "bar",
      data: {
        labels: teams.map((t) => t.team),
        datasets: [
          {
            label: "Elo rating",
            data: teams.map((t) => t.elo_rating || 0),
            backgroundColor: teams.map((_, i) =>
              i < 4 ? PALETTE.green : i < teams.length / 2 ? PALETTE.blue : PALETTE.gray
            ),
            borderRadius: 6,
          },
        ],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { ticks: { color: "#cbd5e1", font: { size: 11 } }, grid: { display: false } },
        },
      },
    });

    $("#strengthTable").innerHTML =
      `<thead><tr><th>Team</th><th>Elo</th><th>DC Attack</th><th>DC Defense</th><th>Dyn Attack</th><th>Dyn Defense</th><th>Form</th></tr></thead><tbody>` +
      teams
        .map(
          (t) =>
            `<tr><td class="team-cell">${t.team}</td><td>${fmtNum(t.elo_rating, 0)}</td><td>${fmtNum(t.attack, 3)}</td><td>${fmtNum(t.defense, 3)}</td><td>${fmtNum(t.dyn_attack, 3)}</td><td>${fmtNum(t.dyn_defense, 3)}</td><td><span class="form-dots small">${(t.form || "")
              .split("")
              .map((c) => `<i class="fd ${c}">${c}</i>`)
              .join("")}</span></td></tr>`
        )
        .join("") +
      `</tbody>`;

    $("#homeSelect").innerHTML = "";
    $("#awaySelect").innerHTML = "";
    teams.forEach((t) => {
      const optH = document.createElement("option");
      optH.value = t.team;
      optH.textContent = t.team;
      $("#homeSelect").appendChild(optH);
      const optA = document.createElement("option");
      optA.value = t.team;
      optA.textContent = t.team;
      $("#awaySelect").appendChild(optA);
    });
    if (teams.length > 1) $("#awaySelect").selectedIndex = 1;

    // A fixture's "Predict" button stashed the teams it wants predicted —
    // select them now that the dropdowns exist, then run the model.
    if (state.pendingPredict) {
      const { home, away } = state.pendingPredict;
      state.pendingPredict = null;
      if (home && away && home !== away) {
        const sel = (select, name) => {
          const el = $(select);
          [...el.options].forEach((o, i) => { if (o.value === name) el.selectedIndex = i; });
        };
        sel("#homeSelect", home);
        sel("#awaySelect", away);
        runPrediction();
      }
    }
  });
}

/* ---------- Predictor ---------- */

$("#predictBtn").addEventListener("click", runPrediction);

async function runPrediction() {
  const home = $("#homeSelect").value;
  const away = $("#awaySelect").value;
  if (!home || !away || home === away) {
    alert("Pick two different teams.");
    return;
  }
  await withLoading(async () => {
    const d = await api(
      `/predict?league=${state.league}&home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`
    );
    renderPrediction(d);
  });
}

/* ---------- Fixtures (new-season previews) ---------- */

let fixState = { teams: [], fixtures: [] };

async function loadFixtures() {
  await withLoading(async () => {
    const d = await api(`/fixtures?league=${state.league}`);
    fixState.teams = d.teams || [];
    fixState.fixtures = d.fixtures || [];
    renderFixtures(d);
  });
}

function fixtureRow(r) {
  const mw = r.matchweek ? `<span class="badge mw">GW${r.matchweek}</span>` : "";
  const unknown = !r.known
    ? ' <span class="badge warn small-badge" title="Team not in training data — model uses league-mean prior">new</span>'
    : "";
  return `<tr>
    <td><span class="fix-date">${r.date}</span></td>
    <td class="team-cell home">${r.home}</td>
    <td class="fix-vs">vs</td>
    <td class="team-cell away">${r.away}${unknown}</td>
    <td>${mw}</td>
    <td class="fix-actions">
      <button class="btn-ghost small fix-predict" data-home="${r.home.replace(/"/g, "&quot;")}" data-away="${r.away.replace(/"/g, "&quot;")}">🎯 Predict</button>
      <button class="btn-icon fix-del" data-home="${r.home.replace(/"/g, "&quot;")}" data-away="${r.away.replace(/"/g, "&quot;")}" data-date="${r.date}" title="Remove fixture">✕</button>
    </td>
  </tr>`;
}

function renderFixtures(d) {
  const rows = d.fixtures;
  const teams = d.teams || [];

  const seasons = d.seasons && d.seasons.length ? d.seasons.join(", ") : "—";
  const mwCount = new Set(rows.map((r) => r.matchweek).filter((x) => x != null)).size;
  $("#fixStats").innerHTML = [
    { label: "Upcoming matches", value: rows.length, icon: "🗓️" },
    { label: "Matchweeks covered", value: rows.length ? mwCount : 0, icon: "📅" },
    { label: "Teams in league", value: teams.length, icon: "🛡️" },
    { label: "Season(s)", value: seasons, icon: "🏆" },
  ]
    .map(
      (s) =>
        `<div class="stat-card"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
    )
    .join("");

  // Populate add-fixture form selects (only teams the model has seen).
  const fill = (sel, selectedIdx) => {
    const el = $(sel);
    el.innerHTML = "";
    teams.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = t;
      o.textContent = t;
      el.appendChild(o);
    });
    if (selectedIdx != null && el.options.length > selectedIdx) el.selectedIndex = selectedIdx;
  };
  fill("#fixHomeSelect", 0);
  fill("#fixAwaySelect", 1);

  // Default the date input to the next Saturday.
  const dateEl = $("#fixDate");
  if (dateEl && !dateEl.value) {
    const next = new Date();
    next.setDate(next.getDate() + ((6 - next.getDay() + 7) % 7) + 7);
    dateEl.value = next.toISOString().slice(0, 10);
  }

  const hasFixtures = rows.length > 0;
  $("#fixEmpty").style.display = hasFixtures ? "none" : "block";
  $("#fixPredictAllBtn").disabled = !hasFixtures;
  $("#fixSeasonLabel").textContent = d.seasons && d.seasons.length ? `· ${d.seasons.join(", ")}` : "";

  $("#fixTable").innerHTML = hasFixtures
    ? `<thead><tr><th>Date</th><th>Home</th><th></th><th>Away</th><th>Round</th><th></th></tr></thead><tbody>${rows.map(fixtureRow).join("")}</tbody>`
    : "";
}

async function addFixture() {
  const home = $("#fixHomeSelect").value;
  const away = $("#fixAwaySelect").value;
  const date = $("#fixDate").value;
  const mw = $("#fixMw").value;
  if (!date) {
    showError("Pick a date for the fixture.");
    return;
  }
  if (!home || !away || home === away) {
    showError("Pick two different teams.");
    return;
  }
  await withLoading(async () => {
    await apiRaw(`/fixtures?league=${state.league}`, {
      method: "POST",
      body: { home, away, date, matchweek: mw ? parseInt(mw, 10) : null },
    });
    $("#fixMw").value = "";
    await loadFixtures();
  });
}

async function removeFixture(home, away, date) {
  await withLoading(async () => {
    await apiRaw(
      `/fixtures?league=${state.league}&home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&date=${encodeURIComponent(date)}`,
      { method: "DELETE" }
    );
    await loadFixtures();
  });
}

async function predictFixture(home, away) {
  // Jump to the Match Predictor with these teams and run the model.
  // loadTeams() repopulates the dropdowns asynchronously, so stash the target
  // teams in state and let loadTeams() apply them once the options exist.
  state.pendingPredict = { home, away };
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  document.querySelector('[data-view="predictor"]').classList.add("active");
  switchView("predictor");
}

function renderFixResults(results) {
  const card = $("#fixResultsCard");
  card.style.display = "block";
  const failed = results.filter((r) => r.error);
  $("#fixResultsHint").textContent = results.length
    ? `${results.length - failed.length}/${results.length} predicted`
    : "";

  $("#fixResultsTable").innerHTML =
    `<thead><tr><th>Match</th><th>H</th><th>D</th><th>A</th><th>xG</th><th>Score</th><th>Pick</th></tr></thead><tbody>` +
    results
      .map((r) => {
        if (r.error) {
          return `<tr><td class="team-cell">${r.home} v ${r.away}</td><td colspan="6" class="muted">⚠️ ${r.error}</td></tr>`;
        }
        const p = r.data.probabilities;
        const xg = r.data.expected_goals;
        const bar = (v) =>
          `<span class="fix-bar"><span style="width:${Math.round(v * 100)}%"></span></span>`;
        const pickCls =
          r.data.predicted_result === "H" ? "h" : r.data.predicted_result === "D" ? "d" : "a";
        return `<tr>
          <td class="team-cell home">${r.home} <span class="muted small">v</span> ${r.away}</td>
          <td>${bar(p.home_win)}<span class="fix-pct">${fmtPct(p.home_win)}</span></td>
          <td>${bar(p.draw)}<span class="fix-pct">${fmtPct(p.draw)}</span></td>
          <td>${bar(p.away_win)}<span class="fix-pct">${fmtPct(p.away_win)}</span></td>
          <td>${fmtNum(xg.home)}–${fmtNum(xg.away)}</td>
          <td>${r.data.most_likely_score}</td>
          <td><span class="badge ${pickCls}">${r.data.predicted_result}</span></td>
        </tr>`;
      })
      .join("") +
    `</tbody>`;
}

async function predictAll() {
  const fixtures = fixState.fixtures;
  if (!fixtures.length) return;
  $("#fixResultsCard").style.display = "block";
  $("#fixResultsHint").textContent = "Predicting…";
  $("#fixResultsTable").innerHTML =
    `<tbody><tr><td colspan="7" class="muted">Running the ensemble on ${fixtures.length} fixtures…</td></tr></tbody>`;

  const results = [];
  let i = 0;
  for (const f of fixtures) {
    i++;
    $("#fixResultsHint").textContent = `Predicting ${i}/${fixtures.length}…`;
    try {
      const data = await api(
        `/predict?league=${state.league}&home=${encodeURIComponent(f.home)}&away=${encodeURIComponent(f.away)}`
      );
      results.push({ home: f.home, away: f.away, date: f.date, data });
    } catch (e) {
      results.push({ home: f.home, away: f.away, date: f.date, error: e.message });
    }
  }
  renderFixResults(results);
}

async function generateRoundRobin() {
  if (
    !confirm(
      "Generate a full double round-robin PLACEHOLDER schedule from the league's current team list (every pair twice, home and away)?\n\nThis is a stand-in until the official new-season fixtures are released — replace it later with the real list."
    )
  ) {
    return;
  }
  await withLoading(async () => {
    const d = await apiRaw(`/fixtures/generate-round-robin?league=${state.league}`, {
      method: "POST",
      body: {},
    });
    await loadFixtures();
    showNotice(`Round-robin saved: ${d.n_added} matches over ${d.n_matchweeks} matchweeks for ${d.n_teams} teams.`);
  });
}

$("#fixAddBtn").addEventListener("click", addFixture);
$("#fixRoundRobinBtn").addEventListener("click", generateRoundRobin);
$("#fixPredictAllBtn").addEventListener("click", predictAll);

document.addEventListener("click", (e) => {
  const predictBtn = e.target.closest(".fix-predict");
  if (predictBtn) {
    predictFixture(predictBtn.dataset.home, predictBtn.dataset.away);
    return;
  }
  const delBtn = e.target.closest(".fix-del");
  if (delBtn) {
    removeFixture(delBtn.dataset.home, delBtn.dataset.away, delBtn.dataset.date);
  }
});

function renderPrediction(d) {
  $("#predictResults").style.display = "block";

  makeChart("probsChart", {
    type: "bar",
    data: {
      labels: [d.home, "Draw", d.away],
      datasets: [
        {
          data: [d.probabilities.home_win, d.probabilities.draw, d.probabilities.away_win],
          backgroundColor: [PALETTE.green, PALETTE.amber, PALETTE.red],
          borderRadius: 8,
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => fmtPct(c.parsed.y) } },
      },
      scales: {
        y: { beginAtZero: true, max: 1, ticks: { color: "#94a3b8", callback: (v) => fmtPct(v) }, grid: { color: "rgba(148,163,184,0.08)" } },
        x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
      },
    },
  });

  $("#xgBoxes").innerHTML = `
    <div class="xg-box"><span class="xg-team">${d.home}</span><span class="xg-val">${fmtNum(d.expected_goals.home)}</span></div>
    <div class="xg-vs">—</div>
    <div class="xg-box away"><span class="xg-team">${d.away}</span><span class="xg-val">${fmtNum(d.expected_goals.away)}</span></div>`;

  $("#miniStats").innerHTML = [
    { k: "Over 2.5 goals", v: fmtPct(d.over_2_5_goals) },
    { k: "BTTS Yes", v: fmtPct(d.btts_yes) },
    { k: "Model pick", v: d.predicted_result },
    { k: "Dynamic xG", v: fmtNum(d.dyn_expected_goals ? d.dyn_expected_goals.home : d.expected_goals.home) + "–" + fmtNum(d.dyn_expected_goals ? d.dyn_expected_goals.away : d.expected_goals.away) },
  ]
    .map((s) => `<div class="mini-stat"><span>${s.k}</span><b>${s.v}</b></div>`)
    .join("");

  $("#scoreBox").textContent = d.most_likely_score;

  $("#marketBox").innerHTML = d.market
    ? `<h5>Market Implied</h5><div class="market-row"><span>H ${fmtPct(d.market.implied_home)}</span><span>D ${fmtPct(d.market.implied_draw)}</span><span>A ${fmtPct(d.market.implied_away)}</span></div>`
    : `<p class="muted">No market odds for this fixture.</p>`;

  const mat = d.score_matrix.probs;
  makeChart("matrixChart", {
    type: "matrix",
    data: {
      datasets: [
        {
          labels: d.score_matrix.away_goals,
          data: mat.flatMap((row, i) =>
            row.map((v, j) => ({ x: j, y: i, v }))
          ),
          backgroundColor(ctx) {
            const v = ctx.dataset.data[ctx.dataIndex].v;
            const alpha = Math.min(1, v * 22);
            return `rgba(56,189,248,${alpha})`;
          },
          width: (ctx) => (ctx.chart.chartArea ? ctx.chart.chartArea.width / 7 - 4 : 30),
          height: (ctx) => (ctx.chart.chartArea ? ctx.chart.chartArea.height / 7 - 4 : 30),
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const it = items[0];
              return `${d.home} ${it.parsed.y} - ${it.parsed.x} ${d.away}`;
            },
            label: (it) => fmtPct(it.parsed.v),
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          offset: true,
          min: 0,
          max: 6,
          ticks: { stepSize: 1, color: "#94a3b8", callback: (v) => `${v}` },
          grid: { display: false },
          title: { display: true, text: d.away + " goals", color: "#94a3b8" },
        },
        y: {
          type: "linear",
          offset: true,
          min: 0,
          max: 6,
          ticks: { stepSize: 1, color: "#94a3b8", callback: (v) => `${v}` },
          grid: { display: false },
          title: { display: true, text: d.home + " goals", color: "#94a3b8" },
        },
      },
    },
  });

  renderComponents(d);
  renderMarketCompare(d);
}

/* Component breakdown: each model family's H/D/A probabilities. */
function renderComponents(d) {
  const comps = d.component_probs || {};
  const names = {
    dixon_coles: "Dixon-Coles",
    dynamic: "Dynamic (Kalman)",
    elo: "Elo",
    xg: "xG Filter",
  };
  const order = ["dixon_coles", "dynamic", "elo", "xg"];
  const rows = order
    .filter((k) => comps[k])
    .map((k) => {
      const [h, dd, a] = comps[k];
      const max = Math.max(h, dd, a, 0.0001);
      return `<div class="comp-row">
        <span class="comp-name">${names[k] || k}</span>
        <div class="comp-track">
          <div class="comp-fill h" style="flex:${(h / max).toFixed(3)}"></div>
          <div class="comp-fill d" style="flex:${(dd / max).toFixed(3)}"></div>
          <div class="comp-fill a" style="flex:${(a / max).toFixed(3)}"></div>
        </div>
        <div class="comp-vals">
          <span class="c-h">${fmtPct(h)}</span><span class="c-d">${fmtPct(dd)}</span><span class="c-a">${fmtPct(a)}</span>
        </div>
      </div>`;
    })
    .join("");
  $("#componentBars").innerHTML =
    rows ||
    `<p class="muted">No component breakdown available.</p>`;
}

/* Market vs model: bar comparison of implied vs model probabilities. */
function renderMarketCompare(d) {
  const el = $("#marketCompareBox");
  if (!d.market) {
    el.innerHTML =
      `<p class="muted">No closing line recorded for this fixture. Pass odds to the API to enable the residual-vs-market layer.</p>`;
    return;
  }
  const mk = d.market;
  const labels = [`${d.home}`, "Draw", `${d.away}`];
  const rows = labels
    .map((label, i) => {
      const implied = [mk.implied_home, mk.implied_draw, mk.implied_away][i];
      const model = [d.probabilities.home_win, d.probabilities.draw, d.probabilities.away_win][i];
      const edge = [mk.edge_home, mk.edge_draw, mk.edge_away][i];
      const cls = edge > 0.01 ? "pos" : edge < -0.01 ? "neg" : "flat";
      return `<div class="mm-row">
        <span class="mm-label">${label}</span>
        <span class="mm-bar-wrap"><span class="mm-bar model" style="width:${Math.round(model * 100)}%"></span></span>
        <span class="mm-val model">${fmtPct(model)}</span>
        <span class="mm-bar-wrap"><span class="mm-bar market" style="width:${Math.round(implied * 100)}%"></span></span>
        <span class="mm-val market">${fmtPct(implied)}</span>
        <span class="mm-edge ${cls}">${edge >= 0 ? "+" : ""}${(edge * 100).toFixed(1)}pt</span>
      </div>`;
    })
    .join("");
  el.innerHTML =
    `<div class="mm-head"><span></span><b>Model</b><b>Market</b><b>Edge</b></div>` + rows +
    `<p class="muted small">Market line from last meeting's closing odds. Positive edge = model sees value the market hasn't priced.</p>`;
}

/* ---------- Backtest Lab ---------- */

async function loadBacktest() {
  await withLoading(async () => {
    const d = await api(`/backtest?league=${state.league}`);
    const m = d.metrics;

    const acc = typeof m.accuracy === "number" && m.accuracy <= 1 ? m.accuracy : m.accuracy / 100;
    const mk = m.market || {};
    const st = m.staking || {};
    const resid = mk.residual_log_loss;

    const statCards = [
      { label: "Matches", value: d.n_matches.toLocaleString(), icon: "📅" },
      { label: "Accuracy", value: fmtPct(acc), icon: "🎯" },
      { label: "Log Loss", value: fmtNum(m.log_loss, 4), icon: "📉" },
      { label: "Brier", value: fmtNum(m.brier, 4), icon: "📊" },
      { label: "Residual log loss", value: resid == null ? "—" : (resid >= 0 ? "+" : "") + resid.toFixed(4), icon: "🎰", tone: resid != null && resid < 0 ? "good" : resid != null && resid > 0 ? "bad" : "" },
      { label: "Edge correlation", value: m.edge_corr == null ? "—" : (m.edge_corr >= 0 ? "+" : "") + m.edge_corr.toFixed(3), icon: "📈", tone: m.edge_corr != null && m.edge_corr > 0 ? "good" : "bad" },
      { label: "Kelly Sharpe", value: st.sharpe == null ? "—" : fmtNum(st.sharpe, 2), icon: "⚖️" },
      { label: "Staking ROI", value: st.roi == null ? "—" : fmtPct(st.roi), icon: "💰" },
    ];

    $("#btStats").innerHTML = statCards
      .map(
        (s) =>
          `<div class="stat-card ${s.tone || ""}"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
      )
      .join("");

    makeChart("btCompareChart", {
      type: "bar",
      data: {
        labels: ["Log Loss", "Brier", "Accuracy"],
        datasets: [
          {
            label: "Model",
            data: [m.log_loss, m.brier, acc],
            backgroundColor: PALETTE.blue,
            borderRadius: 6,
          },
          {
            label: "Baseline",
            data: [m.baseline_log_loss, null, null],
            backgroundColor: "rgba(148,163,184,0.6)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
        scales: {
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        },
      },
    });

    const months = Object.keys(d.monthly_accuracy || {});
    makeChart("btMonthlyChart", {
      type: "line",
      data: {
        labels: months,
        datasets: [
          {
            label: "Monthly accuracy",
            data: months.map((k) => d.monthly_accuracy[k]),
            borderColor: PALETTE.green,
            backgroundColor: "rgba(52,211,153,0.12)",
            fill: true,
            tension: 0.35,
            pointRadius: 3,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#94a3b8", maxRotation: 45, maxTicksLimit: 10 }, grid: { color: "rgba(148,163,184,0.08)" } },
          y: { min: 0, max: 1, ticks: { color: "#94a3b8", callback: (v) => fmtPct(v) }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
      },
    });

    makeChart("btMarketChart", {
      type: "bar",
      data: {
        labels: ["Model", "Market", "Baseline"],
        datasets: [
          {
            label: "Log loss (lower = better)",
            data: [
              mk.model_log_loss != null ? mk.model_log_loss : null,
              mk.market_log_loss != null ? mk.market_log_loss : null,
              m.baseline_log_loss,
            ],
            backgroundColor: [PALETTE.blue, PALETTE.amber, "rgba(148,163,184,0.6)"],
            borderRadius: 6,
          },
        ],
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => c.parsed.y != null ? fmtNum(c.parsed.y, 4) : "—" } },
        },
        scales: {
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        },
      },
    });

    const stCards = [
      { k: "Bets staked", v: fmtNum(st.n, 0) },
      { k: "Total staked (bankroll units)", v: fmtNum(st.total_staked, 2) },
      { k: "Profit (units)", v: st.profit_units == null ? "—" : (st.profit_units >= 0 ? "+" : "") + st.profit_units.toFixed(3) },
      { k: "ROI", v: fmtPct(st.roi) },
      { k: "Sharpe (annualized)", v: fmtNum(st.sharpe, 2) },
      { k: "Max drawdown", v: fmtNum(st.max_drawdown, 3) },
      { k: "CVaR 95%", v: fmtNum(st.cvar95, 3) },
    ];
    $("#stakingPanel").innerHTML =
      `<div class="market-grid">` +
      stCards
        .map(
          (s) =>
            `<div><span class="muted">${s.k}</span><b>${s.v}</b></div>`
        )
        .join("") +
      `</div>` +
      `<p class="muted small">Covariance-adjusted fractional Kelly portfolio sizing. Negative profit/Sharpe means the value edge has not paid out in this window — the honest result.</p>`;

    $("#marketPanel").innerHTML = m.market
      ? `<div class="market-grid">
          <div><span class="muted">Market log loss</span><b>${fmtNum(m.market.market_log_loss, 4)}</b></div>
          <div><span class="muted">Model vs market</span><b>${fmtNum(m.market.model_minus_market_log_loss, 4)}</b></div>
          <div><span class="muted">Value bets flagged</span><b>${m.market.value_bets}</b></div>
          <div><span class="muted">Value bet yield</span><b>${fmtPct(m.market.value_bet_yield)}</b></div>
        </div>
        ${renderValueBetsTable(d.recent_value_bets)}`
      : `<p class="muted">Market data not available for this league.</p>`;
  });
}

/* Recent flagged value bets (most informative when the aggregate looks bad). */
function renderValueBetsTable(bets) {
  if (!bets || !bets.length) return "";
  const rows = bets
    .map(
      (b) => `<tr>
        <td class="muted">${b.date}</td>
        <td class="team-cell">${b.home_team} <span class="muted">vs</span> ${b.away_team}</td>
        <td><b>${b.market}</b></td>
        <td>${fmtPct(b.model_prob)}</td>
        <td>${fmtPct(b.implied_prob)}</td>
        <td class="${b.edge > 0 ? "c-h" : "c-a"}">${(b.edge * 100).toFixed(1)}pt</td>
        <td>${b.odds}</td>
        <td class="${b.pnl >= 0 ? "c-h" : "c-a"}">${b.pnl >= 0 ? "+" : ""}${b.pnl}</td>
      </tr>`
    )
    .join("");
  return `<h5 style="margin-top:16px">Recent value bets</h5>
    <div class="table-wrap" style="max-height:320px;overflow:auto">
      <table class="table">
        <thead><tr><th>Date</th><th>Fixture</th><th>Pick</th><th>Model</th><th>Market</th><th>Edge</th><th>Odds</th><th>P&L</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ---------- Research ---------- */

async function loadResearch() {
  await withLoading(async () => {
    const d = await api(`/research?league=${state.league}`);
    const h = d.holdout_result || {};
    const ctl = d.control || {};
    const pw = d.power || {};

    // Verdict stat cards
    const resStats = [
      { label: "Holdout season", value: h.season || "—", icon: "📅" },
      { label: "Holdout matches", value: h.n_matches != null ? h.n_matches : "—", icon: "🎯" },
      { label: "Residual log loss", value: h.residual_log_loss == null ? "—" : (h.residual_log_loss >= 0 ? "+" : "") + h.residual_log_loss.toFixed(4), icon: "📉", tone: h.residual_log_loss != null && h.residual_log_loss < 0 ? "good" : "bad" },
      { label: "Edge correlation", value: h.edge_corr == null ? "—" : (h.edge_corr >= 0 ? "+" : "") + h.edge_corr.toFixed(3), icon: "📈", tone: h.edge_corr != null && h.edge_corr > 0.02 ? "good" : "bad" },
      { label: "Value-bet ROI", value: h.value_bet_roi == null ? "—" : fmtPct(h.value_bet_roi), icon: "💰", tone: h.value_bet_roi != null && h.value_bet_roi > 0 ? "good" : "bad" },
      { label: "Verdict", value: h.verdict || "—", icon: h.verdict === "NULL" ? "🧪" : "✅" },
    ];
    $("#resStats").innerHTML = resStats
      .map(
        (s) =>
          `<div class="stat-card ${s.tone || ""}"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
      )
      .join("");

    // Pre-registered holdout panel
    $("#holdoutPanel").innerHTML = h.verdict
      ? `<div class="market-grid">
          <div><span class="muted">Protocol</span><b>${d.protocol.doc || "docs/edge_test_preregistration.md"}</b></div>
          <div><span class="muted">Primary bar</span><b>${d.protocol.primary_threshold || "—"}</b></div>
          <div><span class="muted">Secondary bars</span><b>${d.protocol.secondary_thresholds || "—"}</b></div>
          <div><span class="muted">Detail</span><b>${h.verdict_detail || "—"}</b></div>
        </div>
        <p class="muted small">Run: backtest.py --league ${d.league} --xg-dir data/xg --holdout-seasons 1</p>`
      : `<p class="muted">No pre-registered holdout result recorded yet (run with --holdout-seasons 1).</p>`;

    // Market-only control table
    const cRows = (ctl.rows || [])
      .map(
        (r) =>
          `<tr><td class="team-cell">${r.label}</td><td>${r.n}</td><td>${fmtPct(r.roi)}</td><td>${fmtNum(r.sharpe, 2)}</td><td>${fmtPct(r.strike)}</td></tr>`
      )
      .join("");
    const residLines = Object.entries(ctl.residual_by_line || {})
      .map(([k, v]) => `<div class="muted small">${k}: ${v.residual >= 0 ? "+" : ""}${v.residual.toFixed(4)}</div>`)
      .join("");
    $("#controlPanel").innerHTML =
      `<table class="table"><thead><tr><th>Strategy</th><th>n</th><th>ROI</th><th>Sharpe</th><th>Strike</th></tr></thead><tbody>${cRows}</tbody></table>
       <p class="muted small">Avg bookmaker margin: <b>${fmtPct(ctl.avg_margin)}</b>. A control that loses ~the margin means the model's worse result is genuine.</p>
       ${residLines}`;

    // Statistical power panel
    const powerItems = [
      { k: "Per-match residual SD", v: fmtNum(pw.per_match_sd, 4) },
      { k: "Min detectable edge (this sample)", v: pw.min_detectable_edge_1y == null ? "—" : fmtPct(pw.min_detectable_edge_1y) },
      { k: "OOS matches for a 1% edge", v: fmtNum(pw.n_for_1pct_edge, 0) },
      { k: "OOS matches for a 2% edge", v: fmtNum(pw.n_for_2pct_edge, 0) },
      { k: "OOS matches for a 3% edge", v: fmtNum(pw.n_for_3pct_edge, 0) },
      { k: "Edge correlation (tuning window)", v: pw.edge_corr == null ? "—" : (pw.edge_corr >= 0 ? "+" : "") + pw.edge_corr.toFixed(4) },
      { k: "OOS matches to detect corr 0.03", v: fmtNum(pw.n_for_corr_003, 0) },
    ];
    $("#powerPanel").innerHTML =
      `<div class="market-grid">` +
      powerItems.map((s) => `<div><span class="muted">${s.k}</span><b>${s.v}</b></div>`).join("") +
      `</div><p class="muted small">One-sided alpha 0.05, power 0.80. If required n exceeds what you can gather, a null is 'insufficient power for a small edge', not 'no edge'.</p>`;

    // Calibration report table
    const calRows = (d.calibration || [])
      .map(
        (r) =>
          `<tr><td class="team-cell">${r.combo}</td><td>${fmtNum(r.log_loss, 4)}</td><td>${r.residual_log_loss == null ? "—" : (r.residual_log_loss >= 0 ? "+" : "") + r.residual_log_loss.toFixed(4)}</td><td>${r.edge_corr == null ? "—" : fmtNum(r.edge_corr, 3)}</td><td>${fmtNum(r.kelly_sharpe, 2)}</td></tr>`
      )
      .join("");
    $("#calTable").innerHTML =
      d.calibration && d.calibration.length
        ? `<thead><tr><th>Combo</th><th>Log loss</th><th>Residual</th><th>Edge corr</th><th>Sharpe</th></tr></thead><tbody>${calRows}</tbody>`
        : `<tbody><tr><td class="muted">Run scripts/calibrate_model.py to populate.</td></tr></tbody>`;
  });
}

/* ---------- League Compare ---------- */

async function loadCompare() {
  await withLoading(async () => {
    const d = await api(`/compare`);
    const leagues = Object.keys(d);
    if (!leagues.length) {
      $("#cmpStats").innerHTML = "<p class='muted'>No backtest results available.</p>";
      return;
    }

    const bestLL = leagues.map((l) => d[l].log_loss).reduce((a, b) => Math.min(a, b)).toFixed(4);
    const bestBrier = leagues.map((l) => d[l].brier).reduce((a, b) => Math.min(a, b)).toFixed(4);
    const bestAcc = Math.max(...leagues.map((l) => d[l].accuracy));

    $("#cmpStats").innerHTML = [
      { label: "Leagues", value: leagues.length, icon: "🏆" },
      { label: "Best log loss", value: bestLL, icon: "📉" },
      { label: "Best Brier", value: bestBrier, icon: "📊" },
      { label: "Best accuracy", value: fmtPct(bestAcc), icon: "🎯" },
    ]
      .map(
        (s) =>
          `<div class="stat-card"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
      )
      .join("");

    makeChart("cmpLogLoss", {
      type: "bar",
      data: {
        labels: leagues,
        datasets: [
          {
            label: "Model",
            data: leagues.map((l) => d[l].log_loss),
            backgroundColor: PALETTE.blue,
            borderRadius: 8,
          },
          {
            label: "Baseline",
            data: leagues.map((l) => d[l].baseline_log_loss),
            backgroundColor: "rgba(148,163,184,0.55)",
            borderRadius: 8,
          },
        ],
      },
      options: {
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
        scales: {
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        },
      },
    });

    makeChart("cmpBrier", {
      type: "bar",
      data: {
        labels: leagues,
        datasets: [
          {
            label: "Brier",
            data: leagues.map((l) => d[l].brier),
            backgroundColor: leagues.map((_, i) => [PALETTE.green, PALETTE.amber, PALETTE.purple][i % 3]),
            borderRadius: 8,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
          x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        },
      },
    });

    const vbRows = leagues
      .map((l) => {
        const mk = d[l].market || {};
        const st = d[l].staking || {};
        const resid = mk.residual_log_loss;
        return `<tr>
          <td class="team-cell">${LEAGUE_LABELS[l] || l}${d[l].xg_available ? ' <span class="badge xg small-badge">xG</span>' : ""}</td>
          <td>${fmtNum(mk.n || 0, 0)}</td>
          <td>${d[l].value_bets || 0}</td>
          <td>${resid == null ? "—" : (resid >= 0 ? "+" : "") + resid.toFixed(4)}</td>
          <td>${d[l].edge_corr == null ? "—" : (d[l].edge_corr >= 0 ? "+" : "") + d[l].edge_corr.toFixed(3)}</td>
          <td>${st.roi == null ? "—" : fmtPct(st.roi)}</td>
          <td>${st.sharpe == null ? "—" : fmtNum(st.sharpe, 2)}</td>
          <td>${st.max_drawdown == null ? "—" : fmtNum(st.max_drawdown, 3)}</td>
        </tr>`;
      })
      .join("");
    $("#valueBetPanel").innerHTML =
      `<table class="table"><thead><tr>
         <th>League</th><th>Market matches</th><th>Value bets</th><th>Residual log loss</th>
         <th>Edge corr</th><th>Kelly ROI</th><th>Sharpe</th><th>Max DD</th>
       </tr></thead><tbody>` + vbRows + `</tbody></table>`
      + `<p class="muted small">Residual log loss = model − market (negative means the model adds information beyond the closing line). Edge corr = does predicted value edge predict winning?</p>`;
  });
}

/* ---------- Boot ---------- */

(async function boot() {
  try {
    await api("/health");
    $("#apiStatus").textContent = "API online";
    document.querySelector(".dot").classList.add("ok");
  } catch (e) {
    $("#apiStatus").textContent = "API offline";
    document.querySelector(".dot").classList.add("err");
  }
  loadDashboard();
})();