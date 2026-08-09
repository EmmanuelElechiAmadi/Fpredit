/* FootyIntel — frontend logic
 * Talks to the FastAPI backend (app/main.py) and renders Chart.js visualizations.
 */

const state = {
  league: "EPL",
  leagueMeta: { label: "Premier League", code: "E0", country: "England" },
  teams: [],
  charts: {},
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
  state.charts[id] = new Chart(ctx, config);
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
  teams: { title: "Team Intelligence", sub: "Ratings, strength & form per team" },
  predictor: { title: "Match Predictor", sub: "H/D/A, expected goals & score matrix" },
  backtest: { title: "Backtest Lab", sub: "Walk-forward validation vs baselines" },
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

  if (view === "dashboard") loadDashboard();
  if (view === "teams") loadTeams();
  if (view === "predictor") loadTeams();
  if (view === "backtest") loadBacktest();
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

/* ---------- Loading overlay ---------- */

let loadingCount = 0;
function showLoading() {
  loadingCount++;
  $("#loadingOverlay").classList.add("visible");
}
function hideLoading() {
  loadingCount = Math.max(0, loadingCount - 1);
  if (loadingCount === 0) $("#loadingOverlay").classList.remove("visible");
}

async function withLoading(fn) {
  showLoading();
  try {
    return await fn();
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
      `<thead><tr><th>Team</th><th>Elo</th><th>Attack</th><th>Defense</th><th>Form</th></tr></thead><tbody>` +
      teams
        .map(
          (t) =>
            `<tr><td class="team-cell">${t.team}</td><td>${fmtNum(t.elo_rating, 0)}</td><td>${fmtNum(t.attack, 3)}</td><td>${fmtNum(t.defense, 3)}</td><td><span class="form-dots small">${(t.form || "")
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
}

/* ---------- Backtest Lab ---------- */

async function loadBacktest() {
  await withLoading(async () => {
    const d = await api(`/backtest?league=${state.league}`);
    const m = d.metrics;

    const acc = typeof m.accuracy === "number" && m.accuracy <= 1 ? m.accuracy : m.accuracy / 100;

    $("#btStats").innerHTML = [
      { label: "Matches", value: d.n_matches.toLocaleString(), icon: "📅" },
      { label: "Accuracy", value: fmtPct(acc), icon: "🎯" },
      { label: "Log Loss", value: fmtNum(m.log_loss, 4), icon: "📉" },
      { label: "Brier", value: fmtNum(m.brier, 4), icon: "📊" },
    ]
      .map(
        (s) =>
          `<div class="stat-card"><div class="stat-icon">${s.icon}</div><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
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

    $("#marketPanel").innerHTML = m.market
      ? `<div class="market-grid">
          <div><span class="muted">Market log loss</span><b>${fmtNum(m.market.market_log_loss, 4)}</b></div>
          <div><span class="muted">Model vs market</span><b>${fmtNum(m.market.model_minus_market_log_loss, 4)}</b></div>
          <div><span class="muted">Value bets flagged</span><b>${m.market.value_bets}</b></div>
          <div><span class="muted">Value bet yield</span><b>${fmtPct(m.market.value_bet_yield)}</b></div>
        </div>`
      : `<p class="muted">Market data not available for this league.</p>`;
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

    $("#valueBetPanel").innerHTML = `<div class="market-grid">
      ${leagues
        .map(
          (l) =>
            `<div class="value-card"><h5>${LEAGUE_LABELS[l] || l}</h5>
             <span class="muted">Value bets:</span> <b>${d[l].value_bets || 0}</b><br/>
             <span class="muted">Market log loss:</span> <b>${d[l].market ? fmtNum(d[l].market.market_log_loss, 4) : "—"}</b></div>`
        )
        .join("")}
    </div>`;
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