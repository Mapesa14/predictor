import React, { useEffect, useState } from "react";

function useHash() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

function useFetch(url) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then((data) => alive && setState({ loading: false, data, error: null }))
      .catch((e) => alive && setState({ loading: false, data: null, error: e.message }));
    return () => { alive = false; };
  }, [url]);
  return state;
}

// ---- live scores (best-effort; never wrongly match) ------------------------
const ABBREV = {
  utd: "united", manu: "manchester", man: "manchester",
  spurs: "tottenham", "nott'm": "nottingham", nottm: "nottingham",
  wba: "westbromwich", bha: "brighton",
  fc: "", sc: "", cf: "", afc: "", ac: "", as: "", us: "", st: "",
  de: "", ol: "", om: "", cfc: "", pdfc: "", bsc: "", hsc: "", ssc: "",
  "": "",
};

function toks(name) {
  if (!name) return new Set();
  const s = name.replace(/&/g, " ").replace(/[^a-z0-9 ]+/gi, " ");
  const out = new Set();
  for (let w of s.toLowerCase().split(/\s+/)) {
    w = ABBREV[w] ?? w;
    if (w.length >= 4) out.add(w);
  }
  return out;
}

function subMatch(a, b) {
  const ta = toks(a), tb = toks(b);
  if (!ta.size || !tb.size) return false;
  if (ta.size === tb.size) return [...ta].every((w) => tb.has(w));
  const s = ta.size < tb.size ? ta : tb;
  const l = ta.size < tb.size ? tb : ta;
  return [...s].every((w) => l.has(w)) && l.size - s.size <= 1;
}

function useLive(interval = 60000) {
  const [live, setLive] = useState(null);
  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetch("/api/live")
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then((d) => alive && setLive(d))
        .catch(() => alive && setLive((prev) => prev || { enabled: false }));
    poll();
    const t = setInterval(poll, interval);
    return () => { alive = false; clearInterval(t); };
  }, [interval]);
  return live;
}

function LiveChip({ m, live }) {
  if (!live || !live.enabled) return null;
  const row = live.matches.find((lm) => subMatch(lm.home, m.home) && subMatch(lm.away, m.away));
  if (!row) return null;
  const st = row.status;
  const liveNow = ["1H", "HT", "2H", "ET", "BT", "P", "INT"].includes(st);
  const ft = ["FT", "AET", "PEN"].includes(st) || st === "TBD";
  if (st === "NS" || st === "POSTP" || st === "CANC" || st === "SUSP") return null;
  return (
    <em className={liveNow ? "live" : "ft"}>
      {liveNow ? row.minute + "' " : "FT "}
      {row.hg}-{row.ag}
    </em>
  );
}

const pct = (p) => Math.round(p * 100) + "%";

function confLevel(res) {
  const p = Math.max(res["1"], res.X, res["2"]);
  if (p >= 0.5) return "Clear";
  if (p >= 0.38) return "Slight lean";
  return "Close to a coin toss";
}
const confFilter = (lvl) => (m) => {
  const p = Math.max(m.p["1"], m.p.X, m.p["2"]);
  if (lvl === "Clear") return p >= 0.5;
  if (lvl === "Slight") return p >= 0.38;
  if (lvl === "Coin") return p < 0.38;
  return true;
};

function Theme() {
  const [t, setT] = useState(() => localStorage.getItem("fp-theme") || "");
  useEffect(() => {
    const el = document.documentElement;
    if (t) { el.dataset.theme = t; localStorage.setItem("fp-theme", t); }
    else { delete el.dataset.theme; localStorage.removeItem("fp-theme"); }
  }, [t]);
  const next = { "": "Light", Light: "Dark", Dark: "" };
  return (
    <button className="chip" title="Colour theme" onClick={() => setT(next[t])}>
      {t || "Auto"}
    </button>
  );
}

function Slate() {
  const [days, setDays] = useState(1);
  const [q, setQ] = useState("");
  const [lg, setLg] = useState("");
  const [conf, setConf] = useState("");
  const live = useLive();
  const { loading, data, error } = useFetch("/api/slate?days=" + days);

  if (loading) return <p className="note">Loading the slate…</p>;
  if (error) return <p className="note">Can't reach the service: {error}. Is it running on :8000?</p>;

  const ql = q.trim().toLowerCase();
  const filter = (m) =>
    (!lg || m.div === lg) &&
    (!ql || m.home.toLowerCase().includes(ql) || m.away.toLowerCase().includes(ql)) &&
    confFilter(conf)(m);

  const leagues = [...new Set(data.groups.map((g) => g.code))].sort();
  const groups = data.groups.map((g) => ({ ...g, matches: g.matches.filter(filter) }));
  const total = groups.reduce((n, g) => n + g.matches.length, 0);

  return (
    <div>
      <header className="masthead">
        <div>
          <h1>Football <b>Predictor</b></h1>
          <p className="sub">
            {total} fixtures with a kick-off · {new Date(data.generated).toLocaleString()}
            {data.skipped.length > 0 && " · " + data.skipped.length + " not rated"}
          </p>
        </div>
        <div className="tools"><Theme /></div>
      </header>

      <div className="tools" style={{ marginTop: 10 }}>
        {[1, 2, 3].map((n) => (
          <button key={n} className={"chip" + (days === n ? " on" : "")} onClick={() => setDays(n)}>
            {n === 1 ? "Today" : n + " days"}
          </button>
        ))}
        <select className="sel" value={lg} onChange={(e) => setLg(e.target.value)}>
          <option value="">All leagues</option>
          {leagues.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select className="sel" value={conf} onChange={(e) => setConf(e.target.value)}>
          <option value="">Any confidence</option>
          <option value="Clear">Clear</option>
          <option value="Slight">Slight lean</option>
          <option value="Coin">Coin-toss</option>
        </select>
        <input className="textin" placeholder="Search a club…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>

      {live && !live.enabled && (
        <p className="note">Live scores off — set <code>LIVE_API_KEY</code> on the service.</p>
      )}

      <section>
        <div className="shead">
          <h2>Today's fixtures</h2>
          <span className="n">{total}</span>
          <div className="rule" />
        </div>
        {total === 0 && (
          <p className="note">No schedule loaded for these days yet. Offer time for fixtures to be published, then refresh from football-data.co.uk.</p>
        )}
        {groups.map((g) => (
          <div key={g.code}>
            {g.matches.length > 0 && (
              <div className="shead">
                <h2>{g.league}</h2>
                <span className="n">{g.matches.length}</span>
                <div className="rule" />
              </div>
            )}
            {g.matches.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th>Kick-off</th>
                    <th>Match</th>
                    <th className="num">1</th>
                    <th className="num">X</th>
                    <th className="num">2</th>
                    <th className="num">O2.5</th>
                    <th className="num">BTTS</th>
                  </tr>
                </thead>
                <tbody>
                  {g.matches.map((m, mi) => (
                    <tr key={mi}>
                      <td className="kick">{m.kickoff_label}</td>
                      <td>
                        <a className="match" href={"#/card?home=" + encodeURIComponent(m.home) + "&away=" + encodeURIComponent(m.away) + "&div=" + m.div + "&ko=" + encodeURIComponent(m.kickoff_label) + "&lg=" + encodeURIComponent(g.league)}>
                          <span className="hm">{m.home}</span>
                          <span className="vs">v</span>
                          <span className="aw">{m.away}</span>
                          {m.new && <em className="new">new</em>}
                          {m.started && <em className="live">started</em>}
                          <LiveChip m={m} live={live} />
                        </a>
                      </td>
                      <td className={"num" + (m.pick === "1" ? " win" : "")}>{pct(m.p["1"])}</td>
                      <td className={"num" + (m.pick === "X" ? " win" : "")}>{pct(m.p.X)}</td>
                      <td className={"num" + (m.pick === "2" ? " win" : "")}>{pct(m.p["2"])}</td>
                      <td className="num">{pct(m.o25)}</td>
                      <td className="num">{pct(m.btts)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </section>

      {data.pairings.length > 0 && (
        <section className="pairings">
          <div className="shead">
            <h2>Head-to-head ratings — not scheduled fixtures</h2>
            <span className="n">{data.pairings.reduce((n, g) => n + g.matches.length, 0)}</span>
            <div className="rule" />
          </div>
          <p className="note">These pairings have no published kick-off in the loaded schedule. They appear nowhere under "Today's fixtures"; treat them as ratings of a matchup, not a fixture.</p>
          {data.pairings.map((g) => (
            <div key={g.code}>
              <div className="shead">
                <h2>{g.league}</h2>
                <div className="rule" />
              </div>
              <table>
                <thead>
                  <tr><th>Match</th><th className="num">1</th><th className="num">X</th><th className="num">2</th></tr>
                </thead>
                <tbody>
                  {g.matches.map((m, mi) => (
                    <tr key={mi}>
                      <td><span className="hm">{m.home}</span> <span className="vs">v</span> <span className="aw">{m.away}</span>
                        {m.new && <em className="new"> new</em>}
                      </td>
                      <td className={"num" + (m.pick === "1" ? " win" : "")}>{pct(m.p["1"])}</td>
                      <td className={"num" + (m.pick === "X" ? " win" : "")}>{pct(m.p.X)}</td>
                      <td className={"num" + (m.pick === "2" ? " win" : "")}>{pct(m.p["2"])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </section>
      )}

      <p className="note" style={{ marginTop: 36, color: "var(--ink3)" }}>
        18+ only · Gambling can be addictive. Play responsibly. These are model judgements, not tips — the odds carry a margin, and averaged across thousands of bets the model trails the market.
      </p>
    </div>
  );
}

// ---- card -------------------------------------------------------------
function tierOf(s) {
  if (s.market_used) return ["PRICED", "closed price folded into the forecast"];
  if (s.home_new || s.away_new) return ["BRIDGED", "one or both clubs rated from another league"];
  return ["FULL", "both clubs regulars in this division"];
}

function Row({ label, cells }) {
  return (
    <tr><td className="lbl">{label}</td>{cells.map((c, i) => <td key={i} className="num">{c}</td>)}</tr>
  );
}

function Sum({ title, children }) {
  return (
    <details className="sum">
      <summary>{title}</summary>
      {children}
    </details>
  );
}

function ScoreGrid({ g }) {
  const flat = (h, a) => (g && g[h] && g[h][a]) || 0;
  let mh = 0, ma = 0, mp = -1;
  for (let h = 0; h < 6; h++)
    for (let a = 0; a < 6; a++)
      if (flat(h, a) > mp) { mp = flat(h, a); mh = h; ma = a; }
  const [sel, setSel] = useState({ h: mh, a: ma });
  const sp = flat(sel.h, sel.a);
  const margin = sel.h > sel.a ? "a home win" : sel.h < sel.a ? "an away win" : "a draw";
  const total = sel.h + sel.a;

  return (
    <div>
      <div
        className="sgrid"
        role="grid"
        aria-label="Scoreline heatmap. Rows are goals for the home team, columns goals for the away team; deeper red means a more likely exact score."
      >
        <div className="head corner" aria-hidden="true">H\A</div>
        {[0, 1, 2, 3, 4, 5].map((a) => (
          <div className="head" key={"h" + a} aria-hidden="true">{a}</div>
        ))}
        {[0, 1, 2, 3, 4, 5].map((h) => [
          <div className="head" key={"r" + h} aria-hidden="true">{h}</div>,
          [0, 1, 2, 3, 4, 5].map((a) => {
            const p = flat(h, a);
            const alpha = Math.min(0.92, p * 7.5 + 0.05);
            const modal = h === mh && a === ma;
            const selCell = h === sel.h && a === sel.a;
            const style = {
              background: modal
                ? "var(--spot)"
                : "rgba(var(--spot-rgb), " + alpha.toFixed(3) + ")",
              color: modal || alpha > 0.5 ? "var(--on-spot)" : "var(--ink)",
            };
            return (
              <button
                key={h + "-" + a}
                className={"cell" + (modal ? " modal" : "") + (selCell && !modal ? " pick" : "")}
                style={style}
                role="gridcell"
                aria-pressed={selCell}
                aria-label={h + " goals home, " + a + " goals away, " + pct(p)}
                onClick={() => setSel({ h, a })}
              >
                {pct(p)}
              </button>
            );
          }),
        ])}
      </div>
      <p className="gridcap">
        {sel.h}-{sel.a} · {pct(sp)} — {margin} with {total} goals on the board
        <span className="hint">Tap any cell. Deeper red means a more likely exact scoreline.</span>
      </p>
    </div>
  );
}

function Card() {
  const params = new URLSearchParams(window.location.hash.split("?")[1] || "");
  const home = params.get("home") || "";
  const away = params.get("away") || "";
  const div = params.get("div") || "";
  const ko = params.get("ko") || "";
  const lg = params.get("lg") || "";
  const [neutral, setNeutral] = useState(false);
  const url = "/api/card?home=" + encodeURIComponent(home) + "&away=" + encodeURIComponent(away) +
    "&div=" + encodeURIComponent(div) + (neutral ? "&neutral=1" : "");
  const { loading, data, error } = useFetch(url);

  return (
    <div>
      <p><a className="back" href="#/">← All fixtures</a></p>
      {loading && <p className="note">Fitting the models…</p>}
      {error && <p className="note">Card failed: {error}</p>}
      {data && <CardBody s={data} ko={ko} lg={lg} neutral={neutral} setNeutral={setNeutral} />}
    </div>
  );
}

function CardBody({ s, ko, lg, neutral, setNeutral }) {
  const [tier, tierNote] = tierOf(s);
  const fc = s.totals; // {'0.5':{over,under},...}
  const ladder = Object.keys(fc).sort((x, y) => Number(x) - Number(y));
  const cs = s.correct_scores || [];
  const marginLbl = (k) =>
    ({ home_by_1: "Home by 1", away_by_1: "Away by 1", home_by_2: "Home by 2", away_by_2: "Away by 2",
       home_by_3: "Home by 3", away_by_3: "Away by 3", home_by_4: "Home by 4", away_by_4: "Away by 4" }[k] || k);

  return (
    <div className="matchcard">
      <div className="teams">
        <div className="club club-h">
          <div className="league">{lg || s.div || "Match card"}</div>
          <h3>{s.home}</h3>
        </div>
        <div className="vglobe">v</div>
        <div className="club club-a">
          <div className="league">{ko}</div>
          <h3>{s.away}</h3>
        </div>
      </div>

      <div className="bar">
        {[["1", "Home"], ["X", "Draw"], ["2", "Away"]].map(([k, letter]) => (
          <div key={k} className={"seg" + (s.pick === k ? " pick" : "")} style={{ flex: s.result[k] }}>
            <i>{letter}</i>
            <b>{pct(s.result[k])}</b>
          </div>
        ))}
      </div>
      <p className="confphrase">
        Model says <b>{confLevel(s.result)}</b> · most likely {cs[0] && cs[0][0]}-{cs[0][1]} at {pct(cs[0][2])}
      </p>

      <div className="meta">
        <span>xG&nbsp;<b>{s.exp_home.toFixed(2)}</b>–<b>{s.exp_away.toFixed(2)}</b></span>
        <span>games <b>{s.n_train}</b></span>
        <span className="tier" title={tierNote}>{tier}</span>
      </div>

      <div className="gridrow">
        <span className="lbl">Scoreline heatmap · rows home goals, columns away</span>
        <button className={"chip" + (neutral ? " on" : "")} onClick={() => setNeutral(!neutral)}>
          {neutral ? "Neutral venue" : "Home advantage on"}
        </button>
        <span className="lbl" style={{ color: "var(--ink3)" }}>
          {neutral ? "Simulated on a neutral pitch" : "Home advantage modelled: " + pct(s.home_adv)}
        </span>
      </div>

      {s.score_grid ? <ScoreGrid g={s.score_grid} /> : null}

      <Sum title="Over / Under goals">
        <table className="markets">
          <tbody>
            {ladder.map((k) => (
              <Row key={k} label={"Over " + k + " / Under"} cells={[pct(fc[k].over), pct(fc[k].under)]} />
            ))}
          </tbody>
        </table>
      </Sum>

      <Sum title="Both teams to score">
        <table className="markets">
          <tbody>
            <Row label="BTTS yes" cells={[pct(s.btts.yes)]} />
            <Row label="BTTS no" cells={[pct(s.btts.no)]} />
            <Row label="Home keep a clean sheet" cells={[pct(s.clean_sheet.home)]} />
            <Row label="Away keep a clean sheet" cells={[pct(s.clean_sheet.away)]} />
          </tbody>
        </table>
      </Sum>

      <Sum title="Double chance & win / draw / lose without draw">
        <table className="markets">
          <tbody>
            <Row label="1X" cells={[pct(s.double_chance["1X"])]} />
            <Row label="12" cells={[pct(s.double_chance["12"])]} />
            <Row label="X2" cells={[pct(s.double_chance.X2)]} />
            <Row label="Home no draw" cells={[pct(s.draw_no_bet.H)]} />
            <Row label="Away no draw" cells={[pct(s.draw_no_bet.A)]} />
          </tbody>
        </table>
      </Sum>

      <Sum title="Team totals">
        <table className="markets">
          <tbody>
            {["0.5", "1.5", "2.5", "3.5"].map((k) => (
              <Row key={k} label={"Home over " + k}
                cells={[pct((s.team_totals.home[k] || {}).over || 0)]} />
            ))}
            {["0.5", "1.5", "2.5", "3.5"].map((k) => (
              <Row key={k} label={"Away over " + k}
                cells={[pct((s.team_totals.away[k] || {}).over || 0)]} />
            ))}
          </tbody>
        </table>
      </Sum>

      <Sum title="Winning margin & hand results">
        <table className="markets">
          <tbody>
            {Object.entries(s.winning_margin || {}).map(([k, v]) => (
              <Row key={k} label={marginLbl(k)} cells={[pct(v)]} />
            ))}
            <Row label="Home win to nil" cells={[pct(s.win_to_nil.home || 0)]} />
            <Row label="Away win to nil" cells={[pct(s.win_to_nil.away || 0)]} />
          </tbody>
        </table>
      </Sum>

      <Sum title="Asian & European handicap">
        <table className="markets">
          <tbody>
            {Object.entries(s.asian_handicap || {}).map(([k, v]) => (
              <Row key={k} label={"AH " + Number(k).toFixed(2)}
                cells={[pct(v.home), pct(v.away)]} />
            ))}
            {Object.entries(s.european_handicap || {}).map(([k, v]) => (
              <Row key={k} label={"EH " + k}
                cells={[pct(v.home), pct(v.draw), pct(v.away)]} />
            ))}
          </tbody>
        </table>
      </Sum>

      <Sum title="Half time">
        <table className="markets">
          <tbody>
            <Row label="HT 1" cells={[pct(s.ht_result.H)]} />
            <Row label="HT X" cells={[pct(s.ht_result.D)]} />
            <Row label="HT 2" cells={[pct(s.ht_result.A)]} />
            <Row label="HT over 0.5 / under" cells={[pct(s.ht_totals["0.5"].over), pct(s.ht_totals["0.5"].under)]} />
            <Row label="HT over 1.5 / under" cells={[pct(s.ht_totals["1.5"].over), pct(s.ht_totals["1.5"].under)]} />
            <Row label="HT over 2.5 / under" cells={[pct(s.ht_totals["2.5"].over), pct(s.ht_totals["2.5"].under)]} />
            <Row label="Most goals in" cells={[s.half_most_goals]} />
          </tbody>
        </table>
      </Sum>

      {s.market_used && (
        <p className="note">Forecast blended {Math.round(s.market_weight * 100)}% with the closing price — so "1/X/2" is closer to a fair price than a raw model read.</p>
      )}

      <div className="blind">
        <h3>What the model can't see</h3>
        <ul>
          <li>Team news, injuries and suspensions after the fixture file was written.</li>
          <li>Motivation, scheduling congestion and cup rotation.</li>
          <li>Pitch conditions, weather and travel.</li>
          <li>Refereeing tendencies and VAR luck on the day.</li>
        </ul>
        <p className="note" style={{ margin: "8px 0 0" }}>
          {tier === "FULL" ? null : <>Data tier: <b>{tier}</b> — {tierNote}. </>}
          Ratings come from {s.home_played} games for {s.home} and {s.away_played} for {s.away} across every watched league.
        </p>
      </div>

      <p className="note" style={{ marginTop: 24, color: "var(--ink3)" }}>
        18+ only · Gambling can be addictive. Play responsibly. A model judgement, not a tip — the odds carry a margin and the long-run edge is theirs.
      </p>
    </div>
  );
}

export default function App() {
  const hash = useHash();
  return hash.startsWith("#/card") ? <Card /> : <Slate />;
}