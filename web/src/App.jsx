import React, { useEffect, useState } from "react";
import { api, apiBase, setApiBase, isNative, nativePlatform } from "./api.js";

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
  const [state, setState] = useState({ loading: !!url, data: null, error: null });
  useEffect(() => {
    let alive = true;
    // A null url means "not yet" — used by sections that load on demand.
    if (!url) {
      setState({ loading: false, data: null, error: null });
      return;
    }
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
      fetch(api("/api/live"))
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
  // A stale snapshot is worse than none. The server stops polling once no
  // match is in play, so the last reading of a finished game would otherwise
  // sit on the card as "88'" indefinitely.
  if (!live || !live.enabled || live.stale) return null;
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

/** The slate keys 1X2 as "1"/"X"/"2"; a card keys it "H"/"D"/"A". Accept both:
 *  reading the wrong one silently yields NaN, and every card then reported
 *  "close to a coin toss" no matter how lopsided the fixture actually was. */
function as1X2(res) {
  if (!res) return { "1": NaN, X: NaN, "2": NaN };
  if (res["1"] !== undefined) return res;
  return { "1": res.H, X: res.D, "2": res.A };
}

function confLevel(res) {
  const r = as1X2(res);
  const p = Math.max(r["1"], r.X, r["2"]);
  if (!(p > 0)) return "Not rated";
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
  const { loading, data, error } = useFetch(api("/api/slate?days=" + days));

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
      <TopNav hash={window.location.hash} />

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
            <span className="n">{data.pairings.reduce((n, g) => n + g.count, 0)}</span>
            <div className="rule" />
          </div>
          <p className="note">These pairings have no published kick-off in the loaded schedule. They appear nowhere under "Today's fixtures"; treat them as ratings of a matchup, not a fixture. Each league is priced only when you open it.</p>
          {data.pairings.map((g) => <PairingGroup key={g.code} g={g} />)}
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

/** One league's unplayed pairings, priced only when the reader opens it.
 *
 * Shipping all of them with the slate meant ~2,000 model fits and a
 * 42-second, 324KB response for a screen whose real content is a few
 * dozen fixtures. */
function PairingGroup({ g }) {
  const [open, setOpen] = useState(false);
  const { loading, data, error } = useFetch(
    open ? api("/api/pairings?div=" + encodeURIComponent(g.code) + "&limit=60") : null);
  return (
    <div className="pairgroup">
      <div className="shead">
        <h2>
          <button className="disclose" aria-expanded={open}
            onClick={() => setOpen(!open)}>
            {open ? "−" : "+"} {g.league}
          </button>
        </h2>
        <span className="n">{g.count}</span>
        <div className="rule" />
      </div>
      {open && loading && <p className="note">Rating {g.count} pairings…</p>}
      {open && error && <p className="note">Couldn't rate these: {error}</p>}
      {open && data && (
        <>
          <table>
            <thead>
              <tr><th>Match</th><th className="num">1</th><th className="num">X</th><th className="num">2</th></tr>
            </thead>
            <tbody>
              {data.matches.map((m, mi) => (
                <tr key={mi}>
                  <td><span className="hm">{m.home}</span> <span className="vs">v</span> <span className="aw">{m.away}</span></td>
                  <td className={"num" + (m.pick === "1" ? " win" : "")}>{pct(m.p["1"])}</td>
                  <td className={"num" + (m.pick === "X" ? " win" : "")}>{pct(m.p.X)}</td>
                  <td className={"num" + (m.pick === "2" ? " win" : "")}>{pct(m.p["2"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.total > data.shown && (
            <p className="note">Showing {data.shown} of {data.total}.</p>
          )}
        </>
      )}
    </div>
  );
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
  const url = api("/api/card?home=" + encodeURIComponent(home) + "&away=" + encodeURIComponent(away) +
    "&div=" + encodeURIComponent(div) + (neutral ? "&neutral=1" : ""));
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
  const r123 = as1X2(s.result);
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
          <div key={k} className={"seg" + (s.pick === k ? " pick" : "")}
            style={{ flex: r123[k] }}>
            <i>{letter}</i>
            <b>{pct(r123[k])}</b>
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

      {/* Half-time markets need half-time scores in the source data, and not
          every competition publishes them - the Tanzanian league publishes
          none. Say so rather than render a block of blanks, and never read the
          fields unguarded: doing that took the whole card down to the error
          boundary the first time a TZ1 fixture was opened. */}
      {s.ht_result ? (
        <Sum title="Half time">
          <table className="markets">
            <tbody>
              <Row label="HT 1" cells={[pct(s.ht_result.H)]} />
              <Row label="HT X" cells={[pct(s.ht_result.D)]} />
              <Row label="HT 2" cells={[pct(s.ht_result.A)]} />
              {["0.5", "1.5", "2.5"].map((ln) => (
                s.ht_totals && s.ht_totals[ln] ? (
                  <Row key={ln} label={"HT over " + ln + " / under"}
                    cells={[pct(s.ht_totals[ln].over), pct(s.ht_totals[ln].under)]} />
                ) : null
              ))}
              {s.half_most_goals && <>
                <Row label="More goals in the first half"
                  cells={[pct(s.half_most_goals.first)]} />
                <Row label="More goals in the second half"
                  cells={[pct(s.half_most_goals.second)]} />
                <Row label="Same number in each half"
                  cells={[pct(s.half_most_goals.equal)]} />
              </>}
            </tbody>
          </table>
        </Sum>
      ) : (
        <p className="note">
          No half-time markets here: {lg || "this competition"} does not publish
          half-time scores, so there is nothing to fit them on.
        </p>
      )}

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

function TopNav({ hash }) {
  const cur = hash.split("?")[0];
  const items = [
    ["#/", "Today"],
    ["#/market", "Model vs market"],
    ["#/club", "Club"],
    ["#/record", "Record"],
    ["#/fixtures", "My fixtures", "soon"],
  ];
  return (
    <nav className="tabs">
      {items.map(([href, label, soon]) => (
        <a key={href} href={href} className={soon || ""}
          aria-current={cur === href ? "page" : undefined}
          title={soon ? "Coming next" : undefined}>
          {label}
        </a>
      ))}
    </nav>
  );
}

function MarketScreen() {
  const live = useLive();
  const { loading, data, error } = useFetch(api("/api/slate?days=3"));
  if (loading) return <p className="note">Loading the slate…</p>;
  if (error) return <p className="note">Can't reach the service: {error}</p>;

  const rows = [];
  for (const g of data.groups)
    for (const m of g.matches)
      if (m.market) {
        const gap = m.p[m.pick] - m.market[m.pick];
        rows.push({ ...m, gap });
      }
  rows.sort((a, b) => b.gap - a.gap);

  const ahead = rows.filter((r) => r.gap >= 0.05).length;
  const priced = rows.filter((r) => r.market).length;

  const implyPick = (mk) => (mk["1"] > mk.X && mk["1"] > mk["2"] ? "1" : mk.X > mk["2"] ? "X" : "2");

  return (
    <div>
      <header className="masthead">
        <div>
          <h1>Model vs <b>market</b></h1>
          <p className="sub">Where the model agrees with the closing price — and where it doesn't</p>
        </div>
        <div className="tools"><Theme /></div>
      </header>
      <TopNav hash={window.location.hash} />

      <p className="sumline">
        <b>{priced}</b> fixtures with a closing price · on <b>{ahead}</b> the model
        rates its own pick at least 5 points higher than the market does.
      </p>
      <p className="note" style={{ marginTop: 8 }}>
        The market column is the closing price with the bookmaker's margin
        removed (de-vigged by Shin's method), so it is a fair probability, not
        a price you could take. A gap is <b>not</b> a value signal: measured over
        5,948 walk-forward matches, the wider the disagreement the more often
        the model was the one in the wrong — at gaps above 15 points its pick
        won 24.6% of the time against the market's 44.7%. Read this page as a
        list of fixtures the model may be misjudging.
      </p>

      {rows.length > 0 && (
        <section>
          <div className="shead">
            <h2>Gap on the model's pick</h2>
            <div className="rule" />
          </div>
          <table>
            <thead>
              <tr>
                <th>Match</th>
                <th className="num">Model&nbsp;1/X/2</th>
                <th className="num">Price&nbsp;1/X/2</th>
                <th className="num">Model&nbsp;pick</th>
                <th className="num">Gap&nbsp;pp</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m, i) => (
                <tr key={i}>
                  <td>
                    <a className="match" href={"#/card?home=" + encodeURIComponent(m.home) + "&away=" + encodeURIComponent(m.away) + "&div=" + m.div + "&ko=" + encodeURIComponent(m.kickoff_label) + "&lg=" + encodeURIComponent(m.league)}>
                      <span className="hm">{m.home}</span><span className="vs">v</span><span className="aw">{m.away}</span>
                      {m.new && <em className="new">new</em>}
                      <LiveChip m={m} live={live} />
                    </a>
                  </td>
                  <td className="num hda-mini">
                    <span>{pct(m.p["1"])}</span><span>{pct(m.p.X)}</span><span>{pct(m.p["2"])}</span>
                  </td>
                  <td className="num hda-mini">
                    <span>{pct(m.market["1"])}</span><span>{pct(m.market.X)}</span><span>{pct(m.market["2"])}</span>
                  </td>
                  <td className={"num" + (m.market[implyPick(m.market)] === m.market[m.pick] ? " win" : "")}>{m.pick}</td>
                  <td className="num">
                    <span className={"gap " + (m.gap >= 0.05 ? "pos" : "neg")}>
                      {m.gap >= 0 ? "+" : ""}{(m.gap * 100).toFixed(0)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <p className="note" style={{ marginTop: 36, color: "var(--ink3)" }}>
        18+ only · Gambling can be addictive. Play responsibly. No edge is claimed over the bookmaker here.
      </p>
    </div>
  );
}

function ClubScreen() {
  const lg = useFetch(api("/api/leagues"));
  const [div, setDiv] = useState("E0");
  const { loading, data, error } = useFetch(api("/api/clubs?div=" + div));
  const codes = lg.data ? [...new Set(lg.data.map((d) => d.code))].sort() : [];
  if (lg.loading) return <p className="note">Loading leagues…</p>;
  return (
    <div>
      <header className="masthead">
        <div>
          <h1>Club <b>ratings</b></h1>
          <p className="sub">Where each side stands in its division, on the model's numbers</p>
        </div>
        <div className="tools"><Theme /></div>
      </header>
      <TopNav hash={window.location.hash} />

      <div className="tools" style={{ marginTop: 10 }}>
        <select className="sel" value={div} onChange={(e) => setDiv(e.target.value)}>
          {codes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {data && (
        <section>
          <div className="shead">
            <h2>{data.note ? "Not fitted yet" : data.league + " · strength"}</h2>
            <div className="rule" />
          </div>
          {data.note && <p className="note">{data.note}</p>}
          {data.clubs.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>Club</th>
                  <th className="num">GF&nbsp;vs&nbsp;avg</th>
                  <th className="num">GA&nbsp;vs&nbsp;avg</th>
                  <th className="num">Goals/game</th>
                  <th className="num">Str</th>
                  <th className="num">Played</th>
                </tr>
              </thead>
              <tbody>
                {data.clubs.map((c, i) => (
                  <tr key={c.team}>
                    <td className="dom">{i + 1}</td>
                    <td>
                      <span className="hm" style={{ fontSize: 15 }}>{c.team}</span>
                      {c.source !== data.code && <em className="new"> {c.source === "prior" ? "prior" : "bridged"}</em>}
                    </td>
                    <td className="num">{c.gf_home.toFixed(2)}</td>
                    <td className="num">{c.ga.toFixed(2)}</td>
                    <td className="num">{c.goals.toFixed(2)}</td>
                    <td className={"num" + (c.str >= 0 ? " win" : "")}>{(c.str >= 0 ? "+" : "") + c.str}</td>
                    <td className="num">{c.played}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="note" style={{ marginTop: 12 }}>
            GF/GA are expected goals for and against a league-average opponent, from the same
            ratings that drive every forecast. "Str" is logged attack minus defensive solidity —
            useful for comparing sides, not a prediction of any single game.
          </p>
        </section>
      )}
    </div>
  );
}

/** A route that is advertised but not built yet.
 *
 * These two used to fall through to the fixtures list, so tapping them looked
 * like a broken link. Saying plainly what is missing is better than pretending
 * the tap did nothing. */
function NotBuiltYet({ title, sub, children }) {
  return (
    <div>
      <header className="masthead">
        <div>
          <h1>{title}</h1>
          <p className="sub">{sub}</p>
        </div>
        <div className="tools"><Theme /></div>
      </header>
      <TopNav hash={window.location.hash} />
      <section className="soonbox">{children}</section>
      <p><a className="back" href="#/">← Today's fixtures</a></p>
    </div>
  );
}

const dp = (v, n = 4) => (v === null || v === undefined || Number.isNaN(v)
  ? "—" : Number(v).toFixed(n));

function RecordScreen() {
  const { loading, data, error } = useFetch(api("/api/record"));
  const empty = data && !data.published;

  return (
    <div>
      <header className="masthead">
        <div>
          <h1>Track <b>record</b></h1>
          <p className="sub">Every prediction, published before kick-off, never edited after</p>
        </div>
        <div className="tools"><Theme /></div>
      </header>
      <TopNav hash={window.location.hash} />

      {loading && <p className="note">Settling the record…</p>}
      {error && <p className="note">Record failed to load: {error}</p>}

      {empty && (
        <section className="soonbox">
          <p className="note">
            Nothing published yet. The record only counts predictions written
            down before kick-off, so it starts filling from the next time the
            slate is frozen — there is no way to backfill it, which is the
            point.
          </p>
        </section>
      )}

      {data && data.published > 0 && (
        <>
          <p className="note">
            <b>{data.published}</b> predictions published, <b>{data.settled}</b> settled,{" "}
            <b>{data.pending}</b> still to play. First published{" "}
            {String(data.first_published || "").slice(0, 10)}.
          </p>

          {/* The integrity claim, checkable rather than asserted. Every row is
              hashed against the one before it, so an edit anywhere shows up. */}
          <p className={data.chain && data.chain.ok ? "chainok" : "chainbad"}>
            {data.chain && data.chain.ok
              ? "✓ Hash chain intact across all " + data.chain.rows +
                " rows — nothing has been edited, reordered or removed since publication."
              : "⚠ " + (data.chain ? data.chain.note : "chain could not be checked")}
          </p>

          {data.settled === 0 ? (
            <section className="soonbox">
              <p className="note">
                Nothing has finished yet. Accuracy appears here as results land;
                until then there is only the list of what was called in advance.
              </p>
            </section>
          ) : (
            <>
              <Sum title="Accuracy so far">
                <table className="markets">
                  <tbody>
                    <Row label="Matches settled" cells={[String(data.overall.n)]} />
                    <Row label="Log-loss (1X2)" cells={[dp(data.overall.logloss_1x2)]} />
                    <Row label="RPS" cells={[dp(data.overall.rps)]} />
                    <Row label="Favourite came in" cells={[pct(data.overall.acc)]} />
                    {data.market && data.market.n_with_price && <>
                      <Row label={"Closing price, log-loss (" + data.market.n_with_price + " rows)"}
                        cells={[dp(data.market.logloss_market)]} />
                      <Row label="Model on those same rows"
                        cells={[dp(data.market.logloss_model_same_rows)]} />
                    </>}
                  </tbody>
                </table>
                <p className="note">
                  Lower log-loss is better. The model is measured against the
                  price on exactly the rows that carry a price — scoring the
                  model on everything and the price on its own subset is the
                  oldest way to flatter a model.
                </p>
              </Sum>

              {data.calibration.length > 0 && (
                <Sum title="Calibration — did 70% mean 70%?">
                  <table className="markets">
                    <thead><tr><th>Band</th><th>N</th><th>Said</th><th>Happened</th></tr></thead>
                    <tbody>
                      {data.calibration.map((b) => (
                        <tr key={b.bin}>
                          <td>{b.bin}</td><td>{b.n}</td>
                          <td>{pct(b.predicted)}</td><td>{pct(b.realised)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="note">
                    On the favourite in each match. The two right-hand columns
                    should track each other; where they don't, the model is
                    over- or under-confident in that band.
                  </p>
                </Sum>
              )}

              {data.by_league.length > 0 && (
                <Sum title="By competition">
                  <table className="markets">
                    <thead><tr><th>League</th><th>N</th><th>Log-loss</th><th>Hit</th></tr></thead>
                    <tbody>
                      {data.by_league.map((r) => (
                        <tr key={r.div}>
                          <td>{r.league}</td><td>{r.n}</td>
                          <td>{dp(r.logloss)}</td><td>{pct(r.acc)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Sum>
              )}

              <Sum title="The last 40 settled">
                <table className="markets">
                  <thead>
                    <tr><th>Kick-off</th><th>Match</th><th>Called</th><th>Result</th></tr>
                  </thead>
                  <tbody>
                    {data.recent.map((m, i) => (
                      <tr key={i}>
                        <td>{String(m.kickoff).slice(0, 16).replace("T", " ")}</td>
                        <td>{m.home} v {m.away}</td>
                        <td>{m.pick} @ {pct(m.p[m.pick])}{m.predicted_score ? " · " + m.predicted_score : ""}</td>
                        {/* The two calls are scored separately on purpose. A
                            match can land the exact scoreline while the 1X2
                            pick misses — the most likely single score is not
                            the most likely outcome — and one tick covering
                            both would read as a contradiction. */}
                        <td>
                          {m.score}{" "}
                          <span className={m.correct ? "hit" : "miss"}>
                            1X2 {m.correct ? "✓" : "✗"}
                          </span>
                          {m.exact && <span className="hit"> · score ✓</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Sum>
            </>
          )}
        </>
      )}

      <p className="note" style={{ marginTop: 20 }}>
        For context, on historical data: over 5,948 walk-forward matches the
        engine scores 0.9739 log-loss alone and 0.9568 blended with the closing
        price, against the price's own 0.9558. It matches the market; it does
        not beat it.
      </p>
      <p><a className="back" href="#/">← Today's fixtures</a></p>
    </div>
  );
}

function MyFixturesScreen() {
  return (
    <NotBuiltYet title={<>My <b>fixtures</b></>}
      sub="Upload your own list and have it rated — not built yet">
      <p className="note">
        This page will take a pasted list, a CSV or a spreadsheet and rate every
        fixture in it with the same engine that drives the cards.
      </p>
      <p className="note">
        The part that has to be right before it ships is name matching. A loose
        match silently rates a <em>different club</em> and looks confident doing
        it — the engine already refuses fuzzy matching in bulk, and this screen
        will show you exactly what each name was matched to, block anything
        ambiguous until you choose, and list anything it could not find rather
        than dropping it quietly.
      </p>
      <p className="note">
        In the meantime, any two clubs can be rated from the <a href="#/club">Club</a> page.
      </p>
    </NotBuiltYet>
  );
}

function SettingsScreen() {
  const [value, setValue] = useState(apiBase());
  const [saved, setSaved] = useState(false);
  const [probe, setProbe] = useState(null);

  async function check() {
    setProbe("checking");
    try {
      const r = await fetch((value.replace(/\/+$/, "") || "") + "/api/leagues");
      setProbe(r.ok ? "ok" : "HTTP " + r.status);
    } catch (e) {
      setProbe(e.message || "unreachable");
    }
  }

  return (
    <div>
      <header className="masthead">
        <div>
          <h1><b>Settings</b></h1>
          <p className="sub">Where this app looks for the prediction service</p>
        </div>
        <div className="tools"><Theme /></div>
      </header>
      <TopNav hash={window.location.hash} />

      <section className="soonbox">
        <p className="note">
          The phone app and the service are separate things: the app is on the
          handset, the model runs on a server. If this is wrong or empty, every
          screen will sit empty however good the model is.
        </p>
        <label className="fieldlabel" htmlFor="apibase">Service address</label>
        <input id="apibase" className="field" value={value} spellCheck="false"
          autoCapitalize="off" autoCorrect="off" inputMode="url"
          placeholder="https://api.example.com"
          onChange={(e) => { setValue(e.target.value); setSaved(false); setProbe(null); }} />
        <div className="btnrow">
          <button className="btn" onClick={check}>Test</button>
          <button className="btn primary" onClick={() => setSaved(setApiBase(value))}>
            Save
          </button>
        </div>
        {probe === "checking" && <p className="note">Testing…</p>}
        {probe === "ok" && <p className="chainok">✓ The service answered.</p>}
        {probe && probe !== "ok" && probe !== "checking" &&
          <p className="chainbad">⚠ No answer: {probe}</p>}
        {saved && <p className="note">Saved. Pull any screen again to reload.</p>}
        <p className="note">
          It must be <b>https</b> on a phone — Android blocks plain http by
          default, and the request will fail with no visible error. Leave it
          empty in a browser, where the app and the service share an origin.
        </p>
        <p className="note">
          Running as: <b>{isNative ? nativePlatform() : "web"}</b>
          {apiBase() ? <> · currently <code>{apiBase()}</code></> : <> · using this page's origin</>}
        </p>
      </section>
      <p><a className="back" href="#/">← Today's fixtures</a></p>
    </div>
  );
}

/** Android's hardware back button.
 *
 * Without this the button closes the app from any screen, which feels broken:
 * on a phone, back means "up one screen" and only exits from the top. A hash
 * route deeper than the root goes back in history; the root lets the system
 * do what it would anyway (minimise). */
function useHardwareBack() {
  useEffect(() => {
    if (!isNative || !window.Capacitor || !window.Capacitor.Plugins ||
        !window.Capacitor.Plugins.App) return;
    const app = window.Capacitor.Plugins.App;
    const handle = app.addListener("backButton", ({ canGoBack }) => {
      const atRoot = !window.location.hash || window.location.hash === "#/";
      if (atRoot || !canGoBack) app.exitApp();
      else window.history.back();
    });
    return () => { Promise.resolve(handle).then((h) => h && h.remove()); };
  }, []);
}

export default function App() {
  const hash = useHash();
  useHardwareBack();
  useEffect(() => {
    // Lets the stylesheet add status-bar and gesture-bar padding only where
    // there is a status bar to avoid.
    if (isNative) document.documentElement.classList.add("native");
    // A phone build with no service address is not broken, it is unconfigured
    // - but it looks identical: every screen simply empty. Send the first
    // launch to Settings instead of to a blank list of fixtures.
    if (isNative && !apiBase() && !window.location.hash.startsWith("#/settings")) {
      window.location.hash = "#/settings";
    }
  }, []);
  if (hash.startsWith("#/market")) return <MarketScreen />;
  if (hash.startsWith("#/club")) return <ClubScreen />;
  if (hash.startsWith("#/card")) return <Card />;
  if (hash.startsWith("#/record")) return <RecordScreen />;
  if (hash.startsWith("#/settings")) return <SettingsScreen />;
  if (hash.startsWith("#/fixtures")) return <MyFixturesScreen />;
  return <Slate />;
}