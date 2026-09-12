import React, { useCallback, useEffect, useState } from "react";

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

function Bar({ p }) {
  const total = p["1"] + p["X"] + p["2"] || 1;
  return (
    <div className="hda">
      {[["1", "H"], ["X", "D"], ["2", "A"]].map(([k]) => (
        <div key={k} className={"hda-seg hda-" + k} style={{ width: pct(p[k] / total) }}>
          <span>{k}</span>
          <b>{pct(p[k])}</b>
        </div>
      ))}
    </div>
  );
}

function Slate() {
  const { loading, data, error } = useFetch("/api/slate?days=2");
  const live = useLive();
  if (loading) return <p className="muted pad">Loading the slate…</p>;
  if (error) return <p className="muted pad">Can't reach the service: {error}. Is it running on :8000?</p>;

  let total = 0;
  for (const g of data.groups) total += g.matches.length;
  return (
    <div>
      <header className="top">
        <div>
          <h1>Football Predictor</h1>
          <p className="muted">Generated {new Date(data.generated).toLocaleString()} · {total} fixtures · model vs price where priced</p>
          {live && !live.enabled && (
            <p className="muted note">Live scores off — set <code>LIVE_API_KEY</code> on the service.</p>
          )}
        </div>
      </header>

      {data.skipped.length > 0 && (
        <p className="muted pad">Skipped {data.skipped.length}: {data.skipped.join("; ")}</p>
      )}

      {data.groups.map((g, gi) => (
        <section key={g.code}>
          <h2>
            <span className="badge">{g.country}</span> {g.league}
          </h2>
          <table>
            <thead>
              <tr>
                <th className="kof">Kick-off</th>
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
                <tr key={mi} className={m.started ? "started" : ""}>
                  <td className="kof nowrap">{m.kickoff_label}</td>
                  <td>
                    <a className="match" href={"#/card?pick=" + m.pick + "&home=" + encodeURIComponent(m.home) + "&away=" + encodeURIComponent(m.away) + "&div=" + m.div}>
                      <span className="hm">{m.home}</span>
                      <span className="vs">v</span>
                      <span className="aw">{m.away}</span>
                      {m.new && <em> new</em>}
                      {m.started && <em className="live"> started</em>}
                      <LiveChip m={m} live={live} />
                    </a>
                  </td>
                  <td className="num">{pct(m.p["1"])}</td>
                  <td className="num">{pct(m.p.X)}</td>
                  <td className="num">{pct(m.p["2"])}</td>
                  <td className="num">{pct(m.o25)}</td>
                  <td className="num">{pct(m.btts)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}

      {data.pairings.length > 0 && (
        <section className="pairings">
          <h2>Head-to-head ratings — not scheduled fixtures</h2>
          <p className="muted">These pairings have no published kick-off in the loaded schedule; treat them as ratings of a matchup, not fixtures.</p>
          {data.pairings.map((g) => (
            <div key={g.code}>
              <h3>{g.league}</h3>
              <table>
                <thead>
                  <tr><th>Match</th><th className="num">1</th><th className="num">X</th><th className="num">2</th></tr>
                </thead>
                <tbody>
                  {g.matches.map((m, mi) => (
                    <tr key={mi}>
                      <td className="nowrap"><span className="hm">{m.home}</span> <span className="vs">v</span> <span className="aw">{m.away}</span></td>
                      <td className="num">{pct(m.p["1"])}</td>
                      <td className="num">{pct(m.p.X)}</td>
                      <td className="num">{pct(m.p["2"])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function Card() {
  const params = new URLSearchParams(window.location.hash.split("?")[1] || "");
  const home = params.get("home") || "";
  const away = params.get("away") || "";
  const div = params.get("div") || "";
  const url = "/api/card?home=" + encodeURIComponent(home) + "&away=" + encodeURIComponent(away) + "&div=" + encodeURIComponent(div);
  const { loading, data, error } = useFetch(url);
  return (
    <div>
      <header className="top">
        <p><a className="back" href="#/">← All fixtures</a></p>
        <h1>{home} v {away}</h1>
        <p className="muted">{div || "·"} · one matrix, every market</p>
      </header>
      {loading && <p className="muted pad">Fitting the models…</p>}
      {error && <p className="muted pad">Card failed: {error}</p>}
      {data && <CardBody s={data} />}
    </div>
  );
}

function Row({ label, cells }) {
  return (
    <tr><td className="lbl">{label}</td>{cells.map((c, i) => <td key={i} className="num">{c}</td>)}</tr>
  );
}

function CardBody({ s }) {
  const cs = s.correct_scores.slice(0, 5);
  return (
    <div className="card">
      <Bar p={s.result} />

      <table className="plain">
        <tbody>
          <Row label="Over 2.5" cells={[pct(s.totals[2.5].over)]} />
          <Row label="Under 2.5" cells={[pct(s.totals[2.5].under)]} />
          <Row label="BTTS yes" cells={[pct(s.btts.yes)]} />
          <Row label="Expected goals" cells={[s.exp_home.toFixed(2) + " – " + s.exp_away.toFixed(2)]} />
          <Row label="Most likely" cells={[cs[0][0] + "-" + cs[0][1] + "  " + pct(cs[0][2])]} />
        </tbody>
      </table>

      {s.market_used && (
        <p className="note">Blended {Math.round(s.market_weight * 100)}% with the closing price.</p>
      )}

      <h3>Correct scores</h3>
      <table className="plain">
        <tbody>
          {cs.map(([h, a, p], i) => (
            <Row key={i} label={h + "-" + a} cells={[pct(p)]} />
          ))}
        </tbody>
      </table>

      {s.home_new || s.away_new ? (
        <p className="note">One of these clubs has no history in this division — rated from where it played before.</p>
      ) : null}
    </div>
  );
}

export default function App() {
  const hash = useHash();
  return hash.startsWith("#/card") ? <Card /> : <Slate />;
}