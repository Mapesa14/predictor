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
              {g.matches.map((m, mi) => {
                const q = m.pick + "-" + encodeURIComponent(m.home) + "&away=" + encodeURIComponent(m.away) + "&div=" + m.div;
                return (
                  <tr key={mi} className={m.started ? "started" : ""}>
                    <td className="kof nowrap">{m.kickoff_label}</td>
                    <td>
                      <a className="match" href={"#/card?pick=" + m.pick + "&home=" + encodeURIComponent(m.home) + "&away=" + encodeURIComponent(m.away) + "&div=" + m.div}>
                        <span className="hm">{m.home}</span>
                        <span className="vs">v</span>
                        <span className="aw">{m.away}</span>
                        {m.new && <em> new</em>}
                        {m.started && <em className="live"> started</em>}
                      </a>
                    </td>
                    <td className="num">{pct(m.p["1"])}</td>
                    <td className="num">{pct(m.p.X)}</td>
                    <td className="num">{pct(m.p["2"])}</td>
                    <td className="num">{pct(m.o25)}</td>
                    <td className="num">{pct(m.btts)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ))}
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