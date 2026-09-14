"""The deployment config. Each test guards a way the deployed app would have
failed quietly - answering 200 while being asleep, stale, or half its data."""
import os
import sys
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def read(p):
    return open(os.path.join(REPO, p), encoding="utf-8").read()


def fly():
    return tomllib.loads(read("fly.toml"))


def megabytes(s):
    s = s.strip().lower()
    if s.endswith("gb"):
        return int(float(s[:-2]) * 1024)
    if s.endswith("mb"):
        return int(float(s[:-2]))
    return int(s)


# ----------------------------------------------------------------- fly.toml
def test_the_machine_never_sleeps():
    """Asleep, nothing polls live scores, publishes the record or refreshes
    data, and every wake refits thirty divisions."""
    h = fly()["http_service"]
    assert h["auto_stop_machines"] == "off"
    assert h["min_machines_running"] >= 1


def test_it_runs_nearest_to_tanzania():
    assert fly()["primary_region"] == "jnb"


def test_everything_writable_lives_on_the_volume():
    assert fly()["mounts"]["destination"] == "/app/data"
    env = fly()["env"]
    assert env["FOOTBALL_DATA"].startswith("/app/data")
    assert env["RECORD_ROOT"] == "/app"          # record at /app/data/record


def test_the_health_check_probes_a_real_route():
    """The container check once probed a route that did not exist."""
    h = fly()["http_service"]
    assert h["internal_port"] == 8000
    check = h["checks"][0]
    assert check["path"] == "/api/health"
    assert int(check["grace_period"].rstrip("s")) >= 60


def test_enough_memory_for_the_fitted_models():
    """The warm service measured 166 MB resident; 256 MB leaves no headroom."""
    assert megabytes(fly()["vm"][0]["memory"]) >= 512


def test_the_service_maintains_itself():
    """Deployed, nobody runs update, refresh-tanzania or record-publish."""
    assert float(fly()["env"]["AUTO_REFRESH_HOURS"]) > 0
    assert "LIVE_API_KEY" not in fly()["env"]      # a secret, never config


# ------------------------------------------------------------------- image
def dockerignore():
    return [l.strip() for l in read(".dockerignore").splitlines()
            if l.strip() and not l.strip().startswith("#")]


def test_the_image_keeps_what_a_tanzanian_rebuild_needs():
    """The rebuild regenerates the league file from data/raw. Left out of the
    image, a rebuild in production would keep only the current season's
    overlay - 10 results in place of 730."""
    ignored = dockerignore()
    assert "data/raw" not in ignored and "data" not in ignored


def test_the_image_carries_the_built_web_app_and_no_secrets():
    ignored = dockerignore()
    assert "web/dist" not in ignored
    assert "!web/dist" in ignored or "web/*" not in ignored
    assert ".env" in ignored


def test_the_startup_script_keeps_unix_line_endings():
    """A carriage return after the shebang fails as '/bin/sh^M: not found'."""
    attrs = read(".gitattributes")
    assert "*.sh" in attrs and "eol=lf" in attrs
    assert b"\r\n" not in open(os.path.join(REPO, "deploy", "start.sh"), "rb").read()


def test_the_entrypoint_seeds_only_an_empty_volume():
    """After the first boot the volume is the source of truth: refreshes write
    there and the record grows there. Re-seeding would roll both back."""
    s = read("deploy/start.sh")
    assert 'if [ ! -d "$DATA/leagues" ]' in s
    assert "setpriv" in s                         # drops root before serving


# --------------------------------------------------------- one address
def test_the_web_app_and_the_api_share_one_address():
    pytest.importorskip("httpx")
    if not os.path.isfile(os.path.join(REPO, "web", "dist", "index.html")):
        pytest.skip("web app not built")
    from fastapi.testclient import TestClient
    from service import app as svc
    c = TestClient(svc.app)
    page = c.get("/")
    assert page.status_code == 200 and 'id="root"' in page.text
    api = c.get("/api/record/verify")
    assert api.status_code == 200
    assert api.headers["content-type"].startswith("application/json")
    assert c.get("/api/no-such-route").status_code == 404


# --------------------------------------------------------- refresh cycle
def test_one_failing_refresh_step_does_not_stop_the_others(monkeypatch, tmp_path):
    """A football-data outage must not stop the Tanzanian refresh, and neither
    may stop the record being published."""
    from service import app as svc
    from predictor import adapters, refresh, tanzania

    def down(*a, **k):
        raise RuntimeError("football-data is down")

    monkeypatch.setattr(svc, "ROOT", str(tmp_path))
    monkeypatch.setattr(refresh, "refresh_all", down)
    monkeypatch.setattr(tanzania, "sync",
                        lambda root, **k: {"results": 10, "fixtures": 170})
    monkeypatch.setattr(adapters, "build_csvs", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(svc, "_cached_slate", lambda days, league: {"groups": []})
    monkeypatch.setattr(svc.record, "publish",
                        lambda rows, root: {"written": 0, "total": 46})
    done = svc._refresh_cycle()
    assert done[0].startswith("football-data failed")
    assert done[1].startswith("tanzania: 10 results")
    assert done[2].startswith("record:")


def test_the_first_download_is_not_raced(monkeypatch, tmp_path):
    """While the first-boot download is still writing, the refresh must not
    write the same files."""
    from service import app as svc
    from predictor import adapters, refresh, tanzania
    called = []
    (tmp_path / ".bootstrapping").write_text("")
    monkeypatch.setattr(svc, "ROOT", str(tmp_path))
    monkeypatch.setattr(refresh, "refresh_all", lambda *a, **k: called.append(1))
    monkeypatch.setattr(tanzania, "sync",
                        lambda root, **k: {"results": 0, "fixtures": 0})
    monkeypatch.setattr(adapters, "build_csvs", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(svc, "_cached_slate", lambda days, league: {"groups": []})
    monkeypatch.setattr(svc.record, "publish",
                        lambda rows, root: {"written": 0, "total": 0})
    done = svc._refresh_cycle()
    assert called == []
    assert "skipped" in done[0]
