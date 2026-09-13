"""The .env loader: it may fill gaps, and must never override or leak."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import envfile  # noqa: E402

NAMES = ["FP_ENV_A", "FP_ENV_B", "FP_ENV_C", "FP_ENV_D", "FP_ENV_E", "FP_ENV_F"]


@pytest.fixture(autouse=True)
def clean():
    for n in NAMES:
        os.environ.pop(n, None)
    yield
    for n in NAMES:
        os.environ.pop(n, None)


def write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_it_sets_what_is_unset_and_reports_names_only(tmp_path):
    f = write(tmp_path, "FP_ENV_A=secret-value-123\n")
    assert envfile.load(f, force=True) == ["FP_ENV_A"]
    assert os.environ["FP_ENV_A"] == "secret-value-123"


def test_the_real_environment_always_wins(tmp_path):
    """A platform secret store must never be overridden by a stray file."""
    os.environ["FP_ENV_B"] = "from-the-platform"
    f = write(tmp_path, "FP_ENV_B=from-the-file\n")
    assert envfile.load(f, force=True) == []
    assert os.environ["FP_ENV_B"] == "from-the-platform"


def test_an_empty_slot_is_not_an_empty_string(tmp_path):
    """The template's `LIVE_API_KEY=` must leave the variable unset."""
    f = write(tmp_path, "FP_ENV_C=\n")
    envfile.load(f, force=True)
    assert "FP_ENV_C" not in os.environ


def test_comments_quotes_and_export_are_understood(tmp_path):
    f = write(tmp_path, "# a comment\n\nexport FP_ENV_D=plain\n"
                        "FP_ENV_E=\"quoted value\"\nFP_ENV_F=abc # trailing note\n")
    envfile.load(f, force=True)
    assert os.environ["FP_ENV_D"] == "plain"
    assert os.environ["FP_ENV_E"] == "quoted value"
    assert os.environ["FP_ENV_F"] == "abc"


def test_it_does_nothing_under_pytest_unless_forced(tmp_path):
    """A developer's real DATABASE_URL must never reach the test suite."""
    f = write(tmp_path, "FP_ENV_A=should-not-load\n")
    assert envfile.load(f) == []
    assert "FP_ENV_A" not in os.environ


def test_a_missing_file_is_not_an_error(tmp_path):
    assert envfile.load(str(tmp_path / "nope.env"), force=True) == []


def test_the_env_file_it_reads_is_ignored_by_git():
    """The whole point: the key lives in a file that can never be committed."""
    import subprocess
    repo = envfile.REPO
    r = subprocess.run(["git", "check-ignore", "-q", envfile.path()], cwd=repo)
    if r.returncode not in (0, 1):
        pytest.skip("git not available")
    assert r.returncode == 0
