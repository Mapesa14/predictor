"""No test may touch the network.

This is not hygiene, it is damage control. `_refresh_cycle` calls every source
in turn, and two tests run the whole cycle: they fetched BBC Sport for real and
overwrote `data/manual/fixtures_bbc.csv` in the working copy with whatever the
live page happened to hold - twice, silently, after the file had just been
refreshed by hand.

Every source here takes an injectable transport for its tests. Anything that
reaches past one lands on this guard and fails loudly, which is what a test
doing real I/O deserves.
"""
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NetworkUsedInTest(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Block the two ways this codebase actually reaches the internet.

    Narrow on purpose: the sources' own functions stay callable, because the
    tests drive them with injected pages and transports. Only the real fetch
    underneath is cut off.
    """
    def blocked(*a, **k):
        raise NetworkUsedInTest(
            "a test tried to use the network: pass a transport or a page "
            "instead (see service/live.py, service/bbc_fixtures.py)")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", blocked, raising=False)
    try:
        import service.live as live
        monkeypatch.setattr(live, "http_get", blocked, raising=False)
    except ImportError:
        pass
