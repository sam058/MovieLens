"""Plot cleaning and the TMDB fetcher, tested against a fake HTTP opener (no key needed)."""

import io
import json
import urllib.error

import pytest

import fetch_plots
from text_preprocessing import clean_plot


def test_clean_plot():
    assert clean_plot("A <b>heist</b> &amp; a  chase.\n See https://example.com") == "A heist & a chase. See"
    assert clean_plot("No overview found.") == ""
    assert clean_plot(None) == ""
    assert clean_plot(float("nan")) == ""


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("url", code, "error", headers, None)


def opener_returning(*outcomes):
    calls = []

    def opener(request, timeout):
        calls.append(request)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(json.dumps(outcome).encode())

    opener.calls = calls
    return opener


def test_fetch_overview_ok_uses_bearer_header_and_english():
    opener = opener_returning({"overview": "Toys come to life."})
    status, plot = fetch_plots.fetch_overview(862, {"Authorization": "Bearer t"}, {}, opener=opener)
    assert (status, plot) == ("ok", "Toys come to life.")
    request = opener.calls[0]
    assert request.full_url == "https://api.themoviedb.org/3/movie/862?language=en-US"
    assert request.get_header("Authorization") == "Bearer t"


def test_fetch_overview_missing_and_null_overview():
    assert fetch_plots.fetch_overview(1, {}, {}, opener=opener_returning(http_error(404))) == ("not_found", "")
    assert fetch_plots.fetch_overview(1, {}, {}, opener=opener_returning({"overview": None})) == ("ok", "")


def test_fetch_overview_retries_rate_limit(monkeypatch):
    monkeypatch.setattr(fetch_plots.time, "sleep", lambda seconds: None)
    opener = opener_returning(http_error(429, "1"), {"overview": "Plot."})
    assert fetch_plots.fetch_overview(1, {}, {"api_key": "k"}, opener=opener) == ("ok", "Plot.")
    assert len(opener.calls) == 2
    assert "api_key=k" in opener.calls[1].full_url


def test_fetch_overview_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(fetch_plots.time, "sleep", lambda seconds: None)
    errors = [urllib.error.URLError("down")] * fetch_plots.MAX_RETRIES
    assert fetch_plots.fetch_overview(1, {}, {}, opener=opener_returning(*errors)) == ("error", "")


def test_bad_credentials_stop_the_run():
    with pytest.raises(SystemExit):
        fetch_plots.fetch_overview(1, {}, {}, opener=opener_returning(http_error(401)))
