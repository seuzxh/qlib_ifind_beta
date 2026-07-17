from qlib_ifind_beta import ifind


def test_load_refresh_token_prefers_environment(monkeypatch):
    monkeypatch.setenv("IFIND_REFRESH_TOKEN", "from-environment")
    monkeypatch.setattr(ifind, "_REFRESH_TOKEN_PATHS", ())

    assert ifind.load_refresh_token() == "from-environment"


def test_load_refresh_token_falls_back_to_existing_source(tmp_path, monkeypatch):
    source = tmp_path / "credential_source.py"
    source.write_text('    IFIND_REFRESH_TOKEN = "from-source"\n')
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    monkeypatch.setattr(ifind, "_REFRESH_TOKEN_PATHS", (source,))

    assert ifind.load_refresh_token() == "from-source"
