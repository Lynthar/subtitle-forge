"""Tests for config save/load round-tripping.

Locks in the fix for the data-loss bug where save() used a hand-maintained
whitelist and silently dropped fields (ollama.request_timeout / max_retries,
whisper.batch_size / download_root) on rewrite.
"""

from subtitle_forge.models.config import AppConfig


def test_save_load_preserves_all_fields(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = AppConfig()
    cfg.ollama.request_timeout = 600.0
    cfg.ollama.max_retries = 5
    cfg.whisper.batch_size = 8
    cfg.whisper.download_root = "/models"
    cfg.timestamp.linger_ms = 250
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.ollama.request_timeout == 600.0
    assert loaded.ollama.max_retries == 5
    assert loaded.whisper.batch_size == 8
    assert loaded.whisper.download_root == "/models"
    assert loaded.timestamp.linger_ms == 250


def test_hand_edited_field_survives_a_config_set_rewrite(tmp_path):
    # Simulate: user writes request_timeout by hand, then a later save() (as
    # `config set` performs) must not erase it.
    path = tmp_path / "config.yaml"
    cfg = AppConfig()
    cfg.ollama.request_timeout = 600.0
    cfg.save(path)

    reopened = AppConfig.load(path)
    reopened.whisper.model = "large-v2"  # unrelated change
    reopened.save(path)

    assert AppConfig.load(path).ollama.request_timeout == 600.0


def test_defaults_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    AppConfig().save(path)
    loaded = AppConfig.load(path)
    assert loaded.whisper.model == AppConfig().whisper.model
    assert loaded.timestamp.mode == "minimal"


def test_default_yaml_matches_code_defaults():
    # config/default.yaml is the documented reference copy of the defaults —
    # it is NOT loaded at runtime — so this test is what keeps it from
    # silently drifting away from the dataclass defaults it documents.
    import dataclasses
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config" / "default.yaml", encoding="utf-8") as f:
        documented = yaml.safe_load(f)
    code_defaults = dataclasses.asdict(AppConfig())

    def check(doc: dict, code: dict, prefix: str = "") -> None:
        for key, value in doc.items():
            assert key in code, f"{prefix}{key} is documented but not an AppConfig field"
            if isinstance(value, dict):
                check(value, code[key], prefix=f"{prefix}{key}.")
            else:
                assert code[key] == value, (
                    f"{prefix}{key}: default.yaml documents {value!r}, "
                    f"code default is {code[key]!r}"
                )

    check(documented, code_defaults)


def test_config_path_uses_appdata_on_windows(monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(Path("/fake/appdata")))
    assert AppConfig.get_config_path() == Path("/fake/appdata") / "subtitle-forge" / "config.yaml"
