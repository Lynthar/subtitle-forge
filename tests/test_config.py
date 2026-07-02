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
