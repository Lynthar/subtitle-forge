"""Which prompt the CLI reports, exports and reverts to.

Locks in four fixes around the two-field prompt state (`prompt_template` for a
custom string, `prompt_template_id` for a library template):
  * `config reset-prompt` used to clear only the custom field, so after
    `config use-prompt <id>` it printed "Already using default prompt" and
    changed nothing — there was no way back to the default at all;
  * `config show-prompt` / `config export-prompt` read the custom field
    directly and so reported the default while a library template was live;
  * `process --prompt-template <id>` accepted an unknown id, which turns JSON
    mode off yet resolves to the default prompt — the worst parse path — and
    only after transcription had already run.

No Ollama or Whisper needed: every case stops in argument handling or config I/O.
"""

import pytest
from typer.testing import CliRunner

from subtitle_forge.cli.app import app
from subtitle_forge.core.prompt_library import get_prompt_library
from subtitle_forge.models.config import AppConfig

runner = CliRunner()

LIBRARY_ID = "movie-scifi"


def _run(cfg_path, *args):
    return runner.invoke(app, ["--config", str(cfg_path), *args])


def _select_library_template(cfg_path):
    result = _run(cfg_path, "config", "use-prompt", LIBRARY_ID)
    assert result.exit_code == 0, result.output
    assert AppConfig.load(cfg_path).ollama.prompt_template_id == LIBRARY_ID


def test_reset_prompt_clears_a_library_selection(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)

    result = _run(cfg_path, "config", "reset-prompt")

    assert result.exit_code == 0, result.output
    assert "Already using default" not in result.output
    saved = AppConfig.load(cfg_path)
    assert saved.ollama.prompt_template is None
    assert saved.ollama.prompt_template_id is None


def test_reset_prompt_reports_no_change_when_already_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    AppConfig().save(cfg_path)

    result = _run(cfg_path, "config", "reset-prompt")

    assert result.exit_code == 0, result.output
    assert "Already using default" in result.output


def test_show_prompt_shows_the_selected_library_template(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)

    result = _run(cfg_path, "config", "show-prompt")

    assert result.exit_code == 0, result.output
    assert "Using default prompt" not in result.output
    assert LIBRARY_ID in result.output


def test_export_prompt_writes_the_selected_library_template(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _select_library_template(cfg_path)
    out = tmp_path / "prompt.txt"

    result = _run(cfg_path, "config", "export-prompt", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == get_prompt_library().get_template(LIBRARY_ID).template


def test_export_prompt_writes_the_default_when_nothing_is_selected(tmp_path):
    from subtitle_forge.core.translator import SubtitleTranslator

    cfg_path = tmp_path / "config.yaml"
    AppConfig().save(cfg_path)
    out = tmp_path / "prompt.txt"

    result = _run(cfg_path, "config", "export-prompt", "-o", str(out))

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == SubtitleTranslator.DEFAULT_PROMPT_TEMPLATE


def test_process_rejects_an_unknown_prompt_template_id(tmp_path, monkeypatch):
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subtitle_forge.core.transcriber import Transcriber

    def _refuse(*args, **kwargs):
        raise AssertionError("process started work despite an unknown prompt id")

    # Building a Transcriber is the first thing past argument handling, and on
    # a cold cache it downloads gigabytes — guard it so a regression fails here
    # instead of hanging.
    monkeypatch.setattr(Transcriber, "__init__", _refuse)

    cfg_path = tmp_path / "config.yaml"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    result = _run(
        cfg_path, "process", str(video), "-t", "zh", "--prompt-template", "no-such-template"
    )

    assert result.exit_code == 1
    assert "no-such-template" in result.output
    assert "not found" in result.output
