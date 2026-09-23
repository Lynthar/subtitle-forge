"""_parse_translation_response's fallback chain: each strategy hands over until every index is in.

Counting parsed entries is not enough: a reply renumbered from 1 has the right count, wrong keys."""

from subtitle_forge.core.translator import SubtitleTranslator, TranslationConfig
from subtitle_forge.models.subtitle import SubtitleSegment


def _legacy_mode_translator():
    return SubtitleTranslator(
        TranslationConfig(prompt_template="{source_lang}{target_lang}{segments}")
    )


def _segments(*indices):
    return [
        SubtitleSegment(index=i, start=float(i), end=float(i) + 1.0, text=f"original {i}")
        for i in indices
    ]


def test_renumbered_reply_still_reaches_positional_fallback():
    # Segments 41-43, model answered [1]-[3]: three entries parsed, none of them
    # ours. Only the positional fallback can recover this, and it must run.
    t = _legacy_mode_translator()
    out = t._parse_translation_response("[1] bonjour\n[2] salut\n[3] merci", _segments(41, 42, 43))
    assert [s.text for s in out] == ["bonjour", "salut", "merci"]


def test_complete_reply_is_taken_as_is():
    t = _legacy_mode_translator()
    out = t._parse_translation_response("[41] bonjour\n[42] salut", _segments(41, 42))
    assert [s.text for s in out] == ["bonjour", "salut"]
