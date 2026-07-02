"""Regression tests for TimestampProcessor.

These lock in fixes for three timing bugs that shipped in the default paths:
  * minimal mode emitted overlapping subtitles (fast dialogue)
  * "off" mode truncated a segment below its acoustic end via a neighbour's lead-in
  * sentence splitting without word timestamps overflowed the parent span
    (and produced negative-duration segments)

The module is pure/functional and has no heavy dependencies, so these run
without ffmpeg / torch / whisper.
"""

from subtitle_forge.core.timestamp_processor import TimestampProcessor
from subtitle_forge.models.subtitle import SubtitleSegment, WordTiming


def _seg(index, start, end, text="x", words=True):
    w = [WordTiming(text, start, end)] if words else None
    return SubtitleSegment(index=index, start=start, end=end, text=text, words=w)


def _overlaps(segs):
    return [
        (segs[i - 1].index, segs[i].index)
        for i in range(1, len(segs))
        if segs[i].start < segs[i - 1].end - 1e-9
    ]


def test_minimal_mode_no_overlaps_in_fast_dialogue():
    # Three short utterances ~0.15s apart — min_duration extension used to push
    # each one over the next, and nothing fixed it afterwards.
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    out = TimestampProcessor(mode="minimal", min_duration=1.0).process(segs, audio_duration=60.0)
    assert _overlaps(out) == []


def test_minimal_mode_no_overlaps_without_audio_duration():
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    out = TimestampProcessor(mode="minimal", min_duration=1.0).process(segs, audio_duration=None)
    assert _overlaps(out) == []


def test_off_mode_never_truncates_below_acoustic_end():
    # Two adjacent 50ms utterances. In "off" mode the second one's lead-in must
    # not pull the first one's end earlier than where speech actually ended.
    segs = [_seg(1, 9.920, 9.970), _seg(2, 10.000, 10.050)]
    out = TimestampProcessor(mode="off", min_duration=1.0).process(segs, audio_duration=60.0)
    assert out[0].end >= 9.970 - 1e-9
    assert _overlaps(out) == []


def test_all_modes_produce_ordered_nonnegative_durations():
    segs = [_seg(1, 10.00, 10.15), _seg(2, 10.30, 10.45), _seg(3, 10.60, 10.75)]
    for mode in ("off", "minimal", "full"):
        out = TimestampProcessor(mode=mode, min_duration=1.0).process(
            [_seg(s.index, s.start, s.end) for s in segs], audio_duration=60.0
        )
        assert all(s.end >= s.start for s in out), mode
        assert _overlaps(out) == [], mode


def test_proportional_split_stays_within_parent_span():
    # No word timestamps -> proportional fallback. Must never exceed seg.end or
    # go negative (the old code inflated each piece to min_duration and blew past
    # the parent, forcing a negative-duration final piece).
    proc = TimestampProcessor(mode="off", min_duration=1.0)
    parent = SubtitleSegment(1, 5.0, 8.0, "Go. Run. Hide. Wait. Stop. Now.", words=None)
    pieces = proc._split_segment_proportionally(parent)
    assert len(pieces) > 1
    assert all(p.end >= p.start for p in pieces)
    assert all(p.end <= 8.0 + 1e-9 for p in pieces)
    assert pieces[0].start == 5.0
    assert abs(pieces[-1].end - 8.0) < 1e-9


def test_split_sentences_full_pipeline_no_negative_durations():
    proc = TimestampProcessor(mode="minimal", min_duration=1.0, split_sentences=True)
    parent = SubtitleSegment(1, 5.0, 8.0, "Go. Run. Hide. Wait. Stop. Now.", words=None)
    out = proc.process([parent], audio_duration=60.0)
    assert all(s.end >= s.start for s in out)


def test_word_timestamps_preserved_through_minimal():
    segs = [_seg(1, 1.0, 2.0), _seg(2, 5.0, 6.0)]
    out = TimestampProcessor(mode="minimal").process(segs, audio_duration=60.0)
    assert all(s.has_word_timestamps() for s in out)
