import ctypes

from tools.common import WINDOW_FRAMES, WINDOW_US, CatalogEvidence, DecisionState, native


def detector():
    lib = native()
    state = DecisionState()
    lib.decision_reset(ctypes.byref(state))

    def update(candidate, sequence, matches=None, start=None, now=None):
        if start is None:
            start = sequence * 32000
        if now is None:
            now = start + WINDOW_US + 100
        if matches is None:
            matches = [(sequence + 1000, 0, 1, 6, 2)]
        evidence = (CatalogEvidence * len(matches))(*(CatalogEvidence(*m) for m in matches))
        return lib.decision_update(
            ctypes.byref(state), candidate, evidence, len(matches), sequence, start, now
        )

    return update, state


def test_three_aligned_windows_can_confirm_with_one_confident_model_result():
    update, state = detector()
    assert update(-1, 64) == -1
    assert update(1, 128) == -1
    assert update(-1, 192) == 1
    assert state.support_windows == 3
    assert state.aligned_frames == 18
    assert state.first_start_us == 64 * 32000
    assert update(-1, 256, matches=[]) == -1
    assert state.support_windows == 0


def test_matching_title_without_time_alignment_never_confirms():
    update, _ = detector()
    for sequence in range(64, 1024, 64):
        assert update(1, sequence, matches=[(1000, 0, 1, 10, 1)]) == -1


def test_alignment_without_a_confident_model_result_never_confirms():
    update, _ = detector()
    for sequence in range(64, 1024, 64):
        assert update(-1, sequence) == -1


def test_repeated_patterns_and_weak_matches_do_not_confirm():
    for votes, second in [(2, 0), (6, 5), (6, 6)]:
        update, _ = detector()
        for sequence in range(64, 1024, 64):
            assert update(1, sequence, matches=[(sequence + 1000, 0, 1, votes, second)]) == -1


def test_low_evidence_keeps_listening_until_enough_distinct_frames():
    update, state = detector()
    for i in range(1, 6):
        assert update(1, i * 64, matches=[(1000 + i * 64, 0, 1, 3, 0)]) == -1
    assert update(1, 384, matches=[(1384, 0, 1, 3, 0)]) == 1
    assert state.support_windows == 6


def test_short_gaps_preserve_evidence_but_cannot_keep_led_on():
    update, _ = detector()
    assert update(1, 64) == -1
    assert update(-1, 128, matches=[]) == -1
    assert update(-1, 192) == -1
    assert update(-1, 256) == 1
    assert update(-1, 320, matches=[]) == -1


def test_old_evidence_expires_and_cannot_confirm_a_later_sound():
    update, _ = detector()
    update(1, 64)
    update(-1, 128)
    for sequence in range(192, 768, 64):
        assert update(-1, sequence, matches=[]) == -1
    assert update(-1, 768) == -1
    assert update(-1, 832) == -1
    assert update(-1, 896) == -1


def test_versions_of_one_title_cannot_pool_unrelated_offsets():
    update, _ = detector()
    for i in range(1, 4):
        assert update(1, i * 64, matches=[(1000 + i * 64, i, 1, 10, 1)]) == -1


def test_two_qualified_titles_are_rejected_as_ambiguous():
    update, _ = detector()
    for i, candidate in enumerate([1, 2, -1], 1):
        matches = [(1000 + i * 64, 0, 1, 10, 1), (2000 + i * 64, 1, 2, 10, 1)]
        assert update(candidate, i * 64, matches=matches) == -1


def test_alignment_tolerates_two_frames_but_not_larger_drift():
    update, _ = detector()
    update(1, 64, matches=[(1066, 0, 1, 6, 2)])
    update(-1, 128, matches=[(1128, 0, 1, 6, 2)])
    assert update(-1, 192, matches=[(1193, 0, 1, 6, 2)]) == 1
    assert update(-1, 256, matches=[(1262, 0, 1, 6, 2)]) == -1


def test_lost_audio_and_dropped_feature_packets_reset_accumulation():
    update, _ = detector()
    update(1, 64)
    update(1, 128)
    assert update(1, 193) == -1
    assert update(1, 321) == -1
    assert update(1, 385) == -1
    assert update(1, 449) == 1


def test_stale_or_invalid_timestamp_cannot_activate_led():
    update, _ = detector()
    update(1, 64)
    update(1, 128)
    assert update(1, 192, start=0, now=WINDOW_US + 1_000_001) == -1
    update(1, 256)
    update(1, 320)
    assert update(1, 384, start=10, now=9) == -1


def test_sequence_counter_wrap_is_continuous():
    update, _ = detector()
    for i, sequence in enumerate([2**32 - 96, 2**32 - 32, 32]):
        result = update(1, sequence, matches=[(1000 + i * 64, 0, 1, 6, 2)], start=i * WINDOW_US)
        assert result == (1 if i == 2 else -1)


def test_continuous_losses_prevent_confirmation_and_recover_afterwards():
    update, _ = detector()
    sequence = 0
    for _ in range(20):
        sequence += WINDOW_FRAMES + 1
        assert update(1, sequence) == -1
    assert update(1, sequence + 64) == -1
    assert update(1, sequence + 128) == 1
