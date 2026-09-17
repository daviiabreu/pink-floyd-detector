import ctypes
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest

from tools.catalog import match, match_detailed, peaks
from tools.common import (
    FINGERPRINT_FFT,
    FINGERPRINT_FRAMES,
    FRAME_SIZE,
    PCM_PTR,
    PEAK_PTR,
    ROOT,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    native,
)


def numpy_peaks(pcm):
    frames = (
        np.lib.stride_tricks.sliding_window_view(pcm, FINGERPRINT_FFT)[::FRAME_SIZE].astype(float)
        / 32768
    )
    frames -= frames.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(frames * np.hanning(FINGERPRINT_FFT))) ** 2
    local = (power[:, 1:-1] > power[:, :-2]) & (power[:, 1:-1] >= power[:, 2:])
    scores = np.zeros_like(power)
    scores[:, 1:-1] = np.where(local, power[:, 1:-1], 0)
    scores[:, : FINGERPRINT_FFT // 128] = 0
    scores[:, FINGERPRINT_FFT * 15 // 32 + 1 :] = 0
    result = []
    for _ in range(4):
        best = scores.argmax(axis=1)
        best = np.where(scores[np.arange(len(frames)), best] > 1e-10, best, 0)
        result.append(best)
        for delta in range(-5, 6):
            scores[np.arange(len(frames)), np.clip(best + delta, 0, FINGERPRINT_FFT // 2)] = 0
    return np.array(result).T


def test_c_peaks_match_independent_fft_with_arbitrary_audio_alignment():
    rng = np.random.default_rng(1265)
    lib = native()
    pcm = rng.integers(-16000, 16000, WINDOW_SAMPLES * 2, dtype=np.int16)
    for offset in [0, 1, 251, 511, 777]:
        segment = pcm[offset : offset + WINDOW_SAMPLES]
        np.testing.assert_array_equal(peaks(segment, lib), numpy_peaks(segment))
    assert not peaks(np.zeros(WINDOW_SAMPLES, dtype=np.int16), lib).any()


def test_reset_discards_history_before_a_new_contiguous_window():
    class Context(ctypes.Structure):
        _fields_ = [
            ("window", ctypes.c_float * FINGERPRINT_FFT),
            ("real", ctypes.c_float * FINGERPRINT_FFT),
            ("imag", ctypes.c_float * FINGERPRINT_FFT),
            ("pcm", ctypes.c_int16 * FINGERPRINT_FFT),
            ("frames", ctypes.c_uint),
        ]

    lib = native()
    lib.fingerprint_init.argtypes = [ctypes.POINTER(Context)]
    lib.fingerprint_reset.argtypes = [ctypes.POINTER(Context)]
    lib.fingerprint_push.argtypes = [ctypes.POINTER(Context), PCM_PTR, PEAK_PTR]
    context = Context()
    lib.fingerprint_init(ctypes.byref(context))
    rng = np.random.default_rng(45698)
    stale = rng.integers(-20000, 20000, (7, 512), dtype=np.int16)
    temporary = np.zeros(4, dtype=np.uint16)
    for frame in stale:
        lib.fingerprint_push(
            ctypes.byref(context), frame.ctypes.data_as(PCM_PTR), temporary.ctypes.data_as(PEAK_PTR)
        )
    lib.fingerprint_reset(ctypes.byref(context))
    new = rng.integers(-20000, 20000, WINDOW_SAMPLES, dtype=np.int16)
    actual, readiness = [], []
    for frame in new.reshape(WINDOW_FRAMES, FRAME_SIZE):
        ready = lib.fingerprint_push(
            ctypes.byref(context), frame.ctypes.data_as(PCM_PTR), temporary.ctypes.data_as(PEAK_PTR)
        )
        readiness.append(ready)
        if ready:
            actual.append(temporary.copy())
    assert readiness == [0] * (FINGERPRINT_FFT // FRAME_SIZE - 1) + [1] * FINGERPRINT_FRAMES
    np.testing.assert_array_equal(actual, numpy_peaks(new))


def test_catalog_index_matches_its_manifest_and_contains_no_duplicate_entries():
    source = (ROOT / "src/catalog_data.h").read_text().split("CATALOG_INDEX")[1]
    values = [int(v, 16) for v in re.findall(r"0x([0-9a-f]+)ULL", source)]
    manifest = json.loads((ROOT / "model/catalog.json").read_text())
    digest = hashlib.sha256(np.array(values, dtype="<u8").tobytes()).hexdigest()
    assert values == sorted(set(values))
    assert len(values) == manifest["entries"]
    assert digest == manifest["sha256"]
    identity = json.dumps(
        {key: manifest[key] for key in ["contract", "labels", "references"]}, sort_keys=True
    ).encode()
    assert (
        hashlib.sha256(np.array(values, dtype="<u8").tobytes() + identity).hexdigest()[:16]
        == manifest["catalog_id"]
    )
    packed = np.array(values, dtype=np.uint64)
    _, counts = np.unique(packed >> 24, return_counts=True)
    assert counts.max() <= manifest["contract"]["bucket_limit"]
    reference_ids = ((packed >> 16) & 255).astype(int)
    assert set(reference_ids) == set(range(len(manifest["references"])))
    for index, ref in enumerate(manifest["references"]):
        assert np.max(packed[reference_ids == index] & 65535) < ref["frames"]
    assert len(manifest["labels"]) == len({r["label"] for r in manifest["references"]}) + 1


def index_records():
    source = (ROOT / "src/catalog_data.h").read_text().split("CATALOG_INDEX")[1]
    packed = np.array(
        [int(value, 16) for value in re.findall(r"0x([0-9a-f]+)ULL", source)], dtype=np.uint64
    )
    keys = packed >> 24
    frequencies = np.column_stack((keys >> 22, (keys >> 11) & 2047, keys & 2047)).astype(int)
    return frequencies, ((packed >> 16) & 255).astype(int), (packed & 65535).astype(int)


def brute_force_match(query, reference, reference_ids, positions, manifest):
    support = [defaultdict(set) for _ in manifest["references"]]
    coarse = [defaultdict(set) for _ in manifest["references"]]
    for frame, frequencies in enumerate(query):
        hits = np.zeros(len(reference), dtype=bool)
        for triplet in combinations(frequencies, 3):
            if min(triplet) < FINGERPRINT_FFT // 128:
                continue
            hits |= np.all(np.abs(reference - sorted(triplet)) <= 1, axis=1)
        for ref, position in zip(reference_ids[hits], positions[hits], strict=True):
            for offset in (position - frame, position - frame + 1):
                support[ref][offset].add(frame)
                coarse[ref][(offset + FINGERPRINT_FRAMES) // 64].add(frame)
    strength = [max(map(len, h.values()), default=0) for h in coarse]
    shortlist = sorted(range(len(support)), key=lambda r: (-strength[r], r))[:16]
    result = np.zeros((len(manifest["labels"]) - 1, 2), dtype=np.float32)
    evidence = {}
    for ref in shortlist:
        histogram = support[ref]
        if not histogram:
            continue
        best = max(sorted(histogram), key=lambda offset: len(histogram[offset]))
        count = len(histogram[best])
        second = max(
            (len(v) for k, v in histogram.items() if abs(k - best) > WINDOW_FRAMES), default=0
        )
        value = np.array([count, count - second], dtype=np.float32) / FINGERPRINT_FRAMES
        label = manifest["references"][ref]["class_index"]
        evidence[ref] = (best, label + 1, count, second)
        if tuple(value) > tuple(result[label]):
            result[label] = value
    return result.ravel(), evidence


def test_alignment_votes_match_exhaustive_frequency_comparison_and_do_not_double_count():
    reference, reference_ids, positions = index_records()
    manifest = json.loads((ROOT / "model/catalog.json").read_text())
    longest = max(
        range(len(manifest["references"])), key=lambda i: manifest["references"][i]["frames"]
    )
    query = np.zeros((FINGERPRINT_FRAMES, 4), dtype=np.uint16)
    for frame in range(FINGERPRINT_FRAMES):
        hit = np.flatnonzero((reference_ids == longest) & (positions == 4000 + frame // 4 * 4))
        if len(hit):
            query[frame, :3] = reference[hit[0]]
            query[frame, 3] = reference[hit[0], 2]
    lib = native(with_catalog=True)
    songs = len(manifest["labels"]) - 1
    result = match(query, songs, lib)
    expected, expected_evidence = brute_force_match(
        query, reference, reference_ids, positions, manifest
    )
    np.testing.assert_allclose(result, expected, atol=1e-7)
    detailed, evidence, count = match_detailed(query, songs, lib)
    np.testing.assert_array_equal(detailed, result)
    assert {
        e.reference: (e.offset_frames, e.class_index, e.votes, e.second_votes)
        for e in evidence[:count]
    } == expected_evidence
    assert result[2 * manifest["references"][longest]["class_index"]] > 0.2
    assert (result >= 0).all() and (result <= 1).all()
    assert not match(np.zeros_like(query), songs, lib).any()
    with pytest.raises(ValueError):
        match(query[:20], songs, lib)
