import hashlib
import io
import json
import shutil
import struct
import warnings

import numpy as np
import pytest
from scipy.io import wavfile

from tools import negatives
from tools.audio import PACKET_BYTES, parse_packet
from tools.bench import COUNTERS, TIMINGS, summarize
from tools.common import FEATURE_COUNT, SAMPLE_RATE, WINDOW_SAMPLES, file_hash
from tools.train import load_dataset


def test_serial_decoder_handles_split_header_and_boot_noise():
    payload = bytes(1024)
    packet = b"AUD0" + struct.pack("<I", 42) + payload
    buffer = bytearray(b"boot message\n" + packet[:2])
    assert parse_packet(buffer) is None
    buffer.extend(packet[2:20])
    assert parse_packet(buffer) is None
    buffer.extend(packet[20:])
    assert parse_packet(buffer) == (42, payload)
    assert len(buffer) == 0
    assert len(packet) == PACKET_BYTES


def test_dataset_rejects_duplicate_recordings_across_splits(tmp_path):
    rng = np.random.default_rng(12)
    for split in ["train", "val", "test"]:
        for label in ["ambiente", "music_a", "music_b"]:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                wavfile.write(
                    folder / f"{i}.wav",
                    SAMPLE_RATE,
                    rng.integers(-1000, 1000, WINDOW_SAMPLES, dtype=np.int16),
                )
    shutil.copy(tmp_path / "train/music_a/0.wav", tmp_path / "test/music_a/1.wav")
    with pytest.raises(ValueError, match="duplicado"):
        load_dataset(tmp_path)


def test_bench_detects_reboots_and_reports_counter_differences():
    base = {
        "type": "stats",
        **dict.fromkeys(COUNTERS, 0),
        "uptime_us": 1_000_000,
        "frames": 32,
        "free_heap": 100000,
    }
    result = {
        "type": "result",
        "candidate": "music_a",
        "alert": "desconhecida",
        **dict.fromkeys(TIMINGS, 100),
    }
    later = {**base, "uptime_us": 2_000_000, "frames": 63, "ring_drops": 2}
    report = summarize([base, result, later], "music_a")
    assert report["candidate_accuracy"] == 1
    assert report["counter_deltas"]["ring_drops"] == 2
    assert report["frames_per_second"] == 31
    with pytest.raises(ValueError, match="reiniciou"):
        summarize([later, result, base], "music_a")


def test_stress_run_can_report_losses_without_completed_windows():
    base = {
        "type": "stats",
        **dict.fromkeys(COUNTERS, 0),
        "uptime_us": 1_000_000,
        "frames": 32,
        "free_heap": 100000,
    }
    later = {**base, "uptime_us": 2_000_000, "frames": 63, "ring_drops": 19}
    report = summarize([base, later], "desconhecida")
    assert report["windows"] == 0
    assert report["candidate_accuracy"] is None
    assert report["counter_deltas"]["ring_drops"] == 19
    assert report["timings_us"]["inference_us"] is None


def test_bench_rejects_mixed_model_precisions():
    base = {
        "type": "stats",
        **dict.fromkeys(COUNTERS, 0),
        "uptime_us": 1_000_000,
        "frames": 32,
        "free_heap": 100000,
    }
    events = [
        {
            "type": "result",
            "precision": mode,
            "candidate": "money",
            "alert": "money",
            **dict.fromkeys(TIMINGS, 100),
        }
        for mode in ["fp32", "int8"]
    ]
    with pytest.raises(ValueError, match="precis"):
        summarize([base, *events, {**base, "uptime_us": 2_000_000, "frames": 63}], "money")


def test_recording_labels_keep_background_first_even_with_alphabetically_earlier_titles(tmp_path):
    rng = np.random.default_rng(928)
    for split in ["train", "val", "test"]:
        for label in ["a_song", "ambiente", "z_song"]:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                wavfile.write(
                    folder / f"{i}.wav",
                    SAMPLE_RATE,
                    rng.integers(-1000, 1000, WINDOW_SAMPLES, dtype=np.int16),
                )
    labels, splits, _ = load_dataset(tmp_path)
    assert labels == ["ambiente", "a_song", "z_song"]
    assert all(set(targets) == {0, 1, 2} for _, targets, _ in splits.values())


def test_negative_download_fetches_shared_destination_once(tmp_path, monkeypatch):
    content = b"audio fixture"
    track = {
        "path": "same.mp3",
        "title": "Same recording",
        "url": "https://example.test/first.mp3",
        "sha256": hashlib.sha256(content).hexdigest(),
        "split": "train",
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/negative-music.json").write_text(
        json.dumps({"tracks": [track, {**track, "split": "test"}]})
    )
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(negatives, "ROOT", tmp_path)
    monkeypatch.setattr(negatives.urllib.request, "urlopen", fetch)
    with pytest.warns(UserWarning, match="compartilhados"):
        negatives.download(tmp_path / "audio")
    assert calls == [track["url"]]
    assert (tmp_path / "audio/same.mp3").read_bytes() == content


def test_negative_download_rejects_conflicting_destinations_before_fetch(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/negative-music.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "same.mp3", "sha256": digest, "split": "train"}
                    for digest in ["first", "different"]
                ]
            }
        )
    )
    monkeypatch.setattr(negatives, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Mesmo destino"):
        negatives.download(tmp_path / "audio")
    assert list((tmp_path / "audio").iterdir()) == []


def test_negative_manifest_detects_identical_bytes_under_different_names():
    tracks = [
        {"path": name, "sha256": "same", "split": split}
        for name, split in [("first.mp3", "train"), ("renamed.mp3", "test")]
    ]
    assert negatives.manifest_overlap(tracks) == {"first.mp3": ["train", "test"]}


@pytest.mark.parametrize("shared_negative", [False, True])
def test_cached_dataset_reports_shared_negatives_without_changing_queries(
    tmp_path, shared_negative
):
    path = tmp_path / "features.npz"
    arrays = {}
    for split in ["train", "val", "test"]:
        negative = "train" if shared_negative and split == "test" else split
        arrays.update(
            {
                f"{split}_x": np.zeros((3, FEATURE_COUNT), dtype=np.float32),
                f"{split}_y": np.arange(3),
                f"{split}_groups": np.array(
                    [f"negative/{negative}.mp3", "catalog/a.mp3", "catalog/b.mp3"]
                ),
            }
        )
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(
            {"labels": ["ambiente", "a", "b"], "recordings": [], "dataset_sha256": file_hash(path)}
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, splits, _ = load_dataset(path)
    assert len(caught) == int(shared_negative)
    if shared_negative:
        assert "negative/train.mp3" in str(caught[0].message)
    for split, (features, targets, groups) in splits.items():
        np.testing.assert_array_equal(features, arrays[f"{split}_x"])
        np.testing.assert_array_equal(targets, arrays[f"{split}_y"])
        assert groups == arrays[f"{split}_groups"].tolist()
