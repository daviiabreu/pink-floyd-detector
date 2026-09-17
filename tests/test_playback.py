import json

import numpy as np
import pytest

from tools import playback
from tools.common import ROOT, SAMPLE_RATE, WINDOW_SAMPLES, file_hash
from tools.quantize import PRECISIONS


@pytest.mark.parametrize("drop_every", [0, 1])
def test_continuous_playback_records_all_precisions(tmp_path, monkeypatch, drop_every):
    (tmp_path / "model").mkdir()
    (tmp_path / "docs").mkdir()
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    song = tmp_path / "song.wav"
    song.write_bytes(b"fixture")
    catalog = {
        "catalog_id": metadata["catalog_id"],
        "references": [
            {"path": song.name, "sha256": file_hash(song), "label": metadata["labels"][1]}
        ],
    }
    (tmp_path / "model/metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "model/catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "docs/negative-music.json").write_text('{"tracks": []}')
    t = np.arange(6 * WINDOW_SAMPLES) / SAMPLE_RATE
    pcm = np.round(4000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    monkeypatch.setattr(playback, "ROOT", tmp_path)
    monkeypatch.setattr(playback, "decode", lambda path: pcm)

    report = playback.simulate(tmp_path, tmp_path / "report.json", drop_every)
    events = [
        json.loads(line)
        for line in (tmp_path / "runs/report-windows.jsonl").read_text().splitlines()
    ]

    assert report["windows"] == len(events) == 24
    assert set(report["results"]) == set(PRECISIONS)
    assert all(set(event["precisions"]) == set(PRECISIONS) for event in events)
    for result in report["results"].values():
        assert result["target_events"] == 3
        assert 0 <= result["candidate_accuracy"] <= 1
        if drop_every:
            assert result["confirmed_target_events"] == 0
    if drop_every:
        assert all(
            result["alert"] == -1 for event in events for result in event["precisions"].values()
        )
