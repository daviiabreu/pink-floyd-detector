import argparse
import ctypes
import json
import platform
import time
from pathlib import Path

import numpy as np

from tools.catalog import PROFILES, match_detailed, peaks, playback
from tools.common import (
    ROOT,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    WINDOW_US,
    DecisionState,
    extract,
    file_hash,
    native,
    predict,
    windows,
)
from tools.library import decode
from tools.quantize import PRECISIONS


def simulate(source, output, drop_every=0):
    source, output = Path(source), Path(output)
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    catalog = json.loads((ROOT / "model/catalog.json").read_text())
    if metadata.get("catalog_id") != catalog["catalog_id"]:
        raise ValueError("Treine o modelo para o catálogo atual")
    labels = metadata["labels"]
    libraries, states = {}, {}
    for mode, precision in PRECISIONS.items():
        lib = native(with_model=True, precision=precision)
        libraries[mode], states[mode] = lib, DecisionState()
        lib.decision_reset(ctypes.byref(states[mode]))
    events, segments, timings = [], [], []
    clock_us, sequence, window_index = 0, 0, 0
    rng = np.random.default_rng(424242)

    def run_window(pcm, expected, segment):
        nonlocal clock_us, sequence, window_index
        window_index += 1
        dropped = bool(drop_every and window_index % drop_every == 0)
        sequence += WINDOW_FRAMES + 1 if dropped else WINDOW_FRAMES
        start = time.perf_counter_ns()
        acoustic = extract(pcm, libraries["fp32"])
        after_acoustic = time.perf_counter_ns()
        landmark = peaks(pcm, libraries["fp32"])
        after_peaks = time.perf_counter_ns()
        scores, evidence, evidence_count = match_detailed(
            landmark, len(labels) - 1, libraries["fp32"]
        )
        after_match = time.perf_counter_ns()
        vector = np.concatenate((acoustic, scores))
        results = {}
        for mode, lib in libraries.items():
            _, candidate = predict(vector, len(labels), lib)
            alert = lib.decision_update(
                ctypes.byref(states[mode]),
                candidate,
                evidence,
                evidence_count,
                sequence,
                clock_us,
                clock_us + WINDOW_US + 100,
            )
            results[mode] = {"candidate": candidate, "alert": alert}
        timings.append(
            {
                "mfcc_us": (after_acoustic - start) / 1000,
                "spectral_peaks_us": (after_peaks - after_acoustic) / 1000,
                "catalog_match_us": (after_match - after_peaks) / 1000,
                "feature_pipeline_us": (after_match - start) / 1000,
            }
        )
        events.append(
            {
                "window": window_index,
                "segment": segment,
                "expected": expected,
                "virtual_end_s": (clock_us + WINDOW_US + 100) / 1e6,
                "injected_gap": dropped,
                "precisions": results,
            }
        )
        clock_us += WINDOW_US

    def add_segment(pcm, expected, profile, path, offset):
        for _ in range(2):
            run_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16), -1, -1)
        segment = {
            "label": expected,
            "profile": profile,
            "path": path,
            "source_offset_samples": offset,
            "start_s": clock_us / 1e6,
            "first_event": len(events),
        }
        index = len(segments)
        for window in windows(playback(pcm, profile, rng)):
            run_window(window, expected, index)
        segment["end_event"] = len(events)
        segments.append(segment)

    for song in catalog["references"]:
        path = source / song["path"]
        if file_hash(path) != song["sha256"]:
            raise ValueError(f"Referência modificada: {path}")
        pcm = decode(path)
        length = min(6 * WINDOW_SAMPLES, len(pcm))
        offset = int((len(pcm) - length) * 0.37)
        for profile in ["clean", "room", "speaker"]:
            add_segment(
                pcm[offset : offset + length],
                labels.index(song["label"]),
                profile,
                song["path"],
                offset,
            )
        print(f"Fluxo contínuo: {song['label']}", flush=True)
    negative_manifest = json.loads((ROOT / "docs/negative-music.json").read_text())
    for index, track in enumerate(negative_manifest["tracks"]):
        if track["split"] != "test":
            continue
        path = ROOT / "data/negative_music" / track["path"]
        if file_hash(path) != track["sha256"]:
            raise ValueError(f"Áudio modificado: {path}")
        pcm = decode(path)
        length = 10 * WINDOW_SAMPLES
        offset = int((len(pcm) - length) * 0.37)
        add_segment(
            pcm[offset : offset + length],
            -1,
            PROFILES[index % len(PROFILES)],
            "negative_music/" + track["path"],
            offset,
        )
    results = {}
    for mode in libraries:
        confirmations, wrong_alerts = [], 0
        for segment in (s for s in segments if s["label"] >= 0):
            actual = events[segment["first_event"] : segment["end_event"]]
            hits = [e for e in actual if e["precisions"][mode]["alert"] == segment["label"]]
            confirmations.append(hits[0]["virtual_end_s"] - segment["start_s"] if hits else None)
            wrong_alerts += sum(
                e["precisions"][mode]["alert"] not in {-1, segment["label"]} for e in actual
            )
        false_events, previous, previous_expected = 0, -1, -1
        for event in events:
            alert = event["precisions"][mode]["alert"]
            if (
                event["expected"] < 0
                and alert >= 0
                and (alert != previous or previous_expected >= 0)
            ):
                false_events += 1
            previous = alert
            previous_expected = event["expected"]
        measured = [v for v in confirmations if v is not None]
        target_segments = [s for s in segments if s["label"] >= 0]
        confirmed_labels = {
            s["label"]
            for s, elapsed in zip(target_segments, confirmations, strict=True)
            if elapsed is not None
        }
        results[mode] = {
            "confirmed_target_titles": len(confirmed_labels),
            "unconfirmed_target_titles": [
                label for i, label in enumerate(labels) if i > 0 and i not in confirmed_labels
            ],
            "candidate_accuracy": float(
                np.mean([e["expected"] == e["precisions"][mode]["candidate"] for e in events])
            ),
            "target_events": len(confirmations),
            "confirmed_target_events": len(measured),
            "wrong_alert_windows_on_targets": wrong_alerts,
            "false_alert_events": false_events,
            "background_false_alert_windows": sum(
                e["expected"] < 0 and e["precisions"][mode]["alert"] >= 0 for e in events
            ),
            "virtual_confirmation_s": {
                "median": float(np.median(measured)),
                "p95": float(np.percentile(measured, 95)),
                "max": max(measured),
            }
            if measured
            else None,
        }
    report = {
        "model_id": metadata["model_id"],
        "catalog_id": catalog["catalog_id"],
        "kind": "continuous_simulated_playback",
        "decision_policy": "temporal_alignment_v1",
        "hardware_tested": False,
        "dataset_limits": metadata.get("dataset_limits"),
        "seed": 424242,
        "drop_every": drop_every,
        "labels": labels,
        "windows": len(events),
        "target_recordings": len(catalog["references"]),
        "target_titles": len(labels) - 1,
        "negative_music_recordings": sum(s["label"] < 0 for s in segments),
        "virtual_duration_s": clock_us / 1e6,
        "background_duration_s": sum(e["expected"] < 0 for e in events) * WINDOW_US / 1e6,
        "host": platform.platform(),
        "host_timings": {
            key: {
                "p50": float(np.median([t[key] for t in timings])),
                "p95": float(np.percentile([t[key] for t in timings], 95)),
                "max": max(t[key] for t in timings),
            }
            for key in timings[0]
        },
        "results": results,
        "segments": segments,
        "note": "Áudio processado pelo DSP, matcher, modelo e confirmação em C. "
        "Degradação aplicada ao segmento inteiro e duas janelas de silêncio entre segmentos. "
        "Relógio virtual alinhado às janelas, 100 us de atraso fixo para exercitar a decisão. "
        "Não simula I2S, DMA, escalonador FreeRTOS nem mede latência da placa. "
        "Tempos do host incluem a ponte Python e inicialização do DSP por janela.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    raw = ROOT / "runs" / (output.stem + "-windows.jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("".join(json.dumps(e) + "\n" for e in events))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "segments"}, indent=2, ensure_ascii=False
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exercita áudio, classificação e confirmação em C no computador"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reproducao-continua.json")
    parser.add_argument("--drop-every", type=int, default=0)
    args = parser.parse_args()
    if args.drop_every < 0:
        parser.error("--drop-every deve ser zero ou positivo")
    simulate(args.source, args.output, args.drop_every)
