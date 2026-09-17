import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from tools.audio import serial_open

TIMINGS = [
    "capture_max_us",
    "capture_window_us",
    "feature_sum_us",
    "feature_max_us",
    "ring_wait_max_us",
    "queue_wait_us",
    "fingerprint_us",
    "inference_us",
    "software_e2e_us",
    "confirmation_us",
]
COUNTERS = [
    "read_errors",
    "ring_drops",
    "feature_drops",
    "log_drops",
    "dma_overflows",
    "dsp_deadlines",
    "stale_windows",
]


def summarize(events, label):
    results = [event for event in events if event.get("type") == "result"]
    stats = [event for event in events if event.get("type") == "stats"]
    if len(stats) < 2:
        raise ValueError("Ensaio sem estatísticas suficientes")
    for previous, current in pairwise(stats):
        if current["uptime_us"] <= previous["uptime_us"]:
            raise ValueError("A placa reiniciou durante o ensaio; não misture as medições")
        if any(current[key] < previous[key] for key in COUNTERS):
            raise ValueError("Contadores regrediram; reinício ou overflow durante o ensaio")
    elapsed = (stats[-1]["uptime_us"] - stats[0]["uptime_us"]) / 1e6
    delta = {key: stats[-1][key] - stats[0][key] for key in COUNTERS}
    timings = {}
    for key in TIMINGS:
        values = [
            event[key]
            for event in results
            if key in event and (key != "confirmation_us" or event[key] > 0)
        ]
        timings[key] = (
            {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": max(values),
            }
            if values
            else None
        )
    predicted = [event["candidate"] for event in results]
    precisions = sorted({event["precision"] for event in results if "precision" in event})
    if len(precisions) > 1:
        raise ValueError("O ensaio mistura precisões diferentes; use uma pasta por firmware")
    return {
        "hardware_tested": True,
        "ground_truth": label,
        "precision": precisions[0] if precisions else None,
        "windows": len(results),
        "candidate_accuracy": float(np.mean(np.array(predicted) == label)) if predicted else None,
        "alerts_observed": sum(event["alert"] != "desconhecida" for event in results),
        "frames_per_second": (stats[-1]["frames"] - stats[0]["frames"]) / elapsed,
        "counter_deltas": delta,
        "timings_us": timings,
        "min_free_heap": min(event["free_heap"] for event in stats),
        "note": "Rótulo fornecido pelo operador para um trecho contínuo. A contagem de alertas é "
        "por janela; confirmation_us começa na primeira janela com coincidências usadas na confirmação, "
        "não no início acústico.",
    }


def collect(args):
    if args.seconds < 10:
        raise ValueError("Meça ao menos 10 segundos")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "serial.jsonl"
    report_path = args.output / "summary.json"
    if raw.exists() or report_path.exists():
        raise FileExistsError("Use outra pasta para preservar o ensaio anterior")
    events = []
    deadline = time.monotonic() + args.seconds
    with serial_open(args.port, 115200) as device, raw.open("x") as output:
        while time.monotonic() < deadline:
            line = device.readline().decode(errors="replace").strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            output.write(json.dumps(event) + "\n")
    report = summarize(events, args.label)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mede o detector real por serial e salva os dados brutos"
    )
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--label", required=True, help="Rótulo esperado ou 'desconhecida' para negativos"
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())
