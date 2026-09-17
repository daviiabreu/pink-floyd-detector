import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tools.common import (
    BACKGROUND,
    ROOT,
    extract_model,
    file_hash,
    native,
    predict,
    read_wav,
    windows,
)
from tools.train import load_dataset, scores


def evaluate(dataset, output, model_dir=ROOT / "model"):
    metadata = json.loads((model_dir / "metadata.json").read_text())
    if metadata.get("dataset_sha256") and file_hash(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Cache de consultas diferente do registrado no modelo")
    labels, splits, inventory = load_dataset(dataset)
    if labels != metadata["labels"] or inventory != metadata["recordings"]:
        raise ValueError(
            "Dataset diferente do registrado no modelo; treine novamente antes de avaliar"
        )
    lib = native(with_model=True)
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    features, targets, _ = splits["test"]
    onnx_probs = session.run(None, {"features": features})[0]
    predictions, errors, inference_times = [], [], []
    for vector, expected in zip(features, onnx_probs, strict=True):
        start = time.perf_counter_ns()
        probs, candidate = predict(vector, len(labels), lib)
        inference_times.append((time.perf_counter_ns() - start) / 1000)
        errors.append(float(np.max(np.abs(probs - expected))))
        predictions.append(labels.index(BACKGROUND) if candidate < 0 else candidate)
    if max(errors) > 2e-5:
        raise ValueError(f"ONNX e pesos C divergentes: erro máximo {max(errors)}")
    extraction_times = []
    for path in sorted((Path(dataset) / "test").glob("*/*.wav")) if Path(dataset).is_dir() else []:
        for pcm in windows(read_wav(path)):
            start = time.perf_counter_ns()
            extract_model(pcm, lib)
            extraction_times.append((time.perf_counter_ns() - start) / 1000)
    report = {
        "model_id": metadata["model_id"],
        "dataset": metadata["dataset_kind"],
        "audio_identity_verified": metadata.get("audio_identity_verified", False),
        "dataset_limits": metadata.get("dataset_limits"),
        "hardware_tested": False,
        "test": scores(targets, np.array(predictions), labels, labels.index(BACKGROUND)),
        "onnx_c_max_absolute_error": max(errors),
        "host_feature_window_us": {
            "p50": float(np.median(extraction_times)),
            "p95": float(np.percentile(extraction_times, 95)),
            "max": max(extraction_times),
        }
        if extraction_times
        else None,
        "host_inference_with_python_bridge_us": {
            "p50": float(np.median(inference_times)),
            "p95": float(np.percentile(inference_times, 95)),
        },
        "note": "Medições no computador, sem FreeRTOS, microfone ou tempo de aquisição. "
        "Não representam a latência nem a acurácia do ESP32.",
    }
    if Path(dataset).suffix == ".npz":
        with np.load(dataset, allow_pickle=False) as data:
            if "test_profiles" in data:
                profiles = data["test_profiles"]
                report["by_profile"] = {
                    profile: scores(
                        targets[profiles == profile],
                        np.array(predictions)[profiles == profile],
                        labels,
                        labels.index(BACKGROUND),
                    )
                    for profile in sorted(set(profiles.tolist()))
                }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o conjunto de teste e compara ONNX com C")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    evaluate(args.data, args.output)
