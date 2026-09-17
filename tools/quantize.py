import argparse
import json
import platform
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tools.common import FLOAT_PTR, ROOT, file_hash, native
from tools.train import c_array, decisions, load_dataset, scores

PRECISIONS = {"fp32": 32, "fp16": 16, "int16": -16, "int8": 8, "int4": 4}


def quantize_weights(weights, bits):
    maximum = 2 ** (bits - 1) - 1
    scales = np.maximum(np.max(np.abs(weights), axis=0) / maximum, 1e-10).astype(np.float32)
    values = np.clip(np.rint(weights / scales), -maximum, maximum)
    return values.astype(np.int16 if bits == 16 else np.int8), scales


def pack_int4(values):
    values = np.asarray(values).reshape(-1)
    if np.any(values < -8) or np.any(values > 7):
        raise ValueError("Valor fora da faixa int4")
    packed = np.zeros((len(values) + 1) // 2, dtype=np.uint8)
    for index, value in enumerate(values):
        packed[index // 2] |= (int(value) & 15) << (4 * (index % 2))
    return packed


def export_variant(mode, arrays, metadata, calibration, destination):
    weights = arrays["weights"]
    inputs = {name: array.copy() for name, array in arrays.items() if name != "weights"}
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["scaled"]),
    ]
    header = [
        "#pragma once",
        "#include <stdint.h>",
        f"#define MODEL_STORAGE {PRECISIONS[mode]}",
        f"#define MODEL_FEATURES {weights.shape[0]}",
        f"#define MODEL_CLASSES {weights.shape[1]}",
        f"#define MODEL_BACKGROUND_CLASS {metadata['labels'].index('ambiente')}",
        f'#define MODEL_ID "{metadata["model_id"]}"',
        f'#define MODEL_DATASET "{metadata["dataset_kind"]}"',
    ]
    if metadata.get("catalog_id"):
        header.append(f"#define MODEL_CATALOG_ID 0x{metadata['catalog_id']}ULL")
    for key in ["threshold", "margin", "min_log_rms"]:
        header.append(f"#define MODEL_{key.upper()} {metadata[key]:.9e}f")
    header.append(
        "static const char *const MODEL_LABELS[MODEL_CLASSES] = {"
        + ", ".join(json.dumps(label) for label in metadata["labels"])
        + "};"
    )
    for name in ["mean", "scale", "bias"]:
        header.append(c_array("MODEL_" + name.upper(), arrays[name]))
    parameter_bytes = sum(array.nbytes for array in inputs.values())
    activation_clip_rate = 0.0
    if mode == "fp16":
        inputs["weights_half"] = weights.astype(np.float16)
        nodes.append(helper.make_node("Cast", ["weights_half"], ["weights"], to=TensorProto.FLOAT))
        header.append(
            c_array("MODEL_WEIGHTS_HALF", weights.astype(np.float16).view(np.uint16), "uint16_t")
        )
        parameter_bytes += inputs["weights_half"].nbytes
        input_name = "scaled"
    else:
        bits = 16 if mode == "int16" else 4 if mode == "int4" else 8
        activation_bits = 16 if mode == "int16" else 8
        values, weight_scale = quantize_weights(weights, bits)
        input_max = 2 ** (activation_bits - 1) - 1
        bound = np.float32(max(float(np.max(np.abs(calibration))), 1e-5))
        input_scale = np.float32(bound / input_max)
        dtype = np.int16 if activation_bits == 16 else np.int8
        weight_dtype = ml_dtypes.int4 if bits == 4 else values.dtype
        inputs.update(
            {
                "input_scale": np.array(input_scale),
                "input_zero": np.array(0, dtype=dtype),
                "input_min": np.array(-bound),
                "input_max": np.array(bound),
                "weights_q": values.astype(weight_dtype),
                "weight_scale": weight_scale,
                "weight_zero": np.zeros(weights.shape[1], dtype=weight_dtype),
            }
        )
        nodes += [
            helper.make_node("Clip", ["scaled", "input_min", "input_max"], ["bounded"]),
            helper.make_node(
                "QuantizeLinear", ["bounded", "input_scale", "input_zero"], ["input_q"]
            ),
            helper.make_node(
                "DequantizeLinear", ["input_q", "input_scale", "input_zero"], ["input_dq"]
            ),
            helper.make_node(
                "DequantizeLinear",
                ["weights_q", "weight_scale", "weight_zero"],
                ["weights"],
                axis=1,
            ),
        ]
        header += [
            f"#define MODEL_INPUT_QMAX {input_max}.0f",
            f"#define MODEL_INPUT_SCALE {input_scale:.9e}f",
            c_array("MODEL_WEIGHT_SCALE", weight_scale),
        ]
        if bits == 4:
            packed = pack_int4(values)
            header.append(c_array("MODEL_WEIGHTS_PACKED", packed, "uint8_t"))
            weight_bytes = packed.nbytes
        else:
            header.append(c_array("MODEL_WEIGHTS_Q", values, "int16_t" if bits == 16 else "int8_t"))
            weight_bytes = values.nbytes
        parameter_bytes += weight_bytes + weight_scale.nbytes + 4
        input_name = "input_dq"
    nodes += [
        helper.make_node("Gemm", [input_name, "weights", "bias"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        f"music_{mode}",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, weights.shape[0]])],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, weights.shape[1]]
            )
        ],
        initializer=[numpy_helper.from_array(array, name) for name, array in inputs.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    helper.set_model_props(
        model,
        {
            "precision": mode,
            "base_model_id": metadata["model_id"],
            "labels": json.dumps(metadata["labels"]),
            "dataset": metadata["dataset_kind"],
        },
    )
    onnx.checker.check_model(model)
    destination.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination / f"{mode}.onnx")
    (destination / f"{mode}.h").write_text("\n".join(header) + "\n")
    return {"parameter_bytes": parameter_bytes, "activation_clip_rate": activation_clip_rate}


def batch_predict(lib, features, classes):
    features = np.ascontiguousarray(features, dtype=np.float32)
    probabilities = np.empty((len(features), classes), dtype=np.float32)
    lib.model_predict_batch(
        features.ctypes.data_as(FLOAT_PTR), probabilities.ctypes.data_as(FLOAT_PTR), len(features)
    )
    return probabilities


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def benchmark(lib, session, features):
    classes = session.get_outputs()[0].shape[1]
    batch_size = max(8, min(256, 1_000_000 // (features.shape[1] * classes)))
    repetitions = max(1, min(20, 1_000_000 // (batch_size * features.shape[1] * classes)))
    batch = np.ascontiguousarray(features[:batch_size], dtype=np.float32)
    pointer = batch.ctypes.data_as(FLOAT_PTR)
    for _ in range(5):
        lib.model_benchmark(pointer, len(batch), repetitions)
    c_times = []
    for _ in range(30):
        start = time.perf_counter_ns()
        checksum = lib.model_benchmark(pointer, len(batch), repetitions)
        c_times.append((time.perf_counter_ns() - start) / 1000 / (len(batch) * repetitions))
        if not np.isfinite(checksum):
            raise ValueError("Inferência não finita no benchmark")
    for _ in range(100):
        session.run(None, {"features": batch[:1]})
    onnx_times = []
    for i in range(1000):
        item = batch[i % len(batch) : i % len(batch) + 1]
        start = time.perf_counter_ns()
        session.run(None, {"features": item})
        onnx_times.append((time.perf_counter_ns() - start) / 1000)
    return {
        "c_batch_size": len(batch),
        "c_repetitions_per_measurement": repetitions,
        "c_mean_us": float(np.mean(c_times)),
        "c_batch_mean_p95_us": float(np.percentile(c_times, 95)),
        "onnx_single_window_p95_us": float(np.percentile(onnx_times, 95)),
    }


def run(dataset, output):
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    if metadata.get("dataset_sha256") and file_hash(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Cache de consultas diferente do registrado no modelo")
    labels, splits, inventory = load_dataset(dataset)
    if labels != metadata["labels"] or inventory != metadata["recordings"]:
        raise ValueError("Dataset não corresponde ao modelo; treine antes de quantizar")
    model_path = ROOT / "model/model.onnx"
    model = onnx.load(model_path)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    train_features = splits["train"][0]
    calibration = ((train_features - arrays["mean"]) / arrays["scale"]).astype(np.float32)
    variants = ROOT / "model/variants"
    sizes = {"fp32": {"parameter_bytes": sum(value.nbytes for value in arrays.values())}}
    for mode in ["fp16", "int16", "int8", "int4"]:
        sizes[mode] = export_variant(mode, arrays, metadata, calibration, variants)
    test_features, targets, _ = splits["test"]
    background = labels.index("ambiente")
    baseline = cpu_session(model_path).run(None, {"features": test_features})[0]
    baseline_decisions = decisions(
        test_features,
        baseline,
        background,
        metadata["threshold"],
        metadata["margin"],
        metadata["min_log_rms"],
    )
    reports = []
    for mode, precision in PRECISIONS.items():
        print(f"Avaliando {mode}...", flush=True)
        path = model_path if mode == "fp32" else variants / f"{mode}.onnx"
        lib = native(with_model=True, precision=precision)
        session = cpu_session(path)
        actual = batch_predict(lib, test_features, len(labels))
        expected = session.run(None, {"features": test_features})[0]
        error = float(np.max(np.abs(actual - expected)))
        if error > 2e-4:
            raise ValueError(f"{mode}: divergência entre ONNX e C: {error}")
        predicted = decisions(
            test_features,
            actual,
            background,
            metadata["threshold"],
            metadata["margin"],
            metadata["min_log_rms"],
        )
        metrics = scores(targets, predicted, labels, background)
        positive = targets != background
        metric = {
            "precision": mode,
            **sizes[mode],
            "onnx_bytes": path.stat().st_size,
            **metrics,
            "positive_top1_accuracy": float(
                np.mean(actual[positive].argmax(axis=1) == targets[positive])
            ),
            "onnx_c_max_absolute_error": error,
            "fp32_max_probability_change": float(np.max(np.abs(actual - baseline))),
            "fp32_decision_agreement": float(np.mean(predicted == baseline_decisions)),
            "fp32_top1_agreement": float(np.mean(actual.argmax(axis=1) == baseline.argmax(axis=1))),
            **benchmark(lib, session, test_features),
        }
        if mode.startswith("int"):
            standardized = (test_features - arrays["mean"]) / arrays["scale"]
            metric["activation_clip_rate"] = float(
                np.mean(np.abs(standardized) > np.max(np.abs(calibration)))
            )
        reports.append(metric)
    report = {
        "model_id": metadata["model_id"],
        "dataset": metadata["dataset_kind"],
        "audio_identity_verified": metadata.get("audio_identity_verified", False),
        "dataset_limits": metadata.get("dataset_limits"),
        "host": platform.platform(),
        "onnxruntime": ort.__version__,
        "hardware_tested": False,
        "labels": labels,
        "threshold": metadata["threshold"],
        "margin": metadata["margin"],
        "split_method": metadata["split_method"],
        "results": reports,
        "method": "Same weights, splits and threshold; calibration uses train only. "
        "30 measurements of native C batches after warmup, 1000 ONNX single-window calls, CPU 1 thread.",
        "limits": "FP16 stores only weights in half and accumulates in FP32. INT16/INT8 use integer "
        "dense products in C, with floating-point scaling, bias and softmax. INT4 packs weights "
        "and uses INT8 activations. ONNX uses QDQ and its CPU optimizer may choose different kernels. "
        "No ESP32 timing or independent acoustic capture measured.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "results": [
                    {
                        k: row[k]
                        for k in [
                            "precision",
                            "accuracy",
                            "macro_f1",
                            "parameter_bytes",
                            "c_mean_us",
                        ]
                    }
                    for row in reports
                ],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta FP16/INT16/INT8/INT4 e compara com FP32")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/quantizacao.json"))
    args = parser.parse_args()
    run(args.data, args.output)
