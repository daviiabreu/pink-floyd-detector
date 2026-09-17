import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from tools.common import (
    BACKGROUND,
    FEATURE_COUNT,
    ROOT,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    extract,
    file_hash,
    native,
    read_wav,
    windows,
)
from tools.negatives import split_overlap


def negative_overlap(splits):
    return split_overlap(
        {
            split: [group for group in groups if group.startswith("negative/")]
            for split, (_, _, groups) in splits.items()
        }
    )


def load_dataset(root):
    root = Path(root)
    if root.suffix == ".npz":
        info = json.loads(root.with_suffix(".json").read_text())
        if file_hash(root) != info["dataset_sha256"]:
            raise ValueError("Cache de features diferente do manifesto")
        with np.load(root, allow_pickle=False) as data:
            splits = {
                s: (data[f"{s}_x"], data[f"{s}_y"], data[f"{s}_groups"].tolist())
                for s in ["train", "val", "test"]
                if f"{s}_x" in data
            }
        if not {"train", "val"}.issubset(splits):
            raise ValueError("Cache precisa conter train e val")
        for features, targets, groups in splits.values():
            if (
                features.ndim != 2
                or features.shape[1] != info.get("features", FEATURE_COUNT)
                or not np.isfinite(features).all()
            ):
                raise ValueError("Features inválidas no cache")
            if len(features) != len(targets) or len(features) != len(groups):
                raise ValueError("Tamanhos incompatíveis no cache")
            if set(targets) != set(range(len(info["labels"]))):
                raise ValueError("Cada divisão deve conter todas as classes")
        if info.get("catalog_id"):
            catalog = json.loads((ROOT / "model/catalog.json").read_text())
            if info["catalog_id"] != catalog["catalog_id"]:
                raise ValueError("Cache pertence a outro catálogo; extraia as features novamente")
        overlap = negative_overlap(splits)
        if overlap:
            warnings.warn(f"Negativos compartilhados entre divisões: {overlap}", stacklevel=2)
        return info["labels"], splits, info["recordings"]
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if BACKGROUND not in labels or not 3 <= len(labels) <= 257:
        raise ValueError("Use 2 a 256 músicas e uma classe 'ambiente'")
    if any(not re.fullmatch(r"[a-z0-9_]{1,64}", label) for label in labels):
        raise ValueError("Rótulos devem usar letras minúsculas sem acentos, números e underscore")
    labels = [BACKGROUND, *(label for label in labels if label != BACKGROUND)]
    hashes = {}
    inventory, splits = [], {}
    catalog_path = ROOT / "model/catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
    use_catalog = catalog and labels == catalog["labels"]
    lib = native(with_catalog=bool(use_catalog))
    for split in ["train", "val", "test"]:
        folder = root / split
        actual = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if actual != sorted(labels):
            raise ValueError(f"{split}: as classes devem ser as mesmas de train")
        features, targets, sessions = [], [], []
        for index, label in enumerate(labels):
            files = sorted((folder / label).glob("*.wav"))
            if len(files) < 2:
                raise ValueError(f"{split}/{label}: grave pelo menos duas sessões independentes")
            for path in files:
                pcm = read_wav(path)
                digest = hashlib.sha256(pcm.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"Áudio duplicado: {path} e {hashes[digest]}")
                hashes[digest] = str(path)
                relative = path.relative_to(root).as_posix()
                inventory.append({"path": relative, "sha256_pcm": digest, "samples": len(pcm)})
                for window in windows(pcm):
                    if use_catalog:
                        from tools.catalog import features as catalog_features

                        features.append(catalog_features(window, len(labels) - 1, lib))
                    else:
                        features.append(extract(window, lib))
                    targets.append(index)
                    sessions.append(relative)
        splits[split] = (np.array(features, dtype=np.float32), np.array(targets), sessions)
    return labels, splits, inventory


def decisions(features, probabilities, background, threshold, margin, min_rms):
    best = np.argmax(probabilities, axis=1)
    ordered = np.sort(probabilities, axis=1)
    accepted = (
        (ordered[:, -1] >= threshold)
        & (ordered[:, -1] - ordered[:, -2] >= margin)
        & (features[:, 0] >= min_rms)
        & (best != background)
    )
    return np.where(accepted, best, background)


def scores(target, predicted, labels, background):
    negative = target == background
    positive = ~negative
    return {
        "windows": len(target),
        "accuracy": float(accuracy_score(target, predicted)),
        "macro_f1": float(
            f1_score(
                target, predicted, labels=np.arange(len(labels)), average="macro", zero_division=0
            )
        ),
        "false_positive_rate": float(np.mean(predicted[negative] != background)),
        "target_recall": float(np.mean(predicted[positive] == target[positive])),
        "confusion_matrix": confusion_matrix(
            target, predicted, labels=np.arange(len(labels))
        ).tolist(),
        "labels": labels,
    }


def c_array(name, array, ctype="float"):
    array = np.asarray(array)

    def value(item):
        return f"{float(item):.9e}f" if ctype == "float" else str(int(item))

    if array.ndim == 1:
        body = ", ".join(value(item) for item in array)
    else:
        body = ",\n".join("{" + ", ".join(value(item) for item in row) + "}" for row in array)
    shape = "".join(f"[{size}]" for size in array.shape)
    return f"static const {ctype} {name}{shape} = {{\n{body}\n}};"


def export_model(scaler, model, labels, settings, destination, header):
    mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    weights, bias = model.coef_.T.astype(np.float32), model.intercept_.astype(np.float32)
    feature_count = len(mean)
    arrays = {"mean": mean, "scale": scale, "weights": weights, "bias": bias}
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["scaled"]),
        helper.make_node("Gemm", ["scaled", "weights", "bias"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "music_classifier",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, len(labels)])],
        initializer=[numpy_helper.from_array(array, name) for name, array in arrays.items()],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 10
    helper.set_model_props(
        onnx_model,
        {
            "labels": json.dumps(labels),
            "dataset": settings["dataset_kind"],
            "feature_contract": f"dsp.c:16k/512/{WINDOW_FRAMES}:mean_std(log10_rms,centroid,mfcc13)+catalog",
            "catalog_id": settings.get("catalog_id") or "none",
        },
    )
    onnx.checker.check_model(onnx_model)
    destination.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, destination / "model.onnx")
    model_id = hashlib.sha256((destination / "model.onnx").read_bytes()).hexdigest()[:16]
    settings.update({"model_id": model_id, "labels": labels, "features": feature_count})
    (destination / "metadata.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "#pragma once",
        f"#define MODEL_FEATURES {feature_count}",
        f"#define MODEL_CLASSES {len(labels)}",
        f"#define MODEL_BACKGROUND_CLASS {labels.index(BACKGROUND)}",
        f'#define MODEL_ID "{model_id}"',
        f'#define MODEL_DATASET "{settings["dataset_kind"]}"',
    ]
    if settings.get("catalog_id"):
        lines.append(f"#define MODEL_CATALOG_ID 0x{settings['catalog_id']}ULL")
    for key in ["threshold", "margin", "min_log_rms"]:
        lines.append(f"#define MODEL_{key.upper()} {float(settings[key]):.9e}f")
    lines.append(
        "static const char *const MODEL_LABELS[MODEL_CLASSES] = {"
        + ", ".join(json.dumps(label) for label in labels)
        + "};"
    )
    for name, array in arrays.items():
        lines.append(c_array("MODEL_" + name.upper(), array))
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("\n".join(lines) + "\n")
    return model_id


def fit_classifier(train_x, train_y, val_x, val_y, labels):
    scaler = StandardScaler().fit(train_x)
    transformed_train = scaler.transform(train_x)
    transformed_val = scaler.transform(val_x)
    selected, search = None, []
    for weight in [None, "balanced"]:
        for regularization in [0.03, 0.3, 3.0]:
            model = LogisticRegression(
                C=regularization, max_iter=2000, class_weight=weight, random_state=11
            )
            model.fit(transformed_train, train_y)
            probabilities = model.predict_proba(transformed_val)
            candidates = []
            for threshold in [0.5, 0.65, 0.75, 0.85, 0.9, 0.95]:
                for margin in [0.1, 0.2, 0.3]:
                    predicted = decisions(
                        val_x, probabilities, labels.index(BACKGROUND), threshold, margin, -3.5
                    )
                    result = scores(val_y, predicted, labels, labels.index(BACKGROUND))
                    objective = result["macro_f1"] - 2 * result["false_positive_rate"]
                    candidates.append((objective, threshold, margin, result))
            objective, threshold, margin, validation = max(candidates, key=lambda row: row[:3])
            row = {
                "C": regularization,
                "class_weight": weight,
                "threshold": threshold,
                "margin": margin,
                "objective": objective,
                "iterations": int(model.n_iter_[0]),
                **{k: v for k, v in validation.items() if k not in {"confusion_matrix", "labels"}},
            }
            search.append(row)
            if selected is None or objective > selected[0]:
                selected = (objective, model, row, validation)
    _, model, choice, validation = selected
    return scaler, model, choice, validation, search


def train(dataset, destination=ROOT / "model", header=ROOT / "src/model_data.h"):
    labels, splits, inventory = load_dataset(dataset)
    train_x, train_y, _ = splits["train"]
    val_x, val_y, _ = splits["val"]
    scaler, model, choice, validation, search = fit_classifier(
        train_x, train_y, val_x, val_y, labels
    )
    threshold, margin = choice["threshold"], choice["margin"]
    info_path = (
        Path(dataset).with_suffix(".json")
        if Path(dataset).suffix == ".npz"
        else Path(dataset) / "dataset.json"
    )
    info = json.loads(info_path.read_text()) if info_path.exists() else {"kind": "user_recordings"}
    overlap = negative_overlap(splits)
    if overlap:
        info["split_method"] = (
            f"Independent query seeds; negative recordings overlap across splits: {overlap}."
        )
        info["limits"] = (
            f"{info.get('limits', '')} Há negativos compartilhados entre divisões: {overlap}."
        ).strip()
    kind = info.get("kind", "user_recordings")
    if kind not in {
        "user_recordings",
        "full_catalog_simulated_playback",
    }:
        raise ValueError("Tipo de dataset desconhecido")
    catalog_id = info.get("catalog_id")
    if train_x.shape[1] > FEATURE_COUNT and not catalog_id:
        catalog_id = json.loads((ROOT / "model/catalog.json").read_text())["catalog_id"]
    settings = {
        "dataset_kind": kind,
        "dataset_sha256": info.get("dataset_sha256"),
        "catalog_id": catalog_id,
        "threshold": threshold,
        "margin": margin,
        "min_log_rms": -3.5,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "training_windows": len(train_y),
        "classifier_selection": choice,
        "validation_search": search,
        "validation": validation,
        "recordings": inventory,
        "split_method": info.get("split_method", "independent_recording_files"),
        "audio_identity_verified": info.get("audio_identity_verified", False),
        "artist_tag_conflict_count": info.get("artist_tag_conflict_count"),
        "dataset_limits": info.get("limits"),
        "limitations": "Validação por janelas; confirmar eventos e latência na placa. "
        "Arquivos locais e simulações não substituem sessões independentes no INMP441.",
    }
    model_id = export_model(scaler, model, labels, settings, Path(destination), Path(header))
    print(
        json.dumps(
            {
                "model_id": model_id,
                "dataset": kind,
                "validation": {
                    k: v for k, v in validation.items() if k not in {"confusion_matrix", "labels"}
                },
            },
            indent=2,
        )
    )
    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina com sessões separadas e exporta ONNX + pesos C"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "model")
    parser.add_argument("--header", type=Path, default=ROOT / "src/model_data.h")
    args = parser.parse_args()
    train(args.data, args.output, args.header)
