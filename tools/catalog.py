import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from tools.common import (
    FEATURE_COUNT,
    FINGERPRINT_FFT,
    FINGERPRINT_FRAMES,
    FLOAT_PTR,
    FRAME_SIZE,
    PCM_PTR,
    PEAK_PTR,
    ROOT,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    CatalogEvidence,
    extract,
    file_hash,
    native,
)
from tools.library import decode, normalize, slug
from tools.negatives import manifest_overlap

PROFILES = ("clean", "noise20", "room", "speaker", "noise10", "quiet")
SPLIT_SEEDS = {"train": 957381, "val": 436129, "test": 872953}
REFERENCE_STRIDE = 4
BUCKET_LIMIT = 64
SHORTLIST = 16
COARSE_WIDTH = 64


def peaks(pcm, lib=None):
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    rows = max(0, len(pcm) // FRAME_SIZE - FINGERPRINT_FFT // FRAME_SIZE + 1)
    result = np.empty((rows, 4), dtype=np.uint16)
    count = (lib or native()).fingerprint_extract(
        pcm.ctypes.data_as(PCM_PTR), len(pcm), result.ctypes.data_as(PEAK_PTR)
    )
    assert count == len(result)
    return result


def match(landmarks, songs, lib):
    landmarks = np.ascontiguousarray(landmarks, dtype=np.uint16)
    if landmarks.shape != (FINGERPRINT_FRAMES, 4):
        raise ValueError(f"O matcher espera {FINGERPRINT_FRAMES} frames com quatro picos")
    result = np.empty(2 * songs, dtype=np.float32)
    lib.catalog_match(landmarks.ctypes.data_as(PEAK_PTR), result.ctypes.data_as(FLOAT_PTR))
    return result


def match_detailed(landmarks, songs, lib):
    landmarks = np.ascontiguousarray(landmarks, dtype=np.uint16)
    if landmarks.shape != (FINGERPRINT_FRAMES, 4):
        raise ValueError(f"O matcher espera {FINGERPRINT_FRAMES} frames com quatro picos")
    result = np.empty(2 * songs, dtype=np.float32)
    evidence = (CatalogEvidence * 16)()
    count = lib.catalog_match_detailed(
        landmarks.ctypes.data_as(PEAK_PTR), result.ctypes.data_as(FLOAT_PTR), evidence
    )
    return result, evidence, count


def features(pcm, songs, lib):
    return np.concatenate((extract(pcm, lib), match(peaks(pcm, lib), songs, lib)))


def playback(pcm, profile, rng):
    x = pcm.astype(np.float64) / 32768
    if profile in {"room", "speaker"}:
        if profile == "speaker":
            x = sosfilt(butter(2, [180, 6000], btype="bandpass", fs=SAMPLE_RATE, output="sos"), x)
        original = x.copy()
        for milliseconds, amplitude in [(23, 0.25), (61, -0.14), (109, 0.09)]:
            delay = int(milliseconds * SAMPLE_RATE / 1000)
            x[delay:] += amplitude * original[:-delay]
    if profile != "clean":
        snr = 10 if profile == "noise10" else 20
        rms = np.sqrt(np.mean(x * x))
        x += rng.normal(0, rms / 10 ** (snr / 20), len(x))
    x *= 10 ** rng.uniform(-1.8, -1.5) if profile == "quiet" else 10 ** rng.uniform(-1.2, -0.1)
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


def build(source):
    inventory = json.loads((ROOT / "docs/catalogo-local.json").read_text())
    tracks = inventory["tracks"]
    titles = {normalize(t["title"]): slug(t["title"]) for t in tracks}
    if len(set(titles.values())) != len(titles):
        raise ValueError("Dois títulos diferentes geraram o mesmo rótulo; diferencie os nomes")
    labels = ["ambiente", *sorted(titles.values())]
    if len(tracks) > 256:
        raise ValueError("Este formato embarcado permite até 256 arquivos de referência")
    lib = native()
    cache = ROOT / "data/full_catalog/reference_peaks"
    cache.mkdir(parents=True, exist_ok=True)
    algorithm = hashlib.sha256(
        (ROOT / "src/dsp.c").read_bytes()
        + (ROOT / "src/fingerprint.c").read_bytes()
        + (ROOT / "src/fingerprint.h").read_bytes()
        + (ROOT / "src/dsp.h").read_bytes()
    ).hexdigest()[:16]
    arrays, references, bases = [], [], [0]
    for ref, track in enumerate(tracks):
        path = Path(source) / track["path"]
        if file_hash(path) != track["sha256"]:
            raise ValueError(f"Referência mudou desde o inventário: {path}")
        peak_file = cache / (track["sha256"] + "-" + algorithm + ".npy")
        if peak_file.exists():
            pp = np.load(peak_file, allow_pickle=False)
        else:
            pp = peaks(decode(path), lib)
            np.save(peak_file, pp)
        if not len(pp) or len(pp) + FINGERPRINT_FRAMES >= 65535:
            raise ValueError(f"Duração incompatível com offsets de 16 bits: {path}")
        p = np.sort(pp[::REFERENCE_STRIDE, :3].astype(np.uint64), axis=1)
        positions = np.arange(len(p), dtype=np.uint64) * REFERENCE_STRIDE
        keys = (p[:, 0] << 22) | (p[:, 1] << 11) | p[:, 2]
        entries = (keys << 24) | (np.uint64(ref) << 16) | positions
        arrays.append(entries[p[:, 0] > 0])
        label = titles[normalize(track["title"])]
        references.append(
            {
                "path": track["path"],
                "sha256": track["sha256"],
                "title": track["title"],
                "album": track["album"],
                "label": label,
                "class_index": labels.index(label) - 1,
                "frames": len(pp),
            }
        )
        bases.append(bases[-1] + (len(pp) + FINGERPRINT_FRAMES) // COARSE_WIDTH + 1)
        if ref % 20 == 0:
            print(f"Referências: {ref + 1}/{len(tracks)}", flush=True)
    entries = np.sort(np.concatenate(arrays))
    unfiltered = len(entries)
    _, counts = np.unique(entries >> 24, return_counts=True)
    entries = entries[np.repeat(counts <= BUCKET_LIMIT, counts)]
    contract = {
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
        "window_frames": WINDOW_FRAMES,
        "fft": FINGERPRINT_FFT,
        "reference_stride": REFERENCE_STRIDE,
        "bucket_limit": BUCKET_LIMIT,
        "shortlist": SHORTLIST,
        "coarse_width": COARSE_WIDTH,
        "frequency_jitter": 1,
        "offset_smoothing": [0, 1],
        "features_per_song": ["coverage", "prominence"],
    }
    identity = json.dumps(
        {"contract": contract, "labels": labels, "references": references}, sort_keys=True
    ).encode()
    catalog_id = hashlib.sha256(entries.astype("<u8").tobytes() + identity).hexdigest()[:16]
    scratch_bins = max(bases[-1], max(r["frames"] for r in references) + FINGERPRINT_FRAMES + 1)
    manifest = {
        "catalog_id": catalog_id,
        "sha256": hashlib.sha256(entries.astype("<u8").tobytes()).hexdigest(),
        "labels": labels,
        "references": references,
        "reference_count": len(references),
        "unique_titles": len(labels) - 1,
        "entries": len(entries),
        "unfiltered_entries": unfiltered,
        "index_bytes": len(entries) * 8,
        "coarse_bases": bases,
        "scratch_bins": scratch_bins,
        "matcher_scratch_bytes": 2 * scratch_bins
        + FINGERPRINT_FRAMES * 108 * 6
        + 2 * len(references),
        "contract": contract,
        "audio_identity_verified": False,
    }
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        f"#define CATALOG_ID 0x{catalog_id}ULL",
        f"#define CATALOG_SONGS {len(labels) - 1}",
        f"#define CATALOG_REFERENCES {len(references)}",
        f"#define CATALOG_ENTRIES {len(entries)}",
        f"#define CATALOG_BUCKET_LIMIT {BUCKET_LIMIT}",
        f"#define CATALOG_SHORTLIST {SHORTLIST}",
        f"#define CATALOG_COARSE_WIDTH {COARSE_WIDTH}",
        f"#define CATALOG_SCRATCH_BINS {scratch_bins}",
    ]
    for name, values in [
        ("COARSE_BASE", bases),
        ("REFERENCE_FRAMES", [r["frames"] for r in references]),
        ("CLASS", [r["class_index"] for r in references]),
    ]:
        lines.append(
            f"static const uint32_t CATALOG_{name}[] = {{" + ",".join(map(str, values)) + "};"
        )
    lines.append("static const uint64_t CATALOG_INDEX[CATALOG_ENTRIES] = {")
    lines.extend(
        ",".join(f"0x{int(v):016x}ULL" for v in entries[i : i + 8]) + ","
        for i in range(0, len(entries), 8)
    )
    lines.append("};")
    (ROOT / "src/catalog_data.h").write_text("\n".join(lines) + "\n")
    (ROOT / "model/catalog.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "reference_count",
                    "unique_titles",
                    "entries",
                    "index_bytes",
                    "matcher_scratch_bytes",
                ]
            }
        ),
        flush=True,
    )
    return manifest


def prepare(source, output, negatives, splits=("train", "val", "test")):
    source, output, negatives = Path(source), Path(output), Path(negatives)
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "model/catalog.json").read_text())
    negative_manifest = json.loads((ROOT / "docs/negative-music.json").read_text())
    overlap = manifest_overlap(negative_manifest["tracks"])
    if overlap:
        warnings.warn(f"Negativos compartilhados entre divisões: {overlap}", stacklevel=2)
    inventory = json.loads((ROOT / "docs/catalogo-local.json").read_text())
    reference_paths = {r["path"] for r in catalog["references"]}
    labels = catalog["labels"]
    data = {s: {"x": [], "y": [], "groups": [], "profiles": [], "offsets": []} for s in splits}
    rngs = {s: np.random.default_rng(SPLIT_SEEDS[s]) for s in splits}
    lib = native(with_catalog=True)
    records = []

    def add(split, pcm, label, group, profile, offset):
        d = data[split]
        d["x"].append(features(playback(pcm, profile, rngs[split]), len(labels) - 1, lib))
        for key, value in [
            ("y", label),
            ("groups", group),
            ("profiles", profile),
            ("offsets", offset),
        ]:
            d[key].append(value)

    tracks = [
        (source / r["path"], r, "catalog/" + r["path"], r["class_index"] + 1, list(splits))
        for r in catalog["references"]
    ]
    tracks += [
        (negatives / r["path"], r, "negative/" + r["path"], 0, [r["split"]])
        for r in negative_manifest["tracks"]
    ]
    for index, (path, track, group, label, track_splits) in enumerate(tracks):
        track_splits = [s for s in track_splits if s in data]
        if not track_splits:
            continue
        if file_hash(path) != track["sha256"]:
            raise ValueError(f"Áudio mudou desde o manifesto: {path}")
        pcm = decode(path)
        records.append({"path": group, "sha256": track["sha256"], "label": labels[label]})
        for split in track_splits:
            for profile in PROFILES:
                count = {"train": 15, "val": 5, "test": 10}[split] if label else 12
                offsets = rngs[split].integers(0, len(pcm) - WINDOW_SAMPLES, size=count)
                for offset in offsets:
                    add(
                        split,
                        pcm[offset : offset + WINDOW_SAMPLES],
                        label,
                        group,
                        profile,
                        int(offset),
                    )
        if index % 10 == 0:
            print(f"Consultas: {index + 1}/{len(tracks)} — {track['title']}", flush=True)
    for split in splits:
        for i in range(180):
            noise = rngs[split].normal(0, 10 ** rngs[split].uniform(-5, -1), WINDOW_SAMPLES)
            if i < 30:
                noise[:] = 0
            if 30 <= i < 60:
                frequency = rngs[split].uniform(100, 7000)
                noise += 0.1 * np.sin(
                    2 * np.pi * frequency * np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
                )
            pcm = np.round(np.clip(noise, -1, 1) * 32767).astype(np.int16)
            add(split, pcm, 0, f"synthetic_{split}_{i}", "clean", 0)
    arrays = {
        f"{split}_{key}": np.array(values, dtype=np.float32 if key == "x" else None)
        for split, rows in data.items()
        for key, values in rows.items()
    }
    path = output / ("features.npz" if len(splits) == 3 else "development.npz")
    np.savez_compressed(path, **arrays)
    info = {
        "kind": "full_catalog_simulated_playback",
        "labels": labels,
        "features": FEATURE_COUNT + 2 * (len(labels) - 1),
        "catalog_id": catalog["catalog_id"],
        "dataset_sha256": file_hash(path),
        "recordings": records,
        "split_seeds": {s: SPLIT_SEEDS[s] for s in splits},
        "profiles": list(PROFILES),
        "split_method": f"All {len(catalog['references'])} supplied references; duplicate titles share a class; "
        "independent query seeds; "
        + (
            f"negative recordings overlap across splits: {overlap}."
            if overlap
            else "negative songs disjoint by source file."
        ),
        "audio_identity_verified": False,
        "artist_tag_conflict_count": sum(
            conflict["path"] in reference_paths for conflict in inventory["artist_tag_conflicts"]
        ),
        "limits": "Consultas simuladas às mesmas gravações completas do catálogo, sem captura pelo INMP441. "
        f"As {len(negative_manifest['tracks'])} entradas negativas seguem a seleção registrada no manifesto. "
        "Não representam todos os artistas, vozes ou ambientes reais. "
        "Rótulos dos alvos seguem os nomes dos arquivos; identidade das gravações ainda não autenticada."
        + (f" Há negativos compartilhados entre divisões: {overlap}." if overlap else ""),
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({s: len(d["y"]) for s, d in data.items()}), flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cadastra todos os arquivos e prepara a avaliação da discografia"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, default=ROOT / "data/negative_music")
    parser.add_argument("--output", type=Path, default=ROOT / "data/full_catalog")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    build(args.source)
    if not args.build_only:
        prepare(
            args.source,
            args.output,
            args.negatives,
            ("train", "val") if args.development else ("train", "val", "test"),
        )
