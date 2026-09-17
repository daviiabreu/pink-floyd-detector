import ctypes
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
FRAME_SIZE = 512
WINDOW_FRAMES = 64
WINDOW_SAMPLES = FRAME_SIZE * WINDOW_FRAMES
WINDOW_US = WINDOW_SAMPLES * 1_000_000 // SAMPLE_RATE
FINGERPRINT_FFT = 4096
FINGERPRINT_FRAMES = WINDOW_FRAMES - FINGERPRINT_FFT // FRAME_SIZE + 1
FEATURE_COUNT = 30
BACKGROUND = "ambiente"
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
PCM_PTR = ctypes.POINTER(ctypes.c_int16)
PEAK_PTR = ctypes.POINTER(ctypes.c_uint16)


class CatalogEvidence(ctypes.Structure):
    _fields_ = [
        ("offset_frames", ctypes.c_int32),
        ("reference", ctypes.c_uint16),
        ("class_index", ctypes.c_uint16),
        ("votes", ctypes.c_uint8),
        ("second_votes", ctypes.c_uint8),
    ]


class DecisionWindow(ctypes.Structure):
    _fields_ = [
        ("model_candidate", ctypes.c_int),
        ("sequence", ctypes.c_uint32),
        ("started_us", ctypes.c_int64),
        ("count", ctypes.c_uint),
        ("matches", CatalogEvidence * 16),
    ]


class DecisionState(ctypes.Structure):
    _fields_ = [
        ("history", DecisionWindow * 8),
        ("next", ctypes.c_uint),
        ("filled", ctypes.c_uint),
        ("support_windows", ctypes.c_uint),
        ("aligned_frames", ctypes.c_uint),
        ("first_start_us", ctypes.c_int64),
    ]


def native(with_model=False, precision=32, with_catalog=False):
    compiler = ["xcrun", "--sdk", "macosx", "clang"] if platform.system() == "Darwin" else ["cc"]
    sources = [
        ROOT / "src/dsp.c",
        ROOT / "src/decision.c",
        ROOT / "src/fingerprint.c",
        ROOT / "src/panel.c",
    ]
    if with_catalog or with_model:
        sources.append(ROOT / "src/catalog.c")
    if with_model:
        sources.append(ROOT / "src/inference.c")
    dependencies = sources + list((ROOT / "src").glob("*.h"))
    dependencies += sorted((ROOT / "model/variants").glob("*.h"))
    digest = hashlib.sha256(
        str((precision, with_model, with_catalog, compiler)).encode()
        + b"".join(p.read_bytes() for p in dependencies)
    ).hexdigest()[:16]
    build = ROOT / "build/native"
    build.mkdir(parents=True, exist_ok=True)
    library = build / f"{'model' if with_model else 'dsp'}-{digest}.so"
    if not library.exists():
        mode = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
        subprocess.run(
            [
                *compiler,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                f"-DMODEL_PRECISION={precision}",
                mode,
                *map(str, sources),
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(library))
    lib.decision_reset.argtypes = [ctypes.POINTER(DecisionState)]
    lib.decision_reset.restype = None
    lib.decision_update.argtypes = [
        ctypes.POINTER(DecisionState),
        ctypes.c_int,
        ctypes.POINTER(CatalogEvidence),
        ctypes.c_uint,
        ctypes.c_uint32,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.decision_update.restype = ctypes.c_int
    lib.extract_window.argtypes = [PCM_PTR, FLOAT_PTR]
    lib.extract_window.restype = ctypes.c_int
    lib.fingerprint_extract.argtypes = [PCM_PTR, ctypes.c_size_t, PEAK_PTR]
    lib.fingerprint_extract.restype = ctypes.c_size_t
    if with_catalog or with_model:
        lib.catalog_match.argtypes = [PEAK_PTR, FLOAT_PTR]
        lib.catalog_match.restype = None
        lib.catalog_match_detailed.argtypes = [PEAK_PTR, FLOAT_PTR, ctypes.POINTER(CatalogEvidence)]
        lib.catalog_match_detailed.restype = ctypes.c_uint
    if with_model:
        lib.model_feature_count.argtypes = []
        lib.model_feature_count.restype = ctypes.c_int
        lib.model_predict.argtypes = [FLOAT_PTR, FLOAT_PTR]
        lib.model_predict.restype = None
        lib.model_candidate.argtypes = [FLOAT_PTR, FLOAT_PTR]
        lib.model_candidate.restype = ctypes.c_int
        lib.model_predict_batch.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_size_t]
        lib.model_predict_batch.restype = None
        lib.model_benchmark.argtypes = [FLOAT_PTR, ctypes.c_size_t, ctypes.c_uint]
        lib.model_benchmark.restype = ctypes.c_float
    return lib


def extract(pcm, lib=None):
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    if pcm.shape != (WINDOW_SAMPLES,):
        raise ValueError(f"Esperadas {WINDOW_SAMPLES} amostras mono, recebidas {pcm.shape}")
    result = np.empty(FEATURE_COUNT, dtype=np.float32)
    (lib or native()).extract_window(pcm.ctypes.data_as(PCM_PTR), result.ctypes.data_as(FLOAT_PTR))
    return result


def extract_model(pcm, lib=None):
    lib = lib or native(with_model=True)
    if lib.model_feature_count() == FEATURE_COUNT:
        return extract(pcm, lib)
    from tools.catalog import features

    return features(pcm, (lib.model_feature_count() - FEATURE_COUNT) // 2, lib)


def predict(features, classes, lib):
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.shape != (lib.model_feature_count(),):
        raise ValueError("Vetor de features incompatível")
    result = np.empty(classes, dtype=np.float32)
    lib.model_predict(features.ctypes.data_as(FLOAT_PTR), result.ctypes.data_as(FLOAT_PTR))
    candidate = lib.model_candidate(
        features.ctypes.data_as(FLOAT_PTR), result.ctypes.data_as(FLOAT_PTR)
    )
    return result, candidate


def read_wav(path):
    rate, pcm = wavfile.read(path)
    if rate != SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"{path}: use WAV PCM 16-bit mono a 16000 Hz")
    if len(pcm) < WINDOW_SAMPLES:
        raise ValueError(f"{path}: áudio menor que 2,048 s")
    return pcm


def windows(pcm):
    for start in range(0, len(pcm) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        yield pcm[start : start + WINDOW_SAMPLES]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
