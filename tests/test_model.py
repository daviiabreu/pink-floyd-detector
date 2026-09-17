import json

import numpy as np
import onnx
import onnxruntime as ort

from tools.common import ROOT, SAMPLE_RATE, WINDOW_SAMPLES, extract_model, native, predict


def test_onnx_and_firmware_weights_match_on_unseen_inputs():
    onnx.checker.check_model(str(ROOT / "model/model.onnx"))
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    session = ort.InferenceSession(str(ROOT / "model/model.onnx"))
    lib = native(with_model=True)
    rng = np.random.default_rng(91)
    inputs = [
        extract_model(rng.integers(-16000, 16000, WINDOW_SAMPLES, dtype=np.int16), lib)
        for _ in range(12)
    ]
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    for frequency in [196, 523.25, 1396.91]:
        signal = 4000 * np.sin(2 * np.pi * frequency * t) + rng.normal(0, 100, len(t))
        inputs.append(extract_model(np.round(signal).astype(np.int16), lib))
    batch = np.array(inputs, dtype=np.float32)
    expected = session.run(None, {"features": batch})[0]
    for vector, probabilities in zip(batch, expected, strict=True):
        actual, _ = predict(vector, len(metadata["labels"]), lib)
        np.testing.assert_allclose(actual, probabilities, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(actual.sum(), 1, atol=1e-6)


def test_silence_and_nonfinite_input_never_alert():
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    lib = native(with_model=True)
    silent = extract_model(np.zeros(WINDOW_SAMPLES, dtype=np.int16), lib)
    assert predict(silent, len(metadata["labels"]), lib)[1] == -1
    silent[5] = np.nan
    assert predict(silent, len(metadata["labels"]), lib)[1] == -1
