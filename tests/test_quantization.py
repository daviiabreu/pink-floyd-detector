import json

import numpy as np
import onnx
import pytest

from tools.common import ROOT, native
from tools.quantize import (
    PRECISIONS,
    batch_predict,
    cpu_session,
    pack_int4,
    quantize_weights,
)


def test_int4_packing_preserves_sign_and_odd_length():
    original = np.array([-8, -7, -1, 0, 1, 7, 3], dtype=np.int8)
    packed = pack_int4(original)
    decoded = []
    for index in range(len(original)):
        nibble = (int(packed[index // 2]) >> (4 * (index % 2))) & 15
        decoded.append(nibble - 16 if nibble >= 8 else nibble)
    np.testing.assert_array_equal(original, decoded)
    assert packed.nbytes == 4
    with pytest.raises(ValueError):
        pack_int4(np.array([8]))


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_weight_quantization_bounds_error_by_half_a_step(bits):
    weights = np.random.default_rng(17).normal(size=(30, 5)).astype(np.float32)
    weights[:, 0] = 0
    quantized, scale = quantize_weights(weights, bits)
    assert np.isfinite(scale).all() and (scale > 0).all()
    assert np.all(np.abs(quantized.astype(np.int64)) <= 2 ** (bits - 1) - 1)
    assert np.all(np.abs(quantized * scale - weights) <= scale / 2 + 1e-6)


@pytest.mark.parametrize("mode", list(PRECISIONS))
def test_every_precision_matches_onnx_and_handles_out_of_range_inputs(mode):
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    baseline = onnx.load(ROOT / "model/model.onnx")
    arrays = {x.name: onnx.numpy_helper.to_array(x) for x in baseline.graph.initializer}
    rng = np.random.default_rng(99)
    standardized = np.concatenate(
        (
            rng.normal(size=(100, metadata["features"])),
            rng.uniform(-100, 100, (20, metadata["features"])),
        )
    )
    features = (standardized * arrays["scale"] + arrays["mean"]).astype(np.float32)
    path = ROOT / "model/model.onnx" if mode == "fp32" else ROOT / f"model/variants/{mode}.onnx"
    onnx.checker.check_model(str(path))
    if mode != "fp32":
        properties = {item.key: item.value for item in onnx.load(path).metadata_props}
        assert properties["base_model_id"] == metadata["model_id"], (
            "Execute tools.quantize após treinar"
        )
    lib = native(with_model=True, precision=PRECISIONS[mode])
    actual = batch_predict(lib, features, len(metadata["labels"]))
    expected = cpu_session(path).run(None, {"features": features})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    invalid = features[:1].copy()
    invalid[0, 0] = np.nan
    assert np.isnan(batch_predict(lib, invalid, len(metadata["labels"]))).all()
