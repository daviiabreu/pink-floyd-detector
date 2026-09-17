import numpy as np
import pytest

from tools.common import FRAME_SIZE, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES, extract, native


def reference(pcm):
    frames = pcm.astype(np.float64).reshape(WINDOW_FRAMES, FRAME_SIZE) / 32768
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frames = frames / np.maximum(rms[:, None], 1e-6) * np.hanning(FRAME_SIZE)
    power = np.abs(np.fft.rfft(frames)) ** 2 / FRAME_SIZE**2
    hz = np.fft.rfftfreq(FRAME_SIZE, 1 / SAMPLE_RATE)
    magnitude = np.sqrt(power)
    centroid = (magnitude * hz).sum(axis=1) / np.maximum(magnitude.sum(axis=1), 1e-10)
    edges = 700 * (
        10
        ** (np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 7600 / 700), 28) / 2595)
        - 1
    )
    filters = np.array(
        [
            np.maximum(
                0, np.minimum((hz - left) / (center - left), (right - hz) / (right - center))
            )
            for left, center, right in zip(edges, edges[1:], edges[2:])
        ]
    )
    mel = np.log(np.maximum(power @ filters.T, 1e-10))
    dct = np.cos(np.pi * np.arange(13)[:, None] * (np.arange(26) + 0.5) / 26)
    dct *= np.sqrt(np.array([1] + [2] * 12)[:, None] / 26)
    features = np.column_stack((np.log10(np.maximum(rms, 1e-6)), centroid, mel @ dct.T))
    return np.concatenate((features.mean(axis=0), features.std(axis=0)))


@pytest.mark.parametrize("kind", ["silence", "dc", "sine", "noise", "clipped"])
def test_features_match_independent_numpy_reference(kind):
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    rng = np.random.default_rng(42)
    signals = {
        "silence": np.zeros(len(t)),
        "dc": np.full(len(t), 0.3),
        "sine": 0.25 * np.sin(2 * np.pi * 1000 * t),
        "noise": rng.normal(0, 0.15, len(t)),
        "clipped": np.clip(2 * np.sin(2 * np.pi * 333 * t), -1, 1),
    }
    pcm = np.round(signals[kind] * 32767).astype(np.int16)
    actual = extract(pcm, native())
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference(pcm), atol=0.03, rtol=0.002)
    if kind == "sine":
        assert actual[1] == pytest.approx(1000, abs=1)
        assert actual[0] == pytest.approx(np.log10(0.25 / np.sqrt(2)), abs=0.001)


def test_wrong_window_length_rejected():
    with pytest.raises(ValueError):
        extract(np.zeros(512, dtype=np.int16))
