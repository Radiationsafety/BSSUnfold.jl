"""
Compatibility helpers для проверки эквивалентности Python и Julia реализаций.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a, b):
    """Косинусная мера сходства между двумя векторами."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def relative_difference(a, b):
    """Относительная разница ||a-b||/||a||."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


def assert_spectra_equivalent(python_spectrum, julia_spectrum,
                               cos_threshold=0.95, rel_threshold=0.1):
    """Проверить, что два спектра численно эквивалентны."""
    cos = cosine_similarity(python_spectrum, julia_spectrum)
    rel = relative_difference(python_spectrum, julia_spectrum)
    assert cos >= cos_threshold, (
        f"Cosine similarity {cos:.4f} below threshold {cos_threshold:.2f}"
    )
    assert rel <= rel_threshold, (
        f"Relative difference {rel:.4f} above threshold {rel_threshold:.2f}"
    )
    return True
