"""CPU checks of the kernel sources and lane algebra (no CUDA). Run: python3 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import voxel_kernels as vk  # noqa: E402

SHAPE = (3, 2, 4)


def _case(seed=7):
    rng = np.random.default_rng(seed)
    ke = vk.quadrature([1.0, 1.0, 1.0], 0.3)
    n_elem = int(np.prod(SHAPE))
    E = 1e-6 + (1 - 1e-6) * rng.random(n_elem) ** 3          # SIMP-like contrast
    u = rng.standard_normal(3 * int(np.prod(np.array(SHAPE) + 1)))
    return ke, E, u


def test_fp64_source_is_the_measured_prototype():
    assert vk.make_source("double") is vk.PROTOTYPE_SOURCE


def test_fp32_source_is_a_pure_type_substitution():
    src = vk.make_source("float")
    assert "double" not in src
    assert "void modal8_fp32(" in src and "void dense8_fp32(" in src
    back = src.replace("float", "double").replace("modal8_fp32(", "modal8(").replace("dense8_fp32(", "dense8(")
    assert back == vk.PROTOTYPE_SOURCE


@pytest.mark.parametrize("path", ["modal8", "dense8"])
def test_fp64_lane_emulation_matches_assembled_product(path):
    ke, E, u = _case()
    ref = vk.assembled_reference(SHAPE, ke, E, u)
    y = vk.emulate(path, SHAPE, ke, E, u, np.float64)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-13


@pytest.mark.parametrize("path", ["modal8", "dense8"])
def test_fp32_lane_emulation_accuracy(path):
    ke, E, u = _case()
    ref = vk.assembled_reference(SHAPE, ke, E, u)
    y = vk.emulate(path, SHAPE, ke, E, u, np.float32)
    assert y.dtype == np.float32
    rel = np.linalg.norm(y.astype(np.float64) - ref) / np.linalg.norm(ref)
    assert rel < 1e-5, rel     # FP32 unit roundoff 6e-8; tolerance for the GPU protocol is set from this


def test_wrong_shuffle_partner_is_detected():
    """Negative control: the emulator must be sensitive to the kernel's shuffle partners."""
    ke, E, u = _case()
    ref = vk.assembled_reference(SHAPE, ke, E, u)
    original = vk._transform
    try:
        vk._transform = lambda v: v            # drop the butterfly entirely
        y = vk.emulate("modal8", SHAPE, ke, E, u)
    finally:
        vk._transform = original
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) > 1e-2
