"""G/H summary generation must not initialize the donor solver or mutate its tree."""
import builtins
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from preset_metadata import preset_volume_fractions  # noqa: E402


@pytest.mark.parametrize("session", ["g", "h"])
def test_actual_protocol_volumes_without_solver_or_directory_mutation(monkeypatch, session):
    original_import = builtins.__import__

    def no_solver_import(name, *args, **kwargs):
        assert name != "gpu_fem" and not name.startswith("gpu_fem."), name
        return original_import(name, *args, **kwargs)

    def no_directory_mutation(*args, **kwargs):
        pytest.fail("Analysis must not create or remove donor directories")

    monkeypatch.setattr(builtins, "__import__", no_solver_import)
    monkeypatch.setattr(Path, "mkdir", no_directory_mutation)
    monkeypatch.setattr(Path, "rmdir", no_directory_mutation)
    before_path = sys.path.copy()
    module = runpy.run_path(str(ROOT / f"analysis/session_{session}_tables.py"))
    assert module["PROTOCOL_VOLFRAC"] == {
        "cantilever-216k": 0.3,
        "cantilever-512k": 0.3,
        "mbb-514k": 0.5,
        "torsion-499k": 0.25,
    }
    assert sys.path == before_path


def test_dictionary_overrides_match_preset_registry_semantics(tmp_path):
    source = tmp_path / "presets.py"
    source.write_text('''
raise RuntimeError("Preset source must never execute")
PRESETS_3D: dict = {"shared": ProblemSpec(volfrac=0.2)}
PRESETS_GPU = {
    "duplicate": ProblemSpec(volfrac=0.1),
    "duplicate": ProblemSpec(volfrac=0.3),
    "shared": ProblemSpec(volfrac=0.4),
}
PRESETS_2D = {"shared": ProblemSpec(volfrac=0.5)}
ALL_PRESETS = {**PRESETS_3D, **PRESETS_GPU, **PRESETS_2D}
''')
    assert preset_volume_fractions(source) == {"shared": 0.5, "duplicate": 0.3}


@pytest.mark.parametrize("spec", ["ProblemSpec()", "ProblemSpec(volfrac=compute())"])
def test_unsupported_volume_metadata_fails_explicitly(tmp_path, spec):
    source = tmp_path / "presets.py"
    source.write_text(f'ALL_PRESETS = {{"case": {spec}}}\n')
    with pytest.raises(ValueError):
        preset_volume_fractions(source)
