from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule
from quacc import change_settings

from autoSKZCAM.quacc import static_job_mrcc, static_job_orca

FILE_DIR = Path(__file__).parent


def test_static_job_mrcc(tmp_path):
    atoms = Atoms("H2O", positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with change_settings({"RESULTS_DIR": tmp_path}):
        output = static_job_mrcc(
            atoms,
            charge=0,
            mult=1,
            method="SCAN",
            basis="def2-TZVP",
            calc="PBE",
            symm="off",
        )
    assert output['molecule_metadata']["natoms"] == len(atoms)
    assert output["parameters"]["basis"] == "def2-TZVP"
    assert output["parameters"]["calc"] == "PBE"
    assert output["parameters"]["symm"] == "off"
    assert output["parameters"]["charge"] == 0
    assert output["parameters"]["mult"] == 1
    assert output["results"]["energy"] == pytest.approx(-2026.1497783941234)

    # Check if it runs without specifying anything besides atoms
    atoms = Atoms("H2O", positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with change_settings({"RESULTS_DIR": tmp_path}):
        output = static_job_mrcc(atoms)

    assert output["results"]["energy"] == pytest.approx(-2026.1497783941234)


def test_static_job_orca(tmp_path):
    atoms = molecule("H2")
    with change_settings({"RESULTS_DIR": tmp_path}):
        output = static_job_orca(
            atoms, orcasimpleinput="def2-tzvp engrad normalprint wb97x-d3bj xyzfile"
        )
    assert output['molecule_metadata']["natoms"] == len(atoms)
    assert (
        output["parameters"]["orcasimpleinput"]
        == "def2-tzvp engrad normalprint wb97x-d3bj xyzfile"
    )
