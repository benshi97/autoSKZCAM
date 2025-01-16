from __future__ import annotations

import os

import pytest
from ase import Atoms
from ase.calculators.orca import OrcaProfile
from quacc import get_settings
from quacc.calculators.mrcc.mrcc import MrccProfile

from autoSKZCAM.calculators import MRCC, ORCA


def test_SkzcamMrcc(tmp_path):
    calc = MRCC(
        profile=MrccProfile(command=get_settings().MRCC_CMD),
        calc="PBE",
        basis="STO-3G",
        symm="off",
        genbas="Mg:cappedECP",
        directory=tmp_path,
    )

    # Geometry input. Either like this:
    water = Atoms(
        "H2O",
        positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        calculator=calc,
    )
    energy = water.get_potential_energy()

    assert energy == pytest.approx(-2026.1497783941234)
    # Check that the file GENBAS is in tmp_path
    assert os.path.exists(tmp_path / "GENBAS")


def test_SkzcamOrca(tmp_path):
    calc = ORCA(
        profile=OrcaProfile(command=get_settings().ORCA_CMD),
        calc="PBE",
        basis="STO-3G",
        symm="off",
        pointcharges="2\n-1 0 0 0.5\n1 0 0 -0.5",
        directory=tmp_path,
    )

    # Geometry input. Either like this:
    water = Atoms(
        "H2O",
        positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        calculator=calc,
    )

    energy = water.get_potential_energy()
    assert energy == pytest.approx(-2072.5847888589374)
    # Check that the orca.pc file exists
    assert os.path.exists(tmp_path / "orca.pc")
