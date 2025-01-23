from __future__ import annotations

import os

import pytest
from ase import Atoms
from ase.calculators.orca import OrcaProfile
from quacc import get_settings
from quacc.calculators.mrcc.mrcc import MrccProfile

from autoSKZCAM.calculators import MRCC, ORCA, read_orca_outputs


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


def test_read_orca_outputs(tmp_path):
    # Test if it can read RI-MP2 energy from ORCA output file
    output_file = """
----------------
TOTAL SCF ENERGY
----------------

Total Energy       :         -816.16626703 Eh          -22209.01321 eV

-----------------------------------------------
 RI-MP2 CORRELATION ENERGY:     -1.541677519 Eh
-----------------------------------------------

"""

    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(output_file)

    generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    assert generated_outputs["mp2_corr_energy"] == pytest.approx(-41.95118209459777)

    # Test if it can read DLPNO-MP2 energy

    output_file = """
----------------
TOTAL SCF ENERGY
----------------

Total Energy       :        -1326.89463821 Eh          -36106.63873 eV

------------------------------------------------------
 DLPNO-MP2 CORRELATION ENERGY:      -2.150230458791 Eh
------------------------------------------------------

"""
    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(output_file)

    generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    assert generated_outputs["mp2_corr_energy"] == pytest.approx(-58.510751055514184)

    # Test if it can read DFT energy

    output_file = """
----------------
TOTAL SCF ENERGY
----------------

Total Energy       :      -4662.88849808501800 Eh         -126883.64667 eV

"""

    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(output_file)

    generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    assert generated_outputs["scf_energy"] == pytest.approx(-126883.64667)
    assert generated_outputs["energy"] == pytest.approx(-126883.64667)

    # Test if it can read DLPNO-CCSD(T) energy
    output_file = """

-------------------------
FINAL STARTUP INFORMATION
-------------------------

E(0)                                       ...   -112.752924947
E(SL-MP2)                                  ...     -0.284857102
E(SL-MP2) including corrections            ...     -0.284857102
Initial E(tot)                             ...   -113.037782049
<T|T>                                      ...      0.083969504
Number of pairs included                   ... 15
Total number of pairs                      ... 15

----------------------
COUPLED CLUSTER ENERGY
----------------------

E(0)                                       ...   -112.752924947
E(CORR)(strong-pairs)                      ...     -0.309737053
E(CORR)(weak-pairs)                        ...     -0.000048351
E(CORR)(corrected)                         ...     -0.309785403
E(TOT)                                     ...   -113.062710351
Singles Norm <S|S>**1/2                    ...      0.062770223
T1 diagnostic                              ...      0.019849687


Triples Correction (T)                     ...     -0.011884101
Final correlation energy                   ...     -0.321669504
E(CCSD)                                    ...   -113.062710351
E(CCSD(T))                                 ...   -113.074594451
"""

    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(output_file)

    # Check error if scf_energy is not provided

    with pytest.raises(ValueError, match="SCF energy not found in ORCA output file"):
        generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(
            "Total Energy       :       -112.75292494747633 Eh           -3068.16307 eV\n"
            + output_file
        )

    generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    reference_outputs = {
        "energy": -3076.916439167897,
        "scf_energy": -3068.1633661222863,
        "mp2_corr_energy": -7.751356564304554,
        "ccsd_corr_energy": -8.429690185747175,
        "ccsdt_corr_energy": -8.753073045610742,
    }

    for key, value in reference_outputs.items():
        assert generated_outputs[key] == pytest.approx(value)

    # Test if it can read RI-CCSD energy
    output_file = """
----------------
TOTAL SCF ENERGY
----------------

Total Energy       :       -112.75292494745528 Eh           -3068.16307 eV

------------------
CLOSED SHELL GUESS
------------------

Initial guess performed in     0.009 sec
E(0)                                       ...   -112.752924947
E(RI-MP2)                                  ...     -0.303454255
Initial E(tot)                             ...   -113.056379202
<T|T>                                      ...      0.100652061
Number of pairs included                   ... 15
Total number of pairs                      ... 15


------------------------------------------------
                  RHF COUPLED CLUSTER ITERATIONS
------------------------------------------------

Number of amplitudes to be optimized       ... 444620

Iter       E(tot)           E(Corr)          Delta-E          Residual     Time
  0   -113.056379202     -0.303454255      0.000000000      0.016819746   38.65
                           *** Turning on DIIS ***
  1   -113.049061046     -0.296136098      0.007318157      0.005836884   39.07
  2   -113.059198271     -0.306273323     -0.010137225      0.002824426   40.32
  3   -113.061916306     -0.308991359     -0.002718035      0.000710122   41.08
  4   -113.062735656     -0.309810708     -0.000819350      0.000269198   39.40
  5   -113.062832995     -0.309908048     -0.000097340      0.000080593   41.02
  6   -113.062847141     -0.309922193     -0.000014145      0.000038027   39.19
  7   -113.062846860     -0.309921913      0.000000280      0.000011877   39.42
  8   -113.062846573     -0.309921626      0.000000287      0.000002771   39.60
               --- The Coupled-Cluster iterations have converged ---

----------------------
COUPLED CLUSTER ENERGY
----------------------

E(0)                                       ...   -112.752924947
E(CORR)                                    ...     -0.309921626
E(TOT)                                     ...   -113.062846573
Singles Norm <S|S>**1/2                    ...      0.063401762
T1 diagnostic                              ...      0.020049398

"""

    with open(tmp_path / "mrcc.out", "w") as fd:
        fd.write(output_file)

    generated_outputs = read_orca_outputs(tmp_path / "mrcc.out")

    reference_outputs = {
        "energy": -3076.5967631240987,
        "scf_energy": -3068.1633661217134,
        "mp2_corr_energy": -8.257410873541774,
        "ccsd_corr_energy": -8.433397002385572,
        "ccsdt_corr_energy": None,
    }
