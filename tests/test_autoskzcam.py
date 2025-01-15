from __future__ import annotations

import gzip
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from numpy.testing import assert_allclose, assert_equal

from autoSKZCAM.autoskzcam import (
    CreateSKZCAMClusters,
    _get_atom_distances,
    autoSKZCAMPrepare,
    _is_valid_cbs_format
)
from autoSKZCAM.io import MRCCInputGenerator, ORCAInputGenerator

FILE_DIR = Path(__file__).parent


@pytest.fixture
def skzcam_clusters():
    return CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file=None,
    )


@pytest.fixture
def slab_embedded_cluster(skzcam_clusters):
    return skzcam_clusters._convert_pun_to_atoms(
        pun_file=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz")
    )


@pytest.fixture
def distance_matrix(slab_embedded_cluster):
    return slab_embedded_cluster.get_all_distances()


@pytest.fixture
def adsorbate_slab_embedded_cluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "adsorbate_slab_embedded_cluster.npy.gz"), "r"
    ) as file:
        return np.load(file, allow_pickle=True).item()["atoms"]


@pytest.fixture
def mrcc_input_generator(adsorbate_slab_embedded_cluster, element_info):
    return MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )


@pytest.fixture
def orca_input_generator(adsorbate_slab_embedded_cluster, element_info):
    pal_nprocs_block = {"nprocs": 1, "maxcore": 5000}

    method_block = {"Method": "hf", "RI": "on", "RunTyp": "Energy"}

    scf_block = {
        "HFTyp": "rhf",
        "Guess": "MORead",
        "MOInp": '"orca_svp_start.gbw"',
        "SCFMode": "Direct",
        "sthresh": "1e-6",
        "AutoTRAHIter": 60,
        "MaxIter": 1000,
    }

    ecp_info = {
        "Mg": """NewECP
N_core 0
lmax f
s 1
1      1.732000000   14.676000000 2
p 1
1      1.115000000    5.175700000 2
d 1
1      1.203000000   -1.816000000 2
f 1
1      1.000000000    0.000000000 2
end"""
    }
    return ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )


@pytest.fixture
def element_info():
    return {
        "C": {
            "basis": "aug-cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVDZ/C",
        },
        "O": {
            "basis": "aug-cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/JK",
            "ri_cwft_basis": "aug-cc-pVDZ/C",
        },
        "Mg": {
            "basis": "cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "cc-pVDZ/C",
        },
    }


@pytest.fixture
def skzcam_clusters_output(adsorbate_slab_embedded_cluster):
    return {
        "adsorbate_slab_embedded_cluster": adsorbate_slab_embedded_cluster,
        "quantum_cluster_indices_set": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16,
                17,
                18,
                19,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ],
        ],
        "ecp_region_indices_set": [
            [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            [
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                24,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
            ],
        ],
    }


@pytest.fixture
def skzcam_clusters_output(adsorbate_slab_embedded_cluster):
    return {
        "adsorbate_slab_embedded_cluster": adsorbate_slab_embedded_cluster,
        "quantum_cluster_indices_set": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16,
                17,
                18,
                19,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ],
        ],
        "ecp_region_indices_set": [
            [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            [
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                24,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
            ],
        ],
    }


@pytest.fixture
def ref_oniom_layers():
    return {
        "Base": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 2,
                "code": "orca",
            },
        },
        "Delta_Basis and Delta_Core": {
            "ll": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "orca",
            },
            "hl": {
                "method": "MP2",
                "frozen_core": "semicore",
                "basis": "CBS(TZ//QZ)",
                "max_cluster_num": 1,
                "code": "orca",
            },
        },
        "FSE Error": {
            "ll": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "orca",
            },
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 2,
                "code": "orca",
            },
        },
        "DeltaCC": {
            "ll": {
                "method": "LMP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
            },
        },
    }


def test_autoSKZCAMPrepare_init(skzcam_clusters_output, ref_oniom_layers, element_info):
    prep_cluster = autoSKZCAMPrepare(
        skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=ref_oniom_layers,
    )

    assert prep_cluster.oniom_layers == ref_oniom_layers
    assert (
        prep_cluster.adsorbate_slab_embedded_cluster
        == skzcam_clusters_output["adsorbate_slab_embedded_cluster"]
    )
    assert (
        prep_cluster.quantum_cluster_indices_set
        == skzcam_clusters_output["quantum_cluster_indices_set"]
    )
    assert (
        prep_cluster.ecp_region_indices_set
        == skzcam_clusters_output["ecp_region_indices_set"]
    )
    assert prep_cluster.max_cluster == 2

    assert prep_cluster.multiplicities == {
        "adsorbate_slab": 1,
        "adsorbate": 1,
        "slab": 1,
    }
    assert prep_cluster.capped_ecp == {
        "orca": """NewECP
N_core 0
lmax f
s 1
1      1.732000000   14.676000000 2
p 1
1      1.115000000    5.175700000 2
d 1
1      1.203000000   -1.816000000 2
f 1
1      1.000000000    0.000000000 2
end""",
        "mrcc": """
*
    NCORE = 12    LMAX = 3
f
    0.000000000  2     1.000000000
s-f
   14.676000000  2     1.732000000
p-f
    5.175700000  2     1.115000000
d-f
   -1.816000000  2     1.203000000
*""",
    }
    # Check if errors raised when multiplicities is provided wrongly
    with pytest.raises(
        ValueError,
        match="The multiplicities must be provided for all three structures: adsorbate_slab, adsorbate, and slab.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=ref_oniom_layers,
            multiplicities={"adsorbate_slab": 3, "adsorbate": 1},
        )

    # Check everything is read correctly when non-default multiplicities and capped_ecp are provided
    prep_cluster = autoSKZCAMPrepare(
        skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=ref_oniom_layers,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
        capped_ecp={"MRCC": "capped_ecp_mrcc", "ORCA": "capped_ecp_orca"},
    )

    assert prep_cluster.multiplicities == {
        "adsorbate_slab": 3,
        "adsorbate": 1,
        "slab": 2,
    }
    assert prep_cluster.capped_ecp == {
        "mrcc": "capped_ecp_mrcc",
        "orca": "capped_ecp_orca",
    }

    # Check if errors are raised if capped_ecp are not provided correctly
    wrong_capped_ecp = {"asdf": "wrong"}
    with pytest.raises(
        ValueError,
        match="The keys in the capped_ecp dictionary must be either 'mrcc' or 'orca' in the corresponding code format.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=ref_oniom_layers,
            capped_ecp=wrong_capped_ecp,
        )

    # Check if errors are raised if length of quantum_cluster_indices_set is different from ecp_region_indices_set
    with pytest.raises(
        ValueError,
        match="The quantum_cluster_indices_set and ecp_region_indices_set must be the same length.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=[[0]],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=ref_oniom_layers,
        )

    # Check if errors are raised if parameters in oniom_layers are not provided
    for parameter in ["method", "max_cluster_num", "basis", "frozen_core"]:
        with pytest.raises(
            ValueError,
            match=f"The {parameter} parameter must be provided for all ONIOM layers specified.",
        ):
            wrong_oniom_layers = deepcopy(ref_oniom_layers)
            wrong_oniom_layers["Base"]["hl"].pop(parameter)
            prep_cluster = autoSKZCAMPrepare(
                skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
                quantum_cluster_indices_set=skzcam_clusters_output[
                    "quantum_cluster_indices_set"
                ],
                ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
                oniom_layers=wrong_oniom_layers,
            )

    # Check if errors are raised if too large a max_cluster_num given.
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    for max_cluster_num in [0, 3]:
        wrong_oniom_layers["Base"]["hl"]["max_cluster_num"] = max_cluster_num
        with pytest.raises(
            ValueError,
            match="The maximum cluster number for all ONIOM layers must be bigger than 0 and less than or equal to the number of quantum clusters generated by autoSKZCAM.",
        ):
            prep_cluster = autoSKZCAMPrepare(
                skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
                quantum_cluster_indices_set=skzcam_clusters_output[
                    "quantum_cluster_indices_set"
                ],
                ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
                oniom_layers=wrong_oniom_layers,
            )

    # Check if errors are raise if wrong frozen_core parameters are given.
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    wrong_oniom_layers["Base"]["hl"]["frozen_core"] = "wrong"
    with pytest.raises(
        ValueError,
        match="The frozen_core must be specified as either 'valence' or 'semicore'.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )

    # Check if errors are raised when wrong code specified
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    wrong_oniom_layers["Base"]["hl"]["code"] = "wrong"
    with pytest.raises(
        ValueError, match="The code must be specified as either 'mrcc' or 'orca'."
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )

    # Check if errors are raised when wrong basis specified
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    wrong_oniom_layers["Base"]["hl"]["basis"] = "wrong"
    with pytest.raises(
        ValueError,
        match=r"The basis must be specified as either DZ, TZ, QZ, 5Z, CBS\(DZ//TZ\), CBS\(TZ//QZ\) or CBS\(QZ//5Z\)\.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )

    # Check if errors are raised when wrong element in element_info is specified
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    wrong_element_info = deepcopy(element_info)
    wrong_element_info["ABC"] = element_info["C"]
    wrong_oniom_layers["Base"]["hl"]["element_info"] = wrong_element_info
    with pytest.raises(
        ValueError,
        match="The keys in the element_info dictionary must be valid element symbols.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )

    for basis_type in ["basis", "ri_scf_basis", "ri_cwft_basis"]:
        wrong_element_info = {"C": {basis_type: "wrong"}}
        wrong_oniom_layers = deepcopy(ref_oniom_layers)
        wrong_oniom_layers["Delta_Basis and Delta_Core"]["hl"]["element_info"] = (
            wrong_element_info
        )
        with pytest.raises(
            ValueError,
            match=f"The {basis_type}"
            + r" parameter must be provided in the element_info dictionary as format CBS\(X//Y\), where X and Y are the two basis sets.",
        ):
            prep_cluster = autoSKZCAMPrepare(
                skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
                quantum_cluster_indices_set=skzcam_clusters_output[
                    "quantum_cluster_indices_set"
                ],
                ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
                oniom_layers=wrong_oniom_layers,
            )

    # Check if errors are raised when wrong code inputs are specified for ORCA
    wrong_oniom_layers = deepcopy(ref_oniom_layers)
    wrong_oniom_layers["Base"]["hl"]["code_inputs"] = {"wrong": "input"}

    with pytest.raises(
        ValueError,
        match="If the code is orca, the code_inputs dictionary can only contain the orcasimpleinput and orcablocks keys.",
    ):
        prep_cluster = autoSKZCAMPrepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )


def test_autoSKZCAMPrepare_intialize_clusters(
    skzcam_clusters_output, ref_oniom_layers, element_info
):
    prep_cluster = autoSKZCAMPrepare(
        skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=ref_oniom_layers,
    )

    oniom_layer_parameters = {
        "method": "MP2",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "mrcc",
    }

    # Confirm that the MP2 default are created correctly
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )

    assert calculators["adsorbate"].calc.parameters == {
        "calc": "DF-MP2",
        "scftype": "rhf",
        "verbosity": 3,
        "mem": "80000MB",
        "symm": "off",
        "unit": "angs",
        "scfiguess": "small",
        "scfmaxit": 1000,
        "scfalg": "locfit1",
        "basis_sm": "special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n",
        "basis": "special\naug-cc-pVDZ\naug-cc-pVDZ\ncc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\n\n",
        "dfbasis_scf": "special\ndef2/J\ndef2/JK\ndef2/J\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\n\n",
        "dfbasis_cor": "special\naug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\n\n",
        "ecp": "special\nnone\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n",
        "charge": "0",
        "mult": "1",
        "core": "2",
        "geom": "xyz\n8\n\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg                      0.00000000000    0.00000000000    0.00000000000\nO                      -2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000    2.12018425659    0.00567209089\nO                       2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000   -2.12018425659    0.00567209089\nO                       0.00000000000    0.00000000000   -2.14129966123\n",
        "ghost": "serialno\n3,4,5,6,7,8\n\n",
    }

    # Confirm that the LNO-CCSD(T) default are created correctly
    oniom_layer_parameters = {
        "method": "LNO-CCSD(T)",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "mrcc",
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )

    assert calculators["adsorbate"].calc.parameters == {
        "calc": "LNO-CCSD(T)",
        "scftype": "rhf",
        "verbosity": 3,
        "mem": "80000MB",
        "symm": "off",
        "unit": "angs",
        "scfiguess": "small",
        "scfmaxit": 1000,
        "scfalg": "locfit1",
        "lcorthr": "tight",
        "bpedo": 0.99999,
        "ccmaxit": 400,
        "usedisk": 0,
        "ccsdalg": "dfdirect",
        "ccsdthreads": 4,
        "ccsdmkl": "thr",
        "ptthreads": 4,
        "basis_sm": "special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n",
        "basis": "special\naug-cc-pVDZ\naug-cc-pVDZ\ncc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\n\n",
        "dfbasis_scf": "special\ndef2/J\ndef2/JK\ndef2/J\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\n\n",
        "dfbasis_cor": "special\naug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\n\n",
        "ecp": "special\nnone\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n",
        "charge": "0",
        "mult": "1",
        "core": "2",
        "geom": "xyz\n8\n\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg                      0.00000000000    0.00000000000    0.00000000000\nO                      -2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000    2.12018425659    0.00567209089\nO                       2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000   -2.12018425659    0.00567209089\nO                       0.00000000000    0.00000000000   -2.14129966123\n",
        "ghost": "serialno\n3,4,5,6,7,8\n\n",
    }

    # Confirm that calculations work when utilising non-default methods
    oniom_layer_parameters = {
        "method": "dRPA",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "mrcc",
        "code_inputs": {"calc": "dRPA", "scf": "uhf"},
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert calculators["adsorbate"].calc.parameters == {
        "verbosity": 3,
        "mem": "80000MB",
        "symm": "off",
        "unit": "angs",
        "scfiguess": "small",
        "calc": "dRPA",
        "scf": "uhf",
        "basis_sm": "special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n",
        "basis": "special\naug-cc-pVDZ\naug-cc-pVDZ\ncc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\n\n",
        "dfbasis_scf": "special\ndef2/J\ndef2/JK\ndef2/J\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\ndef2/JK\n\n",
        "dfbasis_cor": "special\naug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\n\n",
        "ecp": "special\nnone\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n",
        "charge": "0",
        "mult": "1",
        "core": "2",
        "geom": "xyz\n8\n\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg                      0.00000000000    0.00000000000    0.00000000000\nO                      -2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000    2.12018425659    0.00567209089\nO                       2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000   -2.12018425659    0.00567209089\nO                       0.00000000000    0.00000000000   -2.14129966123\n",
        "ghost": "serialno\n3,4,5,6,7,8\n\n",
    }

    # Now check if the ORCA calculations are created correctly
    oniom_layer_parameters = {
        "method": "MP2",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "orca",
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert calculators["adsorbate"].calc.parameters == {
        "orcasimpleinput": "TightSCF RI-MP2 TightPNO RIJCOSX DIIS",
        "orcablocks": '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n',
    }

    # Check if DLPNO-CCSD(T) and DLPNO-MP2 calculations are created correctly
    oniom_layer_parameters = {
        "method": "DLPNO-CCSD(T)",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "orca",
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert (
        calculators["adsorbate"].calc.parameters["orcasimpleinput"]
        == r"TightSCF DLPNO-CCSD(T) TightPNO RIJCOSX DIIS"
    )
    oniom_layer_parameters = {
        "method": "DLPNO-MP2",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "orca",
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert (
        calculators["adsorbate"].calc.parameters["orcasimpleinput"]
        == r"TightSCF DLPNO-MP2 TightPNO RIJCOSX DIIS"
    )

    # Check that custom code inputs are used correctly
    oniom_layer_parameters = {
        "method": "CEPA",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "orca",
        "code_inputs": {
            "orcasimpleinput": "CEPA",
            "orcablocks": "%scf\nHFTyp uhf\nend",
        },
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert calculators["adsorbate"].calc.parameters == {
        "orcasimpleinput": "CEPA",
        "orcablocks": '%scf\nHFTyp uhf\nend\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n',
    }

    oniom_layer_parameters = {
        "method": "CEPA",
        "frozen_core": "valence",
        "basis": "DZ",
        "max_cluster_num": 2,
        "code": "orca",
        "code_inputs": {"orcasimpleinput": "CEPA"},
    }
    calculators = prep_cluster.initialize_calculator(
        oniom_layer_parameters=oniom_layer_parameters,
        quantum_cluster_indices=skzcam_clusters_output["quantum_cluster_indices_set"][
            0
        ],
        ecp_region_indices=skzcam_clusters_output["ecp_region_indices_set"][0],
        element_info=element_info,
    )
    assert calculators["adsorbate"].calc.parameters == {
        "orcasimpleinput": "CEPA",
        "orcablocks": '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n',
    }


def test_autoSKZCAMPrepare_create_element_info(
    skzcam_clusters_output, ref_oniom_layers
):
    # First for 'DZ' and 'semicore' for MRCC
    prep_cluster = autoSKZCAMPrepare(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=ref_oniom_layers,
    )

    element_info = prep_cluster.create_element_info(
        basis="DZ", frozen_core="semicore", code="mrcc", ecp={}
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": None,
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": None,
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "cc-pwCVDZ",
            "ecp": None,
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "cc-pwCVDZ-RI",
        },
    }

    # Then for 'QZ' and 'valence' for ORCA
    element_info = prep_cluster.create_element_info(
        basis="QZ", frozen_core="valence", code="orca", ecp={}
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVQZ",
            "ecp": None,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVQZ/C",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVQZ",
            "ecp": None,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVQZ/C",
        },
        "Mg": {
            "core": 10,
            "basis": "cc-pVQZ",
            "ecp": None,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "cc-pVQZ/C",
        },
    }

    # Specifying ecp for MRCC
    element_info = prep_cluster.create_element_info(
        basis="DZ", frozen_core="semicore", code="mrcc", ecp={"Mg": "ECP10SDF"}
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": None,
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": None,
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "cc-pwCVDZ",
            "ecp": "ECP10SDF",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "cc-pwCVDZ-RI",
        },
    }

def test_autoSKZCAMPrepare_is_valid_cbs_format(skzcam_clusters_output,ref_oniom_layers):
    
    test_string_1 = "CBS(a//d)"
    test_string_2 = "CBS(/)"
    test_string_3 = "ABC(abc//def)"
    test_string_4 = "CBS(abcdef)"
    test_string_5 = "CBS( def2-SVP/C // def2-TZVPP/C)"

    assert _is_valid_cbs_format(test_string_1) == (True,'a','d')
    assert _is_valid_cbs_format(test_string_2) == (False, None, None)
    assert _is_valid_cbs_format(test_string_3) == (False, None, None)
    assert _is_valid_cbs_format(test_string_4) == (False, None, None)
    assert _is_valid_cbs_format(test_string_5) == (True,'def2-SVP/C','def2-TZVPP/C')



def test_autoSKZCAMPrepare_create_cluster_calcs(skzcam_clusters_output,element_info):
    custom_cbs_element_info = deepcopy(element_info)
    for element in ['C','O','Mg']:
        custom_cbs_element_info[element]['basis'] = 'CBS(def2-TZVPP//def2-QZVPP)'
        custom_cbs_element_info[element]['ri_scf_basis'] = 'CBS(def2-QZVPP-RI-JK//def2-QZVPP-RI-JK)'
        custom_cbs_element_info[element]['ri_cwft_basis'] = 'CBS(def2-TZVPP/C//def2-QZVPP/C)'

    oniom_layers = {
        "Base": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 2,
                "code": "orca",
            },
        },
        "Delta_Basis and Delta_Core": {
            "ll": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "orca",
            },
            "hl": {
                "method": "MP2",
                "frozen_core": "semicore",
                "basis": "CBS(TZ//QZ)",
                "max_cluster_num": 1,
                "code": "orca",
                "element_info": custom_cbs_element_info
            },
        },
        "FSE Error": {
            "ll": {
                "method": "SOS MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "orca",
                "code_inputs": {"orcasimpleinput": "SOS-MP2 FSE"},
                "element_info": element_info
            },
            "hl": {
                "method": "SOS MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 2,
                "code": "orca",
                "code_inputs": {"orcasimpleinput": "SOS-MP2 FSE"},
                "element_info": element_info
            },
        },
        "DeltaCC": {
            "ll": {
                "method": "LMP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"aocd": "extra"}
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"aocd": "extra"}

            },
        },
    }

    prep_cluster = autoSKZCAMPrepare(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=oniom_layers,
    )

    skzcam_cluster_calculators = prep_cluster.create_cluster_calcs()

    assert {key1: [key2 for key2 in value1] for key1,value1 in skzcam_cluster_calculators.items()} == {1: ['orca MP2 valence DZ', 'orca MP2 valence TZ', 'orca MP2 semicore TZ', 'orca MP2 semicore QZ', 'orca SOS_MP2 valence DZ', 'mrcc LMP2 valence DZ', 'mrcc LMP2 valence TZ', 'mrcc LNO-CCSD(T) valence DZ', 'mrcc LNO-CCSD(T) valence TZ'], 2: ['orca MP2 valence DZ', 'orca MP2 valence TZ', 'orca SOS_MP2 valence DZ']}

    # Check that an ORCA calculation with default inputs is created correctly
    assert skzcam_cluster_calculators[1]['orca MP2 valence DZ']['adsorbate'].calc.parameters == {'orcasimpleinput': 'TightSCF RI-MP2 TightPNO RIJCOSX DIIS', 'orcablocks': '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 10 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/J" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n'}

    # Check whether a custom orcasimpleinput is used correctly
    assert skzcam_cluster_calculators[2]['orca SOS_MP2 valence DZ']['adsorbate'].calc.parameters == {'orcasimpleinput': 'SOS-MP2 FSE', 'orcablocks': '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nMg:                    -2.11144262254    2.11144262254   -0.04367284424\nMg:                     2.11144262254    2.11144262254   -0.04367284424\nMg:                     2.11144262254   -2.11144262254   -0.04367284424\nMg:                    -2.11144262254   -2.11144262254   -0.04367284424\nO:                     -2.11070451449    2.11070451449   -2.14923989662\nO:                      2.11070451449    2.11070451449   -2.14923989662\nO:                      2.11070451449   -2.11070451449   -2.14923989662\nO:                     -2.11070451449   -2.11070451449   -2.14923989662\nO:                     -4.22049352791    2.11209139723    0.00772802266\nO:                     -2.11209139723    4.22049352791    0.00772802266\nO:                      2.11209139723    4.22049352791    0.00772802266\nO:                      4.22049352791    2.11209139723    0.00772802266\nO:                      4.22049352791   -2.11209139723    0.00772802266\nO:                      2.11209139723   -4.22049352791    0.00772802266\nO:                     -2.11209139723   -4.22049352791    0.00772802266\nO:                     -4.22049352791   -2.11209139723    0.00772802266\nend\nend\n'}

    # Check the custom element_info is used correctly
    assert skzcam_cluster_calculators[1]['orca MP2 semicore TZ']['adsorbate'].calc.parameters == {'orcasimpleinput': 'TightSCF RI-MP2 TightPNO RIJCOSX DIIS', 'orcablocks': '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "def2-TZVPP" end\nNewAuxJGTO C "def2-QZVPP-RI-JK" end\nNewAuxCGTO C "def2-TZVPP/C" end\nNewGTO Mg "def2-TZVPP" end\nNewAuxJGTO Mg "def2-QZVPP-RI-JK" end\nNewAuxCGTO Mg "def2-TZVPP/C" end\nNewGTO O "def2-TZVPP" end\nNewAuxJGTO O "def2-QZVPP-RI-JK" end\nNewAuxCGTO O "def2-TZVPP/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n'}

    # Check that the MRCC calculations are created correctly
    assert skzcam_cluster_calculators[1]['mrcc LNO-CCSD(T) valence DZ']['adsorbate'].calc.parameters == {'calc': 'LNO-CCSD(T)', 'scftype': 'rhf', 'verbosity': 3, 'mem': '80000MB', 'symm': 'off', 'unit': 'angs', 'scfiguess': 'small', 'scfmaxit': 1000, 'scfalg': 'locfit1', 'lcorthr': 'tight', 'bpedo': 0.99999, 'ccmaxit': 400, 'usedisk': 0, 'ccsdalg': 'dfdirect', 'ccsdthreads': 4, 'ccsdmkl': 'thr', 'ptthreads': 4, 'aocd': 'extra', 'basis_sm': 'special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n', 'basis': 'special\naug-cc-pVDZ\naug-cc-pVDZ\ncc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\n\n', 'dfbasis_scf': 'special\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\n\n', 'dfbasis_cor': 'special\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\ncc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\n\n', 'ecp': 'special\nnone\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n', 'charge': '0', 'mult': '1', 'core': '2', 'geom': 'xyz\n8\n\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg                      0.00000000000    0.00000000000    0.00000000000\nO                      -2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000    2.12018425659    0.00567209089\nO                       2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000   -2.12018425659    0.00567209089\nO                       0.00000000000    0.00000000000   -2.14129966123\n', 'ghost': 'serialno\n3,4,5,6,7,8\n\n'}


def test_CreateSKZCAMClusters_init():
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file="test.pun",
    )

    assert_equal(skzcam_clusters.adsorbate_indices, [0, 1])
    assert skzcam_clusters.slab_center_indices == [32]
    assert skzcam_clusters.atom_oxi_states == {"Mg": 2.0, "O": -2.0}
    assert skzcam_clusters.adsorbate_slab_file == Path(
        FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"
    )
    assert skzcam_clusters.pun_file == "test.pun"

    # Check if error raised if adsorbate_indices and slab_center_indices overlap
    with pytest.raises(
        ValueError, match="The adsorbate and slab center indices cannot be the same."
    ):
        skzcam_clusters = CreateSKZCAMClusters(
            adsorbate_indices=[0, 1],
            slab_center_indices=[0],
            atom_oxi_states={"Mg": 2.0, "O": -2.0},
            adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
            pun_file="test.pun",
        )

    # Check if error raised if both adsorbate_slab_file and pun_file are None
    with pytest.raises(
        ValueError, match="Either the adsorbate_slab_file or pun_file must be provided."
    ):
        skzcam_clusters = CreateSKZCAMClusters(
            adsorbate_indices=[0, 1],
            slab_center_indices=[32],
            atom_oxi_states={"Mg": 2.0, "O": -2.0},
            adsorbate_slab_file=None,
            pun_file=None,
        )


def test_CreateSKZCAMClusters_run_chemshell(skzcam_clusters, tmp_path):
    # Test if xyz file doesn't get written when write_xyz_file=False
    skzcam_clusters_nowrite = deepcopy(skzcam_clusters)
    skzcam_clusters_nowrite.convert_slab_to_atoms()
    skzcam_clusters_nowrite.run_chemshell(
        filepath=tmp_path / "ChemShell_Cluster.pun",
        chemsh_radius_active=5.0,
        chemsh_radius_cluster=10.0,
        write_xyz_file=False,
    )
    assert not os.path.isfile(tmp_path / "ChemShell_Cluster.xyz")

    skzcam_clusters.convert_slab_to_atoms()
    skzcam_clusters.run_chemshell(
        filepath=tmp_path / "ChemShell_Cluster.pun",
        chemsh_radius_active=5.0,
        chemsh_radius_cluster=10.0,
        write_xyz_file=True,
    )

    # Read the output .xyz file
    chemshell_embedded_cluster = read(tmp_path / "ChemShell_Cluster.xyz")

    # Check that the positions and atomic numbers match reference
    assert_allclose(
        chemshell_embedded_cluster.get_positions()[::100],
        np.array(
            [
                [0.00000000e00, 0.00000000e00, -7.72802046e-03],
                [-2.11024616e00, 2.11024616e00, -6.38586825e00],
                [6.33073849e00, -2.11024616e00, -6.38586825e00],
                [-1.09499282e01, -4.53560876e00, 4.95687508e00],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        chemshell_embedded_cluster.get_atomic_numbers()[::40].tolist(),
        [12, 12, 12, 8, 8, 8, 12, 9, 9],
    )


def test_CreateSKZCAMClusters_convert_pun_to_atoms(skzcam_clusters):
    slab_embedded_cluster = skzcam_clusters._convert_pun_to_atoms(
        pun_file=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz")
    )

    # Check that number of atoms matches our reference
    assert len(slab_embedded_cluster) == 390

    # Check that last 10 elements of the oxi_state match our reference
    assert_allclose(
        slab_embedded_cluster.get_array("oxi_states")[-10:],
        np.array(
            [
                -0.80812511,
                2.14427889,
                -0.96000248,
                2.14427887,
                -0.8081251,
                2.10472993,
                -0.89052904,
                2.10472993,
                -0.8081251,
                2.14427887,
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check that first 10 elements of atom_type array match our reference
    assert_equal(
        slab_embedded_cluster.get_array("atom_type")[:10],
        [
            "cation",
            "anion",
            "anion",
            "anion",
            "anion",
            "anion",
            "cation",
            "cation",
            "cation",
            "cation",
        ],
    )

    # Check that the positions of the atom matches
    assert_allclose(
        slab_embedded_cluster[200].position,
        np.array([6.33074029, -2.11024676, -6.37814205]),
        rtol=1e-05,
        atol=1e-07,
    )


def test_CreateSKZCAMClusters_convert_slab_to_atoms():
    # Test for CO on MgO example
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file=None,
    )
    skzcam_clusters.convert_slab_to_atoms()

    # Check adsorbate matches reference
    assert_allclose(
        skzcam_clusters.adsorbate.get_positions(),
        np.array([[0.0, 0.0, 2.44102236], [0.0, 0.0, 3.58784217]]),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(skzcam_clusters.adsorbate.get_atomic_numbers().tolist(), [6, 8])

    # Check slab matches reference
    assert_allclose(
        skzcam_clusters.slab.get_positions()[::10],
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [-2.11024616e00, 0.00000000e00, -6.37814023e00],
                [2.11024616e00, 2.11024616e00, -4.26789407e00],
                [2.10705227e00, 0.00000000e00, -2.14155146e00],
                [-4.22049233e00, -2.11024616e00, -4.26789407e00],
                [0.00000000e00, -4.22049233e00, -6.37814023e00],
                [0.00000000e00, -2.12018365e00, 5.67208927e-03],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.slab.get_atomic_numbers().tolist()[::10],
        [12, 12, 12, 12, 8, 8, 8],
    )

    # Check center_position matches reference
    assert_allclose(
        skzcam_clusters.center_position,
        np.array([0.0, 0.0, 3.09607306]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check vector distance of adsorbate from first center atom (corresponding to first atom index) of slab matches reference
    assert_allclose(
        skzcam_clusters.adsorbate_vector_from_slab,
        np.array([0.0, 0.0, 2.44102236]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Test for NO on MgO example
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32, 33],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "NO_MgO.poscar.gz"),
        pun_file=None,
    )
    skzcam_clusters.convert_slab_to_atoms()

    # Check adsorbate matches reference
    assert_allclose(
        skzcam_clusters.adsorbate.get_positions(),
        np.array(
            [[1.18932285, -0.14368533, 2.0777825], [1.68437633, 0.62999818, 2.83068075]]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(skzcam_clusters.adsorbate.get_atomic_numbers().tolist(), [7, 8])

    # Check slab matches reference
    assert_allclose(
        skzcam_clusters.slab.get_positions()[::10],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [-4.2019821, -2.10867761, -6.39202884],
                [0.01851023, -4.21892378, -4.28178268],
                [0.01903204, -2.105465, -2.15224877],
                [-4.2019821, -2.10867761, -4.28178268],
                [0.01851023, -4.21892378, -6.39202884],
                [0.01900061, -2.11652633, -0.03021786],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.slab.get_atomic_numbers().tolist()[::10],
        [12, 12, 12, 12, 8, 8, 8],
    )

    # Check center_position matches reference
    assert_allclose(
        skzcam_clusters.center_position,
        np.array([1.06307888, -1.06176564, 2.47922285]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check vector distance of adsorbate from first center atom (corresponding to first atom index) of slab matches reference
    assert_allclose(
        skzcam_clusters.adsorbate_vector_from_slab,
        np.array([1.18932285, -0.14368533, 2.0777825]),
        rtol=1e-05,
        atol=1e-07,
    )


def test_CreateSKZCAMClusters_find_cation_shells(
    skzcam_clusters, slab_embedded_cluster
):
    # Get distance of atoms from the center
    distances = _get_atom_distances(
        atoms=slab_embedded_cluster, center_position=[0, 0, 2]
    )

    # Find the cation shells from the distances
    cation_shells_distances, cation_shells_idx = skzcam_clusters._find_cation_shells(
        slab_embedded_cluster=slab_embedded_cluster,
        distances=distances,
        shell_width=0.005,
    )

    # As these list of lists do not have the same length, we flatten first 5 lists into a 1D list for comparison
    cation_shells_distances_flatten = [
        item for row in cation_shells_distances[:5] for item in row
    ]
    cation_shells_idx_flatten = [item for row in cation_shells_idx[:5] for item in row]

    # Check that these lists are correct
    assert_allclose(
        cation_shells_distances_flatten,
        np.array(
            [
                2.0,
                3.6184221134101624,
                3.6184221134101655,
                3.6184221134101655,
                3.6184221134101686,
                4.646732760541734,
                4.646732760541734,
                4.646732760541736,
                4.646732760541736,
                4.6888354582307805,
                4.6888354582307805,
                4.6888354582307805,
                4.6888354582307805,
                6.267895285274443,
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        cation_shells_idx_flatten, [0, 9, 8, 6, 7, 11, 12, 10, 13, 19, 21, 18, 20, 22]
    )


def test_CreateSKZCAMClusters_get_anion_coordination(
    skzcam_clusters, slab_embedded_cluster, distance_matrix
):
    # Get the anions for the second SKZCAM shell
    anion_shell_idx = skzcam_clusters._get_anion_coordination(
        slab_embedded_cluster=slab_embedded_cluster,
        cation_shell_indices=[9, 8, 6, 7],
        dist_matrix=distance_matrix,
    )

    # Check anion indices match with reference
    assert_equal(
        anion_shell_idx, [1, 2, 3, 4, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30]
    )


def test_CreateSKZCAMClusters_get_ecp_region(
    skzcam_clusters, slab_embedded_cluster, distance_matrix
):
    # Find the ECP region for the first cluster
    ecp_region_idx = skzcam_clusters._get_ecp_region(
        slab_embedded_cluster=slab_embedded_cluster,
        quantum_cluster_indices_set=[[0, 1, 2, 3, 4, 5]],
        dist_matrix=distance_matrix,
        ecp_dist=3,
    )

    # Check ECP region indices match with reference
    assert_equal(ecp_region_idx[0], [6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22])


def test_CreateSKZCAMClusters_create_adsorbate_slab_embedded_cluster(
    skzcam_clusters, slab_embedded_cluster
):
    skzcam_clusters.slab_embedded_cluster = slab_embedded_cluster
    skzcam_clusters.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    skzcam_clusters.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    skzcam_clusters._create_adsorbate_slab_embedded_cluster(
        quantum_cluster_indices_set=[[0, 1, 3, 4], [5, 6, 7, 8]],
        ecp_region_indices_set=[[0, 1, 3, 4], [5, 6, 7, 8]],
    )

    # Check that the positions of the first 10 atoms of the embedded cluster matches the reference positions, oxi_states and atom_type
    assert_allclose(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_positions()[:10],
        np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.128],
                [0.0, 0.0, 0.0],
                [-2.12018426, 0.0, 0.00567209],
                [0.0, 2.12018426, 0.00567209],
                [2.12018426, 0.0, 0.00567209],
                [0.0, -2.12018426, 0.00567209],
                [0.0, 0.0, -2.14129966],
                [-2.11144262, 2.11144262, -0.04367284],
                [2.11144262, 2.11144262, -0.04367284],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_chemical_symbols()[:10],
        ["C", "O", "Mg", "O", "O", "O", "O", "O", "Mg", "Mg"],
    )
    assert_allclose(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_array("oxi_states")[:10],
        np.array([0.0, 0.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0]),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_array("atom_type")[:10],
        [
            "adsorbate",
            "adsorbate",
            "cation",
            "anion",
            "anion",
            "anion",
            "anion",
            "anion",
            "cation",
            "cation",
        ],
    )

    # Check that the quantum_idx and ecp_idx match the reference
    assert_equal(
        skzcam_clusters.quantum_cluster_indices_set,
        [[0, 1, 2, 3, 5, 6], [0, 1, 7, 8, 9, 10]],
    )
    assert_equal(skzcam_clusters.ecp_region_indices_set, [[2, 3, 5, 6], [7, 8, 9, 10]])


def test_CreateSKZCAMClusters_run_skzcam(skzcam_clusters, tmp_path):
    # Get quantum cluster and ECP region indices
    skzcam_clusters.center_position = [0, 0, 2]
    skzcam_clusters.pun_file = Path(
        FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"
    )
    skzcam_clusters.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    skzcam_clusters.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    skzcam_clusters.run_skzcam(
        shell_max=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
    )

    # Check quantum cluster indices match with reference
    assert_equal(
        skzcam_clusters.quantum_cluster_indices_set[1],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            16,
            17,
            18,
            19,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ],
    )

    # Check ECP region indices match with reference
    assert_equal(
        skzcam_clusters.ecp_region_indices_set[1],
        [
            12,
            13,
            14,
            15,
            20,
            21,
            22,
            23,
            24,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
        ],
    )
    # Read the written output and check that it matches with the reference positions and atomic numbers
    skzcam_cluster_xyz = read(tmp_path / "SKZCAM_cluster_0.xyz")

    assert_allclose(
        skzcam_cluster_xyz.get_positions(),
        np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.128],
                [0.0, 0.0, 0.0],
                [-2.12018426, 0.0, 0.00567209],
                [0.0, 2.12018426, 0.00567209],
                [2.12018426, 0.0, 0.00567209],
                [0.0, -2.12018426, 0.00567209],
                [0.0, 0.0, -2.14129966],
            ]
        ),
        rtol=1e-04,
        atol=1e-07,
    )

    assert_equal(
        skzcam_cluster_xyz.get_atomic_numbers().tolist(), [6, 8, 12, 8, 8, 8, 8, 8]
    )


def test_get_atom_distances():
    # Creating a H2 molecule as an Atoms object
    h2_molecule = Atoms("H2", positions=[(0, 0, 0), (0, 0, 2)])

    # Run _get_atom_distances function to get distance of h2 molecule atoms from a center position
    atom_distances = _get_atom_distances(atoms=h2_molecule, center_position=[2, 0, 0])

    assert_allclose(atom_distances, np.array([2.0, 2.82842712]), rtol=1e-05, atol=1e-07)
