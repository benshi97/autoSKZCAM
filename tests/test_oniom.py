from __future__ import annotations

import gzip
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autoSKZCAM.oniom import Prepare, _is_valid_cbs_format

FILE_DIR = Path(__file__).parent


@pytest.fixture
def adsorbate_slab_embedded_cluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "adsorbate_slab_embedded_cluster.npy.gz"), "r"
    ) as file:
        return np.load(file, allow_pickle=True).item()["atoms"]


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


def test_Prepare_init(skzcam_clusters_output, ref_oniom_layers, element_info):
    prep_cluster = Prepare(
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
        prep_cluster = Prepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=ref_oniom_layers,
            multiplicities={"adsorbate_slab": 3, "adsorbate": 1},
        )

    # Check everything is read correctly when non-default multiplicities and capped_ecp are provided
    prep_cluster = Prepare(
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
        prep_cluster = Prepare(
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
        prep_cluster = Prepare(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=[[0]],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=ref_oniom_layers,
        )

    # Check if errors are raised if parameters in oniom_layers are not provided

    for parameter in ["method", "max_cluster_num", "basis", "frozen_core"]:
        wrong_oniom_layers = deepcopy(ref_oniom_layers)
        wrong_oniom_layers["Base"]["hl"].pop(parameter)
        with pytest.raises(
            ValueError,
            match=f"The {parameter} parameter must be provided for all ONIOM layers specified.",
        ):
            prep_cluster = Prepare(
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
            prep_cluster = Prepare(
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
        prep_cluster = Prepare(
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
        prep_cluster = Prepare(
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
        prep_cluster = Prepare(
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
        prep_cluster = Prepare(
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
            match=rf"The {basis_type} parameter must be provided in the element_info dictionary as format CBS\(X//Y\), where X and Y are the two basis sets.",
        ):
            prep_cluster = Prepare(
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
        prep_cluster = Prepare(
            skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            oniom_layers=wrong_oniom_layers,
        )


def test_Prepare_intialize_clusters(
    skzcam_clusters_output, ref_oniom_layers, element_info
):
    prep_cluster = Prepare(
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
        "genbas": "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n",
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
        "genbas": "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n",
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
        "genbas": "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n",
    }

    assert (
        calculators["adsorbate"].calc.parameters["genbas"]
        == "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n"
    )
    assert (
        calculators["adsorbate_slab"].calc.parameters["genbas"]
        == "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n"
    )
    assert (
        calculators["slab"].calc.parameters["genbas"]
        == calculators["adsorbate_slab"].calc.parameters["genbas"]
    )

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
        "pointcharges": None,
    }

    assert calculators["adsorbate"].calc.parameters["pointcharges"] is None
    assert_allclose(
        [
            float(x)
            for x in calculators["slab"].calc.parameters["pointcharges"].split()[1::50]
        ],
        [
            -2.0,
            0.0,
            2.0,
            4.22049352791,
            -2.0,
            -6.32954443328,
            -2.0,
            -4.22049352791,
            -2.0,
            -4.22049352791,
            2.0,
            -2.12018425659,
            -2.0,
            8.44098705582,
            -2.0,
            8.44098705582,
            -2.0,
            -2.11024676395,
            2.0,
            -6.32080279923,
            2.0,
            8.44098705582,
            2.0,
            40.09468851513,
            0.8,
            0.0,
            -15.1398030077,
            49.48825903601,
            -1.345733491,
            23.6698692456,
        ],
        rtol=1e-05,
        atol=1e-07,
    )
    assert (
        calculators["slab"].calc.parameters["pointcharges"]
        == calculators["adsorbate_slab"].calc.parameters["pointcharges"]
    )

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
        "pointcharges": None,
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
        "pointcharges": None,
    }


def test_Prepare_create_element_info(skzcam_clusters_output, ref_oniom_layers):
    # First for 'DZ' and 'semicore' for MRCC
    prep_cluster = Prepare(
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


def test_Prepare_is_valid_cbs_format():
    test_string_1 = "CBS(a//d)"
    test_string_2 = "CBS(/)"
    test_string_3 = "ABC(abc//def)"
    test_string_4 = "CBS(abcdef)"
    test_string_5 = "CBS( def2-SVP/C // def2-TZVPP/C)"

    assert _is_valid_cbs_format(test_string_1) == (True, "a", "d")
    assert _is_valid_cbs_format(test_string_2) == (False, None, None)
    assert _is_valid_cbs_format(test_string_3) == (False, None, None)
    assert _is_valid_cbs_format(test_string_4) == (False, None, None)
    assert _is_valid_cbs_format(test_string_5) == (True, "def2-SVP/C", "def2-TZVPP/C")


def test_Prepare_create_cluster_calcs(skzcam_clusters_output, element_info):
    custom_cbs_element_info = deepcopy(element_info)
    for element in ["C", "O", "Mg"]:
        custom_cbs_element_info[element]["basis"] = "CBS(def2-TZVPP//def2-QZVPP)"
        custom_cbs_element_info[element]["ri_scf_basis"] = (
            "CBS(def2-QZVPP-RI-JK//def2-QZVPP-RI-JK)"
        )
        custom_cbs_element_info[element]["ri_cwft_basis"] = (
            "CBS(def2-TZVPP/C//def2-QZVPP/C)"
        )

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
                "element_info": custom_cbs_element_info,
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
                "element_info": element_info,
            },
            "hl": {
                "method": "SOS MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 2,
                "code": "orca",
                "code_inputs": {"orcasimpleinput": "SOS-MP2 FSE"},
                "element_info": element_info,
            },
        },
        "DeltaCC": {
            "ll": {
                "method": "LMP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"aocd": "extra"},
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"aocd": "extra"},
            },
        },
    }

    prep_cluster = Prepare(
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

    assert {
        key1: list(value1) for key1, value1 in skzcam_cluster_calculators.items()
    } == {
        1: [
            "orca MP2 valence DZ",
            "orca MP2 valence TZ",
            "orca MP2 semicore TZ",
            "orca MP2 semicore QZ",
            "orca SOS_MP2 valence DZ",
            "mrcc LMP2 valence DZ",
            "mrcc LMP2 valence TZ",
            "mrcc LNO-CCSD(T) valence DZ",
            "mrcc LNO-CCSD(T) valence TZ",
        ],
        2: ["orca MP2 valence DZ", "orca MP2 valence TZ", "orca SOS_MP2 valence DZ"],
    }

    # Check that an ORCA calculation with default inputs is created correctly
    assert skzcam_cluster_calculators[1]["orca MP2 valence DZ"][
        "adsorbate"
    ].calc.parameters == {
        "orcasimpleinput": "TightSCF RI-MP2 TightPNO RIJCOSX DIIS",
        "orcablocks": '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 10 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/J" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n',
        "pointcharges": None,
    }

    # Check whether a custom orcasimpleinput is used correctly
    assert skzcam_cluster_calculators[2]["orca SOS_MP2 valence DZ"][
        "adsorbate"
    ].calc.parameters == {
        "orcasimpleinput": "SOS-MP2 FSE",
        "orcablocks": '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nMg:                    -2.11144262254    2.11144262254   -0.04367284424\nMg:                     2.11144262254    2.11144262254   -0.04367284424\nMg:                     2.11144262254   -2.11144262254   -0.04367284424\nMg:                    -2.11144262254   -2.11144262254   -0.04367284424\nO:                     -2.11070451449    2.11070451449   -2.14923989662\nO:                      2.11070451449    2.11070451449   -2.14923989662\nO:                      2.11070451449   -2.11070451449   -2.14923989662\nO:                     -2.11070451449   -2.11070451449   -2.14923989662\nO:                     -4.22049352791    2.11209139723    0.00772802266\nO:                     -2.11209139723    4.22049352791    0.00772802266\nO:                      2.11209139723    4.22049352791    0.00772802266\nO:                      4.22049352791    2.11209139723    0.00772802266\nO:                      4.22049352791   -2.11209139723    0.00772802266\nO:                      2.11209139723   -4.22049352791    0.00772802266\nO:                     -2.11209139723   -4.22049352791    0.00772802266\nO:                     -4.22049352791   -2.11209139723    0.00772802266\nend\nend\n',
        "pointcharges": None,
    }

    # Check the custom element_info is used correctly
    assert skzcam_cluster_calculators[1]["orca MP2 semicore TZ"][
        "adsorbate"
    ].calc.parameters == {
        "orcasimpleinput": "TightSCF RI-MP2 TightPNO RIJCOSX DIIS",
        "orcablocks": '\n%pal nprocs 8 end\n%maxcore 25000\n%method\nMethod hf\nRI on\nRunTyp Energy\nend\n%scf\nHFTyp rhf\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "def2-TZVPP" end\nNewAuxJGTO C "def2-QZVPP-RI-JK" end\nNewAuxCGTO C "def2-TZVPP/C" end\nNewGTO Mg "def2-TZVPP" end\nNewAuxJGTO Mg "def2-QZVPP-RI-JK" end\nNewAuxCGTO Mg "def2-TZVPP/C" end\nNewGTO O "def2-TZVPP" end\nNewAuxJGTO O "def2-QZVPP-RI-JK" end\nNewAuxCGTO O "def2-TZVPP/C" end\nend\n%coords\nCTyp xyz\nMult 1\nUnits angs\nCharge 0\ncoords\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg:                     0.00000000000    0.00000000000    0.00000000000\nO:                     -2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000    2.12018425659    0.00567209089\nO:                      2.12018425659    0.00000000000    0.00567209089\nO:                      0.00000000000   -2.12018425659    0.00567209089\nO:                      0.00000000000    0.00000000000   -2.14129966123\nend\nend\n',
        "pointcharges": None,
    }

    # Check that the MRCC calculations are created correctly
    assert skzcam_cluster_calculators[1]["mrcc LNO-CCSD(T) valence DZ"][
        "adsorbate"
    ].calc.parameters == {
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
        "aocd": "extra",
        "basis_sm": "special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n",
        "basis": "special\naug-cc-pVDZ\naug-cc-pVDZ\ncc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\naug-cc-pVDZ\n\n",
        "dfbasis_scf": "special\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\ndef2-QZVPP-RI-JK\n\n",
        "dfbasis_cor": "special\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\ncc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\naug-cc-pVDZ-RI\n\n",
        "ecp": "special\nnone\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n",
        "charge": "0",
        "mult": "1",
        "core": "2",
        "geom": "xyz\n8\n\nC                       0.00000000000    0.00000000000    2.00000000000\nO                       0.00000000000    0.00000000000    3.12800000000\nMg                      0.00000000000    0.00000000000    0.00000000000\nO                      -2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000    2.12018425659    0.00567209089\nO                       2.12018425659    0.00000000000    0.00567209089\nO                       0.00000000000   -2.12018425659    0.00567209089\nO                       0.00000000000    0.00000000000   -2.14129966123\n",
        "ghost": "serialno\n3,4,5,6,7,8\n\n",
        "genbas": "Mg:cappedECP\nINSERT_cappedECP\n\nMg:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\nMg:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n",
    }
