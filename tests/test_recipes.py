from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pytest

from autoSKZCAM.oniom import Prepare
from autoSKZCAM.recipes import write_inputs

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


def test_write_inputs(skzcam_clusters_output, ref_oniom_layers, tmp_path):
    prep_cluster = Prepare(
        skzcam_clusters_output["adsorbate_slab_embedded_cluster"],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        oniom_layers=ref_oniom_layers,
    )

    skzcam_cluster_calculators = prep_cluster.create_cluster_calcs()

    write_inputs(skzcam_cluster_calculators, tmp_path)

    # Initialize an empty list to store the paths
    paths = []

    for dirpath, dirnames, filenames in os.walk(tmp_path):
        # Add folder paths
        paths.extend(os.path.relpath(os.path.join(dirpath, dirname), tmp_path) for dirname in dirnames)

        # Add file paths
        paths.extend(os.path.relpath(os.path.join(dirpath, filename), tmp_path) for filename in filenames)


    # Sort the paths list
    paths = sorted(paths)
    assert paths == [
        "1",
        "1/mrcc",
        "1/mrcc/LMP2_DZ_valence",
        "1/mrcc/LMP2_DZ_valence/adsorbate",
        "1/mrcc/LMP2_DZ_valence/adsorbate/GENBAS",
        "1/mrcc/LMP2_DZ_valence/adsorbate/MINP",
        "1/mrcc/LMP2_DZ_valence/adsorbate_slab",
        "1/mrcc/LMP2_DZ_valence/adsorbate_slab/GENBAS",
        "1/mrcc/LMP2_DZ_valence/adsorbate_slab/MINP",
        "1/mrcc/LMP2_DZ_valence/slab",
        "1/mrcc/LMP2_DZ_valence/slab/GENBAS",
        "1/mrcc/LMP2_DZ_valence/slab/MINP",
        "1/mrcc/LMP2_TZ_valence",
        "1/mrcc/LMP2_TZ_valence/adsorbate",
        "1/mrcc/LMP2_TZ_valence/adsorbate/GENBAS",
        "1/mrcc/LMP2_TZ_valence/adsorbate/MINP",
        "1/mrcc/LMP2_TZ_valence/adsorbate_slab",
        "1/mrcc/LMP2_TZ_valence/adsorbate_slab/GENBAS",
        "1/mrcc/LMP2_TZ_valence/adsorbate_slab/MINP",
        "1/mrcc/LMP2_TZ_valence/slab",
        "1/mrcc/LMP2_TZ_valence/slab/GENBAS",
        "1/mrcc/LMP2_TZ_valence/slab/MINP",
        "1/mrcc/LNO-CCSD(T)_DZ_valence",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate/GENBAS",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate/MINP",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate_slab",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate_slab/GENBAS",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/adsorbate_slab/MINP",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/slab",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/slab/GENBAS",
        "1/mrcc/LNO-CCSD(T)_DZ_valence/slab/MINP",
        "1/mrcc/LNO-CCSD(T)_TZ_valence",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate/GENBAS",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate/MINP",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate_slab",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate_slab/GENBAS",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/adsorbate_slab/MINP",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/slab",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/slab/GENBAS",
        "1/mrcc/LNO-CCSD(T)_TZ_valence/slab/MINP",
        "1/orca",
        "1/orca/MP2_DZ_valence",
        "1/orca/MP2_DZ_valence/adsorbate",
        "1/orca/MP2_DZ_valence/adsorbate/orca.inp",
        "1/orca/MP2_DZ_valence/adsorbate_slab",
        "1/orca/MP2_DZ_valence/adsorbate_slab/orca.inp",
        "1/orca/MP2_DZ_valence/adsorbate_slab/orca.pc",
        "1/orca/MP2_DZ_valence/slab",
        "1/orca/MP2_DZ_valence/slab/orca.inp",
        "1/orca/MP2_DZ_valence/slab/orca.pc",
        "1/orca/MP2_QZ_semicore",
        "1/orca/MP2_QZ_semicore/adsorbate",
        "1/orca/MP2_QZ_semicore/adsorbate/orca.inp",
        "1/orca/MP2_QZ_semicore/adsorbate_slab",
        "1/orca/MP2_QZ_semicore/adsorbate_slab/orca.inp",
        "1/orca/MP2_QZ_semicore/adsorbate_slab/orca.pc",
        "1/orca/MP2_QZ_semicore/slab",
        "1/orca/MP2_QZ_semicore/slab/orca.inp",
        "1/orca/MP2_QZ_semicore/slab/orca.pc",
        "1/orca/MP2_TZ_semicore",
        "1/orca/MP2_TZ_semicore/adsorbate",
        "1/orca/MP2_TZ_semicore/adsorbate/orca.inp",
        "1/orca/MP2_TZ_semicore/adsorbate_slab",
        "1/orca/MP2_TZ_semicore/adsorbate_slab/orca.inp",
        "1/orca/MP2_TZ_semicore/adsorbate_slab/orca.pc",
        "1/orca/MP2_TZ_semicore/slab",
        "1/orca/MP2_TZ_semicore/slab/orca.inp",
        "1/orca/MP2_TZ_semicore/slab/orca.pc",
        "1/orca/MP2_TZ_valence",
        "1/orca/MP2_TZ_valence/adsorbate",
        "1/orca/MP2_TZ_valence/adsorbate/orca.inp",
        "1/orca/MP2_TZ_valence/adsorbate_slab",
        "1/orca/MP2_TZ_valence/adsorbate_slab/orca.inp",
        "1/orca/MP2_TZ_valence/adsorbate_slab/orca.pc",
        "1/orca/MP2_TZ_valence/slab",
        "1/orca/MP2_TZ_valence/slab/orca.inp",
        "1/orca/MP2_TZ_valence/slab/orca.pc",
        "2",
        "2/orca",
        "2/orca/MP2_DZ_valence",
        "2/orca/MP2_DZ_valence/adsorbate",
        "2/orca/MP2_DZ_valence/adsorbate/orca.inp",
        "2/orca/MP2_DZ_valence/adsorbate_slab",
        "2/orca/MP2_DZ_valence/adsorbate_slab/orca.inp",
        "2/orca/MP2_DZ_valence/adsorbate_slab/orca.pc",
        "2/orca/MP2_DZ_valence/slab",
        "2/orca/MP2_DZ_valence/slab/orca.inp",
        "2/orca/MP2_DZ_valence/slab/orca.pc",
        "2/orca/MP2_TZ_valence",
        "2/orca/MP2_TZ_valence/adsorbate",
        "2/orca/MP2_TZ_valence/adsorbate/orca.inp",
        "2/orca/MP2_TZ_valence/adsorbate_slab",
        "2/orca/MP2_TZ_valence/adsorbate_slab/orca.inp",
        "2/orca/MP2_TZ_valence/adsorbate_slab/orca.pc",
        "2/orca/MP2_TZ_valence/slab",
        "2/orca/MP2_TZ_valence/slab/orca.inp",
        "2/orca/MP2_TZ_valence/slab/orca.pc",
    ]
