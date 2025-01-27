from __future__ import annotations

import gzip
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from numpy.testing import assert_allclose, assert_equal

from autoSKZCAM.embed import CreateEmbeddedCluster
from autoSKZCAM.recipes import skzcam_analyse, dft_ensemble_analyse, get_final_autoSKZCAM_Hads

FILE_DIR = Path(__file__).parent

@pytest.fixture
def ref_EmbeddedCluster1():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "r"
    ) as file:
        EmbeddedCluster = np.load(file, allow_pickle=True).item()
    EmbeddedCluster.OniomInfo = None
    return EmbeddedCluster

def test_get_final_autoSKZCAM_Hads(ref_EmbeddedCluster1):
    OniomInfo = {
        "Extrapolated Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 5,
                "code": "orca",
            },
        },
        "Delta_Basis and Delta_Core": {
            "ll": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 3,
                "code": "orca",
            },
            "hl": {
                "method": "MP2",
                "frozen_core": "semicore",
                "basis": "CBS(TZ//QZ)",
                "max_cluster_num": 3,
                "code": "orca",
            },
        },
        "FSE Error": {
            "ll": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 5,
                "code": "orca",
            },
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 7,
                "code": "orca",
            },
        },
        "DeltaCC": {
            "ll": {
                "method": "LMP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 3,
                "code": "mrcc",
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 3,
                "code": "mrcc",
            },
        },
    }

    skzcam_int_ene = skzcam_analyse(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        OniomInfo=OniomInfo,
        EmbeddedCluster=ref_EmbeddedCluster1,
    )

    xc_ensemble = [
        "PBE-D2-Ne",
        "revPBE-D4",
        "vdW-DF",
        "rev-vdW-DF2",
        "PBE0-D4",
        "B3LYP-D2-Ne",
    ]
    vib_xc_ensemble = ["PBE-D2-Ne", "revPBE-D4", "vdW-DF", "rev-vdW-DF2"]

    dft_ensemble_analysis = dft_ensemble_analyse(
        calc_dir=Path(FILE_DIR, "mocked_vasp_runs", "dft_calc_dir"),
        xc_ensemble=xc_ensemble,
        geom_error_xc="revPBE-D4",
        vib_xc_ensemble=vib_xc_ensemble,
        freeze_surface_vib=True,
        temperature=61,
    )

    final_Hads = get_final_autoSKZCAM_Hads(
        skzcam_int_ene, dft_ensemble_analysis
    )

    ref_final_Hads = {'Extrapolated Bulk MP2': [-167.64139187528542, 0], 'Delta_Basis and Delta_Core': [-27.31886802347076, 6.153211064960163], 'FSE Error': [0, 2.9936482822501693], 'DeltaCC': [-10.131662669145438, 1.0088038275787747], 'Overall Eint': [-205.09192256790163, 6.9167638105045315], 'DFT Erlx': [8.499119999989091, 19.779237597405302], 'DFT DeltaH': [24.15531290723979, 3.081997964011643], 'Final Hads': [-172.43748966067275, 21.179201424867006]}

    for key, value in final_Hads.items():
        assert_allclose(value, ref_final_Hads[key], rtol=1e-5)