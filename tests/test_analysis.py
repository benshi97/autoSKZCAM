from __future__ import annotations

import gzip
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autoSKZCAM.analysis import analyze_calculations

FILE_DIR = Path(__file__).parent


@pytest.fixture
def ref_EmbeddedCluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "r"
    ) as file:
        return np.load(file, allow_pickle=True).item()


def test_analyze_calculations(ref_EmbeddedCluster, tmp_path):
    # Analyse the calculations
    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR,"skzcam_files", "calc_dir"), EmbeddedCluster=ref_EmbeddedCluster
    )
    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
    ]

    assert_allclose(
        int_ene_list,
        [
            -0.03284702839164311,
            -0.10002692993521123,
            -0.12020787701339941,
            -0.1411872966214105,
            -0.025912366752436355,
            -0.10258896400000594,
            -0.08071781613034545,
            -0.1302253350158935,
            -0.15567740909318672,
            -0.17364676742954543,
            -0.07717695935116353,
            -0.1377337846079172,
            -0.08969258877777975,
            -0.13729463704385125,
            -0.16452589373557203,
            -0.1787724484915998,
            -0.08868831855761528,
            -0.14426471928936735,
            -0.09523525992926807,
            -0.14081419945659945,
            -0.10027903776290259,
            -0.14356644329836854,
            -0.10442318622199309,
            -0.10807691514310136,
        ],
        rtol=1e-05,
        atol=1e-07,
    )

    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "r"
    ) as file:
        skzcam_calcs_analysis = analyze_calculations(
            calc_dir=Path("skzcam_files", "calc_dir"), embedded_cluster_path=file
        )

    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
    ]

    assert_allclose(
        int_ene_list,
        [
            -0.03284702839164311,
            -0.10002692993521123,
            -0.12020787701339941,
            -0.1411872966214105,
            -0.025912366752436355,
            -0.10258896400000594,
            -0.08071781613034545,
            -0.1302253350158935,
            -0.15567740909318672,
            -0.17364676742954543,
            -0.07717695935116353,
            -0.1377337846079172,
            -0.08969258877777975,
            -0.13729463704385125,
            -0.16452589373557203,
            -0.1787724484915998,
            -0.08868831855761528,
            -0.14426471928936735,
            -0.09523525992926807,
            -0.14081419945659945,
            -0.10027903776290259,
            -0.14356644329836854,
            -0.10442318622199309,
            -0.10807691514310136,
        ],
        rtol=1e-05,
        atol=1e-07,
    )

    # Copy calc_dir folder to tmp_path and embeded_cluster.npy.gz to tmp_path/calc_dir
    calc_dir = Path(tmp_path, "calc_dir")
    shutil.copytree(Path("skzcam_files", "calc_dir"), calc_dir)

    with (
        gzip.open(
            Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "rb"
        ) as file_in,
        open(Path(calc_dir, "embedded_cluster.npy"), "wb") as file_out,
    ):
        shutil.copyfileobj(file_in, file_out)
    skzcam_calcs_analysis = analyze_calculations(calc_dir=calc_dir)

    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
    ]

    assert_allclose(
        int_ene_list,
        [
            -0.03284702839164311,
            -0.10002692993521123,
            -0.12020787701339941,
            -0.1411872966214105,
            -0.025912366752436355,
            -0.10258896400000594,
            -0.08071781613034545,
            -0.1302253350158935,
            -0.15567740909318672,
            -0.17364676742954543,
            -0.07717695935116353,
            -0.1377337846079172,
            -0.08969258877777975,
            -0.13729463704385125,
            -0.16452589373557203,
            -0.1787724484915998,
            -0.08868831855761528,
            -0.14426471928936735,
            -0.09523525992926807,
            -0.14081419945659945,
            -0.10027903776290259,
            -0.14356644329836854,
            -0.10442318622199309,
            -0.10807691514310136,
        ],
        rtol=1e-05,
        atol=1e-07,
    )

    # Confirm error raised in both EmbeddedCluster and embedded_cluster_path are None
    with pytest.raises(
        ValueError,
        match="The embedded_cluster_path or EmbeddedCluster object must be provided.",
    ):
        analyze_calculations(calc_dir=Path("skzcam_files", "calc_dir"))

    test_EmbeddedCluster = deepcopy(ref_EmbeddedCluster)
    test_EmbeddedCluster.skzcam_calcs = None
    with pytest.raises(
        ValueError,
        match="The skzcam_calcs attribute of the EmbeddedCluster object is None.",
    ):
        analyze_calculations(
            calc_dir=Path("skzcam_files", "calc_dir"),
            EmbeddedCluster=test_EmbeddedCluster,
        )
