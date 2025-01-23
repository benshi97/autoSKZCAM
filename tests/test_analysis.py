from __future__ import annotations

import gzip
import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autoSKZCAM.analysis import (
    _get_method_int_ene,
    analyze_calculations,
    compute_skzcam_int_ene,
    extrapolate_to_bulk,
    get_cbs_extrapolation,
)

FILE_DIR = Path(__file__).parent


@pytest.fixture
def ref_EmbeddedCluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "r"
    ) as file:
        return np.load(file, allow_pickle=True).item()


def test_compute_skzcam_int_ene(ref_EmbeddedCluster):
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

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    skzcam_int_ene = compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )

    ref_skzcam_int_ene = {
        "Extrapolated Bulk MP2": [-0.16764139187528543, 0],
        "Delta_Basis and Delta_Core": [-0.02731886802347076, 0.0061532110649601636],
        "FSE Error": [0, 0.002993648282250169],
        "DeltaCC": [-0.010131662669145438, 0.0010088038275787748],
        "final": [-0.20509192256790162, 0.006916763810504532],
    }

    for key, value in ref_skzcam_int_ene.items():
        assert_allclose(skzcam_int_ene[key], value, rtol=1e-05, atol=1e-07)

    # Check whether def2 case is covered
    OniomInfo = {
        "Extrapolated Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 5,
                "code": "orca",
                "element_info": {
                    element: {"basis": "CBS(def2-SVP//def2-TZVPP)"}
                    for element in ["C", "Mg", "O"]
                },
            },
        }
    }

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )

    OniomInfo = {
        "DeltaCC": {
            "ll": {
                "method": "LNO-CCSD",
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
        }
    }

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    skzcam_int_ene = compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )
    assert_allclose(
        skzcam_int_ene["final"],
        [-0.03847792236925364, 0.01283915060710039],
        rtol=1e-05,
        atol=1e-07,
    )

    # Check if error raised when method that isn't MP2, CCSD or CCSD(T) is used
    OniomInfo = {
        "DeltaCC": {
            "ll": {
                "method": "RPA",
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
        }
    }

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )

    with pytest.raises(
        ValueError, match="The method cannot be analysed automatically."
    ):
        skzcam_int_ene = compute_skzcam_int_ene(
            skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
        )

    # Check if non-extrapolated version of bulk is also called
    OniomInfo = {
        "Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 5,
                "code": "orca",
            },
        }
    }

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    skzcam_int_ene = compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )
    assert skzcam_int_ene["final"][0] == pytest.approx(
        skzcam_calcs_analysis[5]["orca MP2 valence DZ"]["int_ene"]["energy"]
    )

    # Check if no error bar provided when Delta is less than 3
    OniomInfo = {
        "DeltaCC": {
            "ll": {
                "method": "LNO-CCSD",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 2,
                "code": "mrcc",
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "CBS(DZ//TZ)",
                "max_cluster_num": 2,
                "code": "mrcc",
            },
        }
    }

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    skzcam_int_ene = compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )
    assert skzcam_int_ene["final"][1] == pytest.approx(0)


def test_get_method_int_ene():
    int_ene_dict = {
        "energy": 10.0,
        "scf_energy": 1.0,
        "mp2_corr_energy": 2.0,
        "ccsd_corr_energy": 3.0,
        "ccsdt_corr_energy": None,
    }

    assert _get_method_int_ene(int_ene_dict, "mp2") == 2.0
    assert _get_method_int_ene(int_ene_dict, "ccsd") == 3.0
    assert _get_method_int_ene(int_ene_dict, "scf") == 1.0

    with pytest.raises(ValueError, match="The energy is None."):
        _get_method_int_ene(int_ene_dict, "ccsdt")


def test_extrapolate_to_bulk():
    bulk_int_ene = extrapolate_to_bulk(
        [6, 22, 34, 42, 58, 82, 100],
        [
            -0.03284702839164311,
            -0.08071781613034545,
            -0.08969258877777975,
            -0.09523525992926807,
            -0.10027903776290259,
            -0.10442318622199309,
            -0.10807691514310136,
        ],
    )

    assert bulk_int_ene == pytest.approx(-0.10723242133702465)


def test_get_cbs_extrapolation(ref_EmbeddedCluster):
    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )

    cbs_extrapolated_ene = get_cbs_extrapolation(
        0.05710643952079408,
        -0.0899534679109486,
        0.027170816450961865,
        -0.12719774638527426,
        X_size="DZ",
        Y_size="TZ",
        family="mixcc",
    )

    assert_allclose(
        (0.017185301080196724, -0.1486152208466108, -0.13142991976641408),
        cbs_extrapolated_ene,
        rtol=1e-05,
        atol=1e-07,
    )

    with pytest.raises(
        ValueError, match=re.escape("The cardinal number of Y does not equal X+1")
    ):
        get_cbs_extrapolation(
            0.05710643952079408,
            -0.0899534679109486,
            0.027170816450961865,
            -0.12719774638527426,
            family="mixcc",
            X_size="TZ",
            Y_size="DZ",
        )


def test_analyze_calculations(ref_EmbeddedCluster, tmp_path):
    # Analyse the calculations
    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        EmbeddedCluster=ref_EmbeddedCluster,
    )
    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
        if calculation_label != "cluster_size"
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
            calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
            embedded_cluster_path=file,
        )

    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
        if calculation_label != "cluster_size"
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
    shutil.copytree(Path(FILE_DIR, "skzcam_files", "calc_dir"), calc_dir)

    with (
        gzip.open(
            Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "rb"
        ) as file_in,
        open(Path(FILE_DIR, calc_dir, "embedded_cluster.npy"), "wb") as file_out,
    ):
        shutil.copyfileobj(file_in, file_out)
    skzcam_calcs_analysis = analyze_calculations(calc_dir=calc_dir)

    int_ene_list = [
        skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["energy"]
        for cluster_num in skzcam_calcs_analysis
        for calculation_label in skzcam_calcs_analysis[cluster_num]
        if calculation_label != "cluster_size"
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
        analyze_calculations(calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"))

    test_EmbeddedCluster = deepcopy(ref_EmbeddedCluster)
    test_EmbeddedCluster.skzcam_calcs = None
    with pytest.raises(
        ValueError,
        match="The skzcam_calcs attribute of the EmbeddedCluster object is None.",
    ):
        analyze_calculations(
            calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
            EmbeddedCluster=test_EmbeddedCluster,
        )
