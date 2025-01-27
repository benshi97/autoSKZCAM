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
from autoSKZCAM.oniom import Prepare
from autoSKZCAM.recipes_skzcam import (
    skzcam_analyse,
    skzcam_calculate_job,
    skzcam_eint_flow,
    skzcam_generate_job,
    skzcam_initialize,
    skzcam_write_inputs,
)

FILE_DIR = Path(__file__).parent


@pytest.fixture
def ref_EmbeddedCluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "adsorbate_slab_embedded_cluster.npy.gz"), "r"
    ) as file:
        adsorbate_slab_embedded_cluster = np.load(file, allow_pickle=True).item()[
            "atoms"
        ]

    EmbeddedCluster = CreateEmbeddedCluster(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=None,
    )

    EmbeddedCluster.adsorbate_slab_embedded_cluster = adsorbate_slab_embedded_cluster
    EmbeddedCluster.quantum_cluster_indices_set = [
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
    ]
    EmbeddedCluster.ecp_region_indices_set = [
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
    ]
    return EmbeddedCluster


@pytest.fixture
def ref_oniom_layers():
    return {
        "Bulk MP2": {
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


@pytest.fixture
def ref_EmbeddedCluster1():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "embedded_cluster.npy.gz"), "r"
    ) as file:
        EmbeddedCluster = np.load(file, allow_pickle=True).item()
    EmbeddedCluster.OniomInfo = None
    return EmbeddedCluster


def test_skzcam_analyse(tmp_path, ref_EmbeddedCluster1):
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

    ref_skzcam_int_ene = {
        "Extrapolated Bulk MP2": [-167.64139187528542, 0],
        "Delta_Basis and Delta_Core": [-27.31886802347076, 6.153211064960163],
        "FSE Error": [0, 2.9936482822501693],
        "DeltaCC": [-10.131662669145438, 1.0088038275787747],
        "Overall Eint": [-205.09192256790163, 6.9167638105045315],
    }

    for key, value in ref_skzcam_int_ene.items():
        assert_allclose(skzcam_int_ene[key], value, rtol=1e-05, atol=1e-07)

    # Check for errors if the calc_dir does not exist
    with pytest.raises(
        ValueError,
        match="Either the EmbeddedCluster object must be provided or embedded_cluster_npy_path is set or embedded_cluster.npy is provided in calc_dir.",
    ):
        skzcam_int_ene = skzcam_analyse(
            calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
            OniomInfo=None,
            EmbeddedCluster=None,
        )

    # Check for errors if EmbeddedCluster does not contain OniomInfo
    wrong_EmbeddedCluster = deepcopy(ref_EmbeddedCluster1)
    wrong_EmbeddedCluster.OniomInfo = None
    with pytest.raises(
        ValueError,
        match="The OniomInfo dictionary must be provided in EmbeddedCluster or as an argument.",
    ):
        skzcam_int_ene = skzcam_analyse(
            calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
            OniomInfo=None,
            EmbeddedCluster=wrong_EmbeddedCluster,
        )

    ref_EmbeddedCluster2 = deepcopy(ref_EmbeddedCluster1)
    ref_EmbeddedCluster2.OniomInfo = OniomInfo
    skzcam_int_ene = skzcam_analyse(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        OniomInfo=None,
        EmbeddedCluster=ref_EmbeddedCluster2,
    )

    for key, value in ref_skzcam_int_ene.items():
        assert_allclose(skzcam_int_ene[key], value, rtol=1e-05, atol=1e-07)

    # Copy calc_dir folder to tmp_path and embeded_cluster.npy.gz to tmp_path/calc_dir
    calc_dir = Path(tmp_path, "calc_dir")
    shutil.copytree(Path(FILE_DIR, "skzcam_files", "calc_dir"), calc_dir)
    np.save(Path(calc_dir, "embedded_cluster.npy"), ref_EmbeddedCluster2)
    skzcam_int_ene = skzcam_analyse(calc_dir=calc_dir)

    for key, value in ref_skzcam_int_ene.items():
        assert_allclose(skzcam_int_ene[key], value, rtol=1e-05, atol=1e-07)

    skzcam_int_ene = skzcam_analyse(
        calc_dir=Path(FILE_DIR, "skzcam_files", "calc_dir"),
        embedded_cluster_npy_path=Path(calc_dir, "embedded_cluster.npy"),
    )

    for key, value in ref_skzcam_int_ene.items():
        assert_allclose(skzcam_int_ene[key], value, rtol=1e-05, atol=1e-07)


def test_skzcam_eint_flow(tmp_path, ref_oniom_layers):
    EmbeddedCluster = skzcam_initialize(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"),
    )
    skzcam_eint_flow(
        EmbeddedCluster=EmbeddedCluster,
        OniomInfo=ref_oniom_layers,
        calc_dir=tmp_path,
        dryrun=True,
    )

    # Get list of files in the subfolders in tmp_path
    paths = []
    for dirpath, dirnames, filenames in os.walk(tmp_path):
        # Add folder paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, dirname), tmp_path)
            for dirname in dirnames
        )

        # Add file paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, filename), tmp_path)
            for filename in filenames
        )

    paths = sorted(paths)
    assert paths == [
        "1",
        "1/mrcc",
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
        "embedded_cluster.npy",
    ]


def test_skzcam_initialize(tmp_path):
    with pytest.raises(
        ValueError,
        match="The path to the .pun file from ChemShell must be provided in EmbeddedCluster if run_chemshell is False.",
    ):
        skzcam_initialize(
            adsorbate_indices=[0, 1],
            slab_center_indices=[32],
            atom_oxi_states={"Mg": 2.0, "O": -2.0},
            adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
            pun_filepath="test.pun",
        )
    EmbeddedCluster = skzcam_initialize(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"),
    )

    assert_equal(EmbeddedCluster.adsorbate_indices, [0, 1])
    assert EmbeddedCluster.slab_center_indices == [32]
    assert EmbeddedCluster.atom_oxi_states == {"Mg": 2.0, "O": -2.0}
    assert EmbeddedCluster.adsorbate_slab_file == Path(
        FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"
    )
    assert EmbeddedCluster.pun_filepath == Path(
        FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"
    )

    # Check adsorbate matches reference
    assert_allclose(
        EmbeddedCluster.adsorbate.get_positions(),
        np.array([[0.0, 0.0, 2.44102236], [0.0, 0.0, 3.58784217]]),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(EmbeddedCluster.adsorbate.get_atomic_numbers().tolist(), [6, 8])

    # Check slab matches reference
    assert_allclose(
        EmbeddedCluster.slab.get_positions()[::10],
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
        EmbeddedCluster.slab.get_atomic_numbers().tolist()[::10],
        [12, 12, 12, 12, 8, 8, 8],
    )

    # Check center_position matches reference
    assert_allclose(
        EmbeddedCluster.center_position,
        np.array([0.0, 0.0, 3.09607306]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check vector distance of adsorbate from first center atom (corresponding to first atom index) of slab matches reference
    assert_allclose(
        EmbeddedCluster.adsorbate_vector_from_slab,
        np.array([0.0, 0.0, 2.44102236]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Run ChemShell
    EmbeddedCluster = skzcam_initialize(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=tmp_path / "ChemShell_Cluster.pun",
        run_chemshell=True,
        chemsh_radius_active=5.0,
        chemsh_radius_cluster=10.0,
        write_xyz_file=True,
    )

    # Check that Path(tmp_path, "ChemShell_Cluster.pun") exists
    assert Path(tmp_path, "ChemShell_Cluster.pun").exists()

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


def test_skzcam_generate_job(tmp_path):
    # Confirm that everything works as expected
    EmbeddedCluster = CreateEmbeddedCluster(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"),
    )

    # Get quantum cluster and ECP region indices
    EmbeddedCluster.center_position = [0, 0, 2]
    EmbeddedCluster.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    EmbeddedCluster.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    EmbeddedCluster.pun_filepath = None

    with pytest.raises(
        ValueError,
        match="The path pun_filepath to the .pun file from ChemShell must be provided in EmbeddedCluster.",
    ):
        skzcam_generate_job(
            EmbeddedCluster=EmbeddedCluster,
            max_cluster_num=2,
            ecp_dist=3.0,
            shell_width=0.005,
            write_clusters=True,
            write_clusters_path=tmp_path,
        )

    EmbeddedCluster.pun_filepath = Path(
        FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"
    )

    skzcam_generate_job(
        EmbeddedCluster=EmbeddedCluster,
        max_cluster_num=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
    )

    assert hasattr(EmbeddedCluster, "adsorbate_slab")

    EmbeddedCluster = skzcam_initialize(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"),
    )

    # Get quantum cluster and ECP region indices
    EmbeddedCluster.center_position = [0, 0, 2]
    EmbeddedCluster.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    EmbeddedCluster.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    EmbeddedCluster.pun_filepath = Path(
        FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"
    )

    skzcam_generate_job(
        EmbeddedCluster=EmbeddedCluster,
        max_cluster_num=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
    )

    # Check quantum cluster indices match with reference
    assert_equal(
        EmbeddedCluster.quantum_cluster_indices_set[1],
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
        EmbeddedCluster.ecp_region_indices_set[1],
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


def test_skzcam_calculate_job(tmp_path):
    EmbeddedCluster = skzcam_initialize(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_filepath=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"),
    )

    # Get quantum cluster and ECP region indices
    EmbeddedCluster.center_position = [0, 0, 2]
    EmbeddedCluster.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    EmbeddedCluster.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    skzcam_generate_job(
        EmbeddedCluster=EmbeddedCluster,
        max_cluster_num=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
    )

    oniom_layers = {
        "Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "orca",
                "code_inputs": {"orcablocks": "%pal nprocs 2 end\n%maxcore 1000\n"},
            },
        }
    }

    skzcam_calculate_job(
        EmbeddedCluster=EmbeddedCluster,
        OniomInfo=oniom_layers,
        dryrun=False,
        calc_dir=tmp_path,
    )

    # Check that "****ORCA TERMINATED NORMALLY****" is in the output file
    with open(
        tmp_path / "1" / "orca" / "MP2_DZ_valence" / "adsorbate" / "orca.out"
    ) as f:
        assert "****ORCA TERMINATED NORMALLY****" in f.read()

    # Check whether quacc works as well
    skzcam_calculate_job(
        EmbeddedCluster=EmbeddedCluster,
        OniomInfo=oniom_layers,
        dryrun=False,
        use_quacc=True,
        calc_dir=Path(tmp_path, "quacc"),
    )

    # Check that "****ORCA TERMINATED NORMALLY****" is in the output file
    with open(
        tmp_path / "quacc" / "1" / "orca" / "MP2_DZ_valence" / "adsorbate" / "orca.out"
    ) as f:
        assert "****ORCA TERMINATED NORMALLY****" in f.read()

    oniom_layers = {
        "Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "mrcc",
            },
        }
    }
    skzcam_calculate_job(
        EmbeddedCluster=EmbeddedCluster,
        OniomInfo=oniom_layers,
        dryrun=False,
        use_quacc=True,
        calc_dir=Path(tmp_path, "quacc"),
    )

    # Check that "****ORCA TERMINATED NORMALLY****" is in the output file
    with open(
        tmp_path / "quacc" / "1" / "mrcc" / "MP2_DZ_valence" / "adsorbate" / "mrcc.out"
    ) as f:
        assert "Normal termination of mrcc" in f.read()

    oniom_layers = {
        "Bulk MP2": {
            "ll": None,
            "hl": {
                "method": "MP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "orca",
                "code_inputs": {"orcablocks": "%pal nprocs 2 end\n%maxcore 1000\n"},
            },
        },
        "DeltaCC": {
            "ll": {
                "method": "LMP2",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"mem": "1000MB"},
            },
            "hl": {
                "method": "LNO-CCSD(T)",
                "frozen_core": "valence",
                "basis": "DZ",
                "max_cluster_num": 1,
                "code": "mrcc",
                "code_inputs": {"mem": "1000MB"},
            },
        },
    }

    skzcam_calculate_job(
        EmbeddedCluster=EmbeddedCluster,
        OniomInfo=oniom_layers,
        dryrun=True,
        calc_dir=Path(tmp_path, "inputs"),
    )

    # Initialize an empty list to store the paths
    paths = []

    for dirpath, dirnames, filenames in os.walk(Path(tmp_path, "inputs")):
        # Add folder paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, dirname), Path(tmp_path, "inputs"))
            for dirname in dirnames
        )

        # Add file paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, filename), Path(tmp_path, "inputs"))
            for filename in filenames
        )

    # Sort the paths list
    paths = sorted(paths)
    assert paths == [
        "1",
        "1/mrcc",
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
        "embedded_cluster.npy",
    ]


def test_skzcam_write_inputs(ref_EmbeddedCluster, ref_oniom_layers, tmp_path):
    Prepare(
        EmbeddedCluster=ref_EmbeddedCluster, OniomInfo=ref_oniom_layers
    ).create_cluster_calcs()

    skzcam_write_inputs(ref_EmbeddedCluster, tmp_path)

    # Initialize an empty list to store the paths
    paths = []

    for dirpath, dirnames, filenames in os.walk(tmp_path):
        # Add folder paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, dirname), tmp_path)
            for dirname in dirnames
        )

        # Add file paths
        paths.extend(
            os.path.relpath(os.path.join(dirpath, filename), tmp_path)
            for filename in filenames
        )

    # Sort the paths list
    paths = sorted(paths)
    assert paths == [
        "1",
        "1/mrcc",
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
        "embedded_cluster.npy",
    ]

    wrong_EmbeddedCluster = deepcopy(ref_EmbeddedCluster)
    wrong_EmbeddedCluster.skzcam_calcs = None

    with pytest.raises(
        ValueError,
        match="The EmbeddedCluster object must have the skzcam_calcs attribute set using oniom.Prepare.",
    ):
        skzcam_write_inputs(wrong_EmbeddedCluster, tmp_path)
