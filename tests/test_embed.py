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

from autoSKZCAM.embed import CreateSKZCAMClusters, _get_atom_distances
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


# def test_autoSKZCAMPrepare_write_inputs(skzcam_clusters_output, ref_oniom_layers):
#     prep_cluster = autoSKZCAMPrepare(
#         adsorbate_slab_embedded_cluster=skzcam_clusters_output[
#             "adsorbate_slab_embedded_cluster"
#         ],
#         quantum_cluster_indices_set=skzcam_clusters_output[
#             "quantum_cluster_indices_set"
#         ],
#         ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
#         oniom_layers=ref_oniom_layers,
#     )

#     skzcam_cluster_calculators = prep_cluster.create_cluster_calcs()
#     # print(skzcam_cluster_calculators)
#     prep_cluster.write_inputs(skzcam_cluster_calculators, input_dir="./calculations")


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

    # Test the write_ecp
    skzcam_clusters.run_skzcam(
        shell_max=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
        write_include_ecp=True,
    )

    skzcam_cluster_xyz = read(tmp_path / "SKZCAM_cluster_0.xyz")
    assert len(skzcam_cluster_xyz) == 21


def test_get_atom_distances():
    # Creating a H2 molecule as an Atoms object
    h2_molecule = Atoms("H2", positions=[(0, 0, 0), (0, 0, 2)])

    # Run _get_atom_distances function to get distance of h2 molecule atoms from a center position
    atom_distances = _get_atom_distances(atoms=h2_molecule, center_position=[2, 0, 0])

    assert_allclose(atom_distances, np.array([2.0, 2.82842712]), rtol=1e-05, atol=1e-07)
