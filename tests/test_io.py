from __future__ import annotations

import gzip
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import compare_atoms
from numpy.testing import assert_allclose, assert_equal

from autoSKZCAM.autoskzcam import CreateSKZCAMClusters
from autoSKZCAM.io import (
    MRCCInputGenerator,
    ORCAInputGenerator,
    create_atom_coord_string,
)

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


def test_MRCCInputGenerator_init(adsorbate_slab_embedded_cluster, element_info):
    # Check what happens if multiplicities is not provided
    mrcc_input_generator = MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
    )

    assert mrcc_input_generator.multiplicities == {
        "adsorbate_slab": 1,
        "adsorbate": 1,
        "slab": 1,
    }

    mrcc_input_generator = MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )

    assert not compare_atoms(
        mrcc_input_generator.adsorbate_slab_embedded_cluster,
        adsorbate_slab_embedded_cluster,
    )
    assert_equal(mrcc_input_generator.quantum_cluster_indices, [0, 1, 2, 3, 4, 5, 6, 7])
    assert_equal(mrcc_input_generator.adsorbate_indices, [0, 1])
    assert_equal(mrcc_input_generator.slab_indices, [2, 3, 4, 5, 6, 7])
    assert_equal(
        mrcc_input_generator.ecp_region_indices,
        [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
    )
    assert mrcc_input_generator.element_info == element_info
    assert mrcc_input_generator.include_cp is True
    assert mrcc_input_generator.multiplicities == {
        "adsorbate_slab": 3,
        "adsorbate": 1,
        "slab": 2,
    }

    # Check if error raise if quantum_cluster_indices and ecp_region_indices overlap

    with pytest.raises(
        ValueError, match="An atom in the quantum cluster is also in the ECP region."
    ):
        mrcc_input_generator = MRCCInputGenerator(
            adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
            quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            ecp_region_indices=[7, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            element_info=element_info,
            include_cp=True,
        )


def test_MRCCInputGenerator_generate_input(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    input_dict = mrcc_input_generator_nocp.generate_input()

    # Check that the input_dictionary is correct
    assert (
        input_dict["adsorbate"]["geom"].split()[1],
        input_dict["slab"]["geom"].split()[1],
        input_dict["adsorbate_slab"]["geom"].split()[1],
    ) == ("2", "19", "21")

    mrcc_input_generator.generate_input()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=special", "def2/JK", "cappedECP"],
        },
        "adsorbate": {"float": [8.0], "string": ["basis_sm=special", "C"]},
        "slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=special", "def2/JK", "cappedECP"],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=special", "def2/JK", "cappedECP"],
        },
        "adsorbate": {"float": [2.0], "string": ["basis_sm=special"]},
        "slab": {
            "float": [
                19.0,
                -4.22049352791,
                4.22049352791,
                4.22049352791,
                2.11024676395,
                -0.0,
            ],
            "string": ["basis_sm=special", "no-basis-set", "charge=-8"],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::300]
        generated_block_collated[system]["string"] = [
            x
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::50]

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::300]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::50]

        assert_equal(
            generated_block_collated[system]["string"],
            reference_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        assert_equal(
            generated_block_nocp_collated[system]["string"],
            reference_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_MRCCInputGenerator_generate_basis_ecp_block(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)
    mrcc_input_generator_ecp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    mrcc_input_generator_nocp._generate_basis_ecp_block()

    mrcc_input_generator._generate_basis_ecp_block()

    reference_mrcc_blocks_collated = {
        "adsorbate_slab": [
            "basis_sm=special",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "cappedECP",
        ],
        "slab": [
            "basis_sm=special",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "cappedECP",
        ],
        "adsorbate": [
            "basis_sm=special",
            "aug-cc-pVDZ",
            "def2/JK",
            "cc-pVDZ/C",
            "none",
        ],
    }

    reference_mrcc_blocks_nocp_collated = {
        "adsorbate_slab": [
            "basis_sm=special",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "cappedECP",
        ],
        "slab": [
            "basis_sm=special",
            "no-basis-set",
            "basis=special",
            "no-basis-set",
            "dfbasis_scf=special",
            "no-basis-set",
            "dfbasis_cor=special",
            "no-basis-set",
            "ecp=special",
            "cappedECP",
        ],
        "adsorbate": ["basis_sm=special", "aug-cc-pVDZ/C"],
    }

    generated_mrcc_blocks_nocp_collated = {
        system: [] for system in ["adsorbate_slab", "slab", "adsorbate"]
    }
    generated_mrcc_blocks_collated = {
        system: [] for system in ["adsorbate_slab", "slab", "adsorbate"]
    }
    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_mrcc_blocks_collated[system] = mrcc_input_generator.skzcam_input_str[
            system
        ].split()[::10]
        generated_mrcc_blocks_nocp_collated[system] = (
            mrcc_input_generator_nocp.skzcam_input_str[system].split()[::10]
        )

        assert_equal(
            generated_mrcc_blocks_collated[system],
            reference_mrcc_blocks_collated[system],
        )
        assert_equal(
            generated_mrcc_blocks_nocp_collated[system],
            reference_mrcc_blocks_nocp_collated[system],
        )

    # Test if atom ecps are added
    # Check the case if the element_info has all of the same values
    element_info = {
        "C": {
            "basis": "def2-SVP",
            "core": 2,
            "ecp": "ccECP",
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "O": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "Mg": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
    }
    mrcc_input_generator_ecp.element_info = element_info
    mrcc_input_generator_ecp._generate_basis_ecp_block()
    mrcc_input_generator_ecp.include_cp = False

    assert (
        mrcc_input_generator_ecp.skzcam_input_str["adsorbate"]
        == "\nbasis_sm=special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n\nbasis=special\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\ndef2-SVP\n\n\ndfbasis_scf=special\ndef2/J\ndef2/J\ndef2/J\ndef2/J\ndef2/J\ndef2/J\ndef2/J\ndef2/J\n\n\ndfbasis_cor=special\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\ndef2-SVP/C\n\n\necp=special\nccECP\nnone\nnone\nnone\nnone\nnone\nnone\nnone\n\n"
    )


def test_MRCCInputGenerator_create_atomtype_basis(mrcc_input_generator):
    generated_basis_block_without_ecp = mrcc_input_generator._create_atomtype_basis(
        quantum_region=mrcc_input_generator.adsorbate_slab_cluster,
        element_basis_info={
            element: mrcc_input_generator.element_info[element]["ri_cwft_basis"]
            for element in mrcc_input_generator.element_info
        },
    )
    generated_basis_block_with_ecp = mrcc_input_generator._create_atomtype_basis(
        quantum_region=mrcc_input_generator.adsorbate_slab_cluster,
        element_basis_info={
            element: mrcc_input_generator.element_info[element]["ri_cwft_basis"]
            for element in mrcc_input_generator.element_info
        },
        ecp_region=mrcc_input_generator.ecp_region,
    )

    reference_basis_block_without_ecp = "aug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\n"
    reference_basis_block_with_ecp = "aug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\n"

    assert generated_basis_block_without_ecp == reference_basis_block_without_ecp
    assert generated_basis_block_with_ecp == reference_basis_block_with_ecp


def test_MRCCInputGenerator_generate_coords_block(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    mrcc_input_generator_nocp._generate_coords_block()

    mrcc_input_generator._generate_coords_block()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
        "adsorbate": {
            "float": [8.0, -2.12018425659, -2.12018425659],
            "string": ["charge=0", "C", "O"],
        },
        "slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
        "adsorbate": {"float": [2.0], "string": ["charge=0", "C"]},
        "slab": {
            "float": [19.0, 2.12018425659, 2.11144262254, -0.04367284424, 0.0, 0.0],
            "string": ["charge=-8", "Mg", "O", "Mg", "Mg"],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::10]
        generated_block_collated[system]["string"] = [
            x
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::5]

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::10]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::5]

        assert_equal(
            generated_block_collated[system]["string"],
            reference_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        assert_equal(
            generated_block_nocp_collated[system]["string"],
            reference_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_MRCCInputGenerator_generate_point_charge_block(mrcc_input_generator):
    generated_point_charge_block = mrcc_input_generator._generate_point_charge_block()

    generated_point_charge_block_shortened = [
        float(x) for x in generated_point_charge_block.split()[5::180]
    ]

    reference_point_charge_block_shortened = [
        -0.04367284424,
        -0.03992370948,
        -2.14923989662,
        -6.37814204923,
        -2.1415520695,
        -4.26789528527,
        -2.1415520695,
        -0.03992370948,
        0.0,
    ]

    assert_allclose(
        generated_point_charge_block_shortened,
        reference_point_charge_block_shortened,
        rtol=1e-05,
        atol=1e-07,
    )


def test_ORCAInputGenerator_init(adsorbate_slab_embedded_cluster, element_info):
    orca_input_generator = ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
    )

    # Check when multiplicities is not provided
    assert orca_input_generator.multiplicities == {
        "adsorbate_slab": 1,
        "adsorbate": 1,
        "slab": 1,
    }

    orca_input_generator = ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )

    assert not compare_atoms(
        orca_input_generator.adsorbate_slab_embedded_cluster,
        adsorbate_slab_embedded_cluster,
    )
    assert_equal(orca_input_generator.quantum_cluster_indices, [0, 1, 2, 3, 4, 5, 6, 7])
    assert_equal(orca_input_generator.adsorbate_indices, [0, 1])
    assert_equal(orca_input_generator.slab_indices, [2, 3, 4, 5, 6, 7])
    assert_equal(
        orca_input_generator.ecp_region_indices,
        [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
    )
    assert orca_input_generator.element_info == element_info
    assert orca_input_generator.include_cp is True
    assert orca_input_generator.multiplicities == {
        "adsorbate_slab": 3,
        "adsorbate": 1,
        "slab": 2,
    }

    # Check if error raise if quantum_cluster_indices and ecp_region_indices overlap

    with pytest.raises(
        ValueError, match="An atom in the quantum cluster is also in the ECP region."
    ):
        orca_input_generator = ORCAInputGenerator(
            adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
            quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            ecp_region_indices=[7, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            element_info=element_info,
            include_cp=True,
            multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
        )


def test_ORCAInputGenerator_generate_input(orca_input_generator):
    orca_input_generator_nocp = deepcopy(orca_input_generator)

    orca_input_generator_nocp.include_cp = False
    orca_input_generator_nocp.generate_input()

    orca_input_generator.generate_input()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [2.0, 2.0],
            "string": [
                "%pointcharges",
                "end",
                "NewAuxCGTO",
                "%coords",
                "cappedECP",
                "Mg>",
            ],
        },
        "adsorbate": {"float": [2.0], "string": ["%method", "C", '"cc-pVDZ/C"', "xyz"]},
        "slab": {
            "float": [2.0, 2.0],
            "string": [
                "%pointcharges",
                "end",
                "NewAuxCGTO",
                "%coords",
                "cappedECP",
                "Mg>",
            ],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [2.0, 2.0],
            "string": [
                "%pointcharges",
                "end",
                "NewAuxCGTO",
                "%coords",
                "cappedECP",
                "Mg>",
            ],
        },
        "adsorbate": {"float": [2.0], "string": ["%method", "C", '"cc-pVDZ/C"', "xyz"]},
        "slab": {
            "float": [2.0],
            "string": [
                "%pointcharges",
                "end",
                "NewAuxCGTO",
                "%coords",
                "cappedECP",
                "Mg>",
            ],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::77]
        generated_block_collated[system]["string"] = [
            x
            for x in orca_input_generator.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::17]
        assert_equal(
            reference_block_collated[system]["string"],
            generated_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::77]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::17]

        assert_equal(
            reference_block_nocp_collated[system]["string"],
            generated_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_create_atom_coord_string(adsorbate_slab_embedded_cluster):
    atom = adsorbate_slab_embedded_cluster[0]

    # First let's try the case where it's a normal atom.
    atom_coord_string = create_atom_coord_string(atom=atom)

    with pytest.raises(ValueError, match="Capped ECP cannot be a ghost atom."):
        create_atom_coord_string(atom, is_capped_ecp=True, is_ghost_atom=True)

    with pytest.raises(
        ValueError, match="Point charge value must be given for atoms with ECP info."
    ):
        create_atom_coord_string(atom, is_capped_ecp=True)

    assert (
        atom_coord_string
        == "C                       0.00000000000    0.00000000000    2.00000000000\n"
    )

    # Let's now try the case where it is a ghost atom.
    atom_coord_string = create_atom_coord_string(atom=atom, is_ghost_atom=True)
    assert (
        atom_coord_string
        == "C:                      0.00000000000    0.00000000000    2.00000000000\n"
    )

    # Let's now try the case where it is an atom in the ECP region.
    atom_coord_string = create_atom_coord_string(
        atom=atom, is_capped_ecp=True, pc_charge=2.0
    )
    assert (
        atom_coord_string
        == "C>     2.00000000000    0.00000000000    0.00000000000    2.00000000000\ncappedECP\n"
    )


def test_ORCAInputGenerator_generate_coords_block(orca_input_generator):
    orca_input_generator_nocp = deepcopy(orca_input_generator)

    orca_input_generator_nocp.include_cp = False
    orca_input_generator_nocp._generate_coords_block()

    orca_input_generator._generate_coords_block()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [3.0, 0.0, 2.11144262254, 2.0, -2.10705287155, 2.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "cappedECP",
                "Mg>",
                "cappedECP",
                "end",
            ],
        },
        "adsorbate": {"float": [1.0, 0.0], "string": ["%coords", "coords", "O:"]},
        "slab": {
            "float": [2.0, 0.0, 2.11144262254, 2.0, -2.10705287155, 2.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "cappedECP",
                "Mg>",
                "cappedECP",
                "end",
            ],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [3.0, 0.0, 2.11144262254, 2.0, -2.10705287155, 2.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "cappedECP",
                "Mg>",
                "cappedECP",
                "end",
            ],
        },
        "adsorbate": {"float": [1.0], "string": ["%coords", "coords"]},
        "slab": {
            "float": [2.0, 0.0, 2.0, 2.10705287155, 2.0, 0.0],
            "string": ["%coords", "coords", "Mg>", "cappedECP", "Mg>", "cappedECP"],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::14]
        generated_block_collated[system]["string"] = [
            x
            for x in orca_input_generator.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::7]

        assert_equal(
            reference_block_collated[system]["string"],
            generated_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::14]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::7]

        assert_equal(
            reference_block_nocp_collated[system]["string"],
            generated_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_ORCAInputGenerator_generate_preamble_block(orca_input_generator):
    # Make copy of orca_input_generator for further tests
    orca_input_generator_1 = deepcopy(orca_input_generator)
    orca_input_generator_2 = deepcopy(orca_input_generator)
    orca_input_generator_3 = deepcopy(orca_input_generator)

    # Generate the orca input preamble
    orca_input_generator_1._generate_preamble_block()

    assert (
        orca_input_generator_1.orcablocks["adsorbate_slab"]
        == '%pointcharges "orca.pc"\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n'
    )

    assert (
        orca_input_generator_1.orcablocks["adsorbate"]
        == '%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewGTO Mg "cc-pVDZ" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n'
    )
    assert (
        orca_input_generator_1.orcablocks["adsorbate_slab"]
        == orca_input_generator_1.orcablocks["slab"]
    )

    # Check the case if the element_info has all of the same values
    element_info = {
        "C": {
            "basis": "def2-SVP",
            "core": 2,
            "ecp": "ccECP",
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "O": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "Mg": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
    }
    orca_input_generator_2.element_info = element_info
    orca_input_generator_2._generate_preamble_block()

    assert (
        orca_input_generator_2.orcablocks["adsorbate_slab"]
        == '%pointcharges "orca.pc"\n%method\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "def2-SVP" end\nNewAuxJGTO C "def2/J" end\nNewAuxCGTO C "def2-SVP/C" end\nNewECP C "ccECP" end\nNewGTO Mg "def2-SVP" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxCGTO Mg "def2-SVP/C" end\nNewGTO O "def2-SVP" end\nNewAuxJGTO O "def2/J" end\nNewAuxCGTO O "def2-SVP/C" end\nend\n'
    )

    # Check whether error raised if not all element_info is provided
    element_info_error = {"C": element_info["C"]}
    orca_input_generator_3.element_info = element_info_error
    with pytest.raises(
        ValueError,
        match="Not all element symbols are provided in the element_info dictionary.",
    ):
        orca_input_generator_3._generate_preamble_block()


def test_ORCAInputGenerator_create_point_charge_file(orca_input_generator, tmp_path):
    # Create the point charge file
    orca_input_generator.create_point_charge_file(pc_file=tmp_path / "orca.pc")

    # Read the written file
    orca_pc_file = np.loadtxt(tmp_path / "orca.pc", skiprows=1)

    # Check that the contents of the file match the reference
    assert len(orca_pc_file) == 371

    assert_allclose(
        orca_pc_file[::30],
        np.array(
            [
                [-2.00000000e00, -2.11070451e00, 2.11070451e00, -2.14923990e00],
                [2.00000000e00, 2.11024676e00, -2.11024676e00, -4.26789529e00],
                [2.00000000e00, 6.32954443e00, 2.11144262e00, -4.36728442e-02],
                [-2.00000000e00, -4.22049353e00, 6.32889566e00, 7.72802266e-03],
                [2.00000000e00, -6.33074029e00, -2.11024676e00, -4.26789529e00],
                [-2.00000000e00, 4.22049353e00, -6.33074029e00, -4.26789529e00],
                [-2.00000000e00, 6.33074029e00, 2.11024676e00, -6.37814205e00],
                [-2.00000000e00, 2.11024676e00, -8.44098706e00, -4.26789529e00],
                [-2.00000000e00, -8.44098706e00, -6.32080280e00, 5.67209089e-03],
                [2.00000000e00, -2.11024676e00, 8.44098706e00, -6.37814205e00],
                [8.00000000e-01, -4.64254288e01, 3.79844418e01, -3.99237095e-02],
                [3.12302613e00, -0.00000000e00, -5.71441194e01, -2.36698692e01],
                [2.10472999e00, -2.36698692e01, 5.71441194e01, 2.59086514e01],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
