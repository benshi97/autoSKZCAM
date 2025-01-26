from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import sort, surface
from ase.constraints import FixAtoms
from ase.io import read
from numpy.testing import assert_allclose

from autoSKZCAM.recipes_dft import adsorbate_slab_rss_flow, dft_ensemble_flow

FILE_DIR = Path(__file__).parent


def test_dft_ensemble_flow(tmpdir):
    xc_ensemble_params = {
        "PBE-D2-Ne": {"GGA": "PE", "IVDW": 1},
        "revPBE-D4": {
            "GGA": "RE",
            "IVDW": 13,
            "VDW_S8": 1.7468,
            "VDW_A1": 0.5363,
            "VDW_A2": 3.0726,
        },
        "vdW-DF": {"GGA": "RE", "AGGAC": 0.0, "LUSE_VDW": True, "LASPH": True},
        "rev-vdW-DF2": {
            "GGA": "MK",
            "LUSE_VDW": True,
            "PARAM1": 0.1234,
            "PARAM2": 0.711357,
            "ZAB_VDW": -1.8867,
            "AGGAC": 0.0,
        },
        "PBE0-D4": {
            "LHFCALC": True,
            "GGA": "PE",
            "IVDW": 13,
            "VDW_S8": 1.2007,
            "VDW_A1": 0.4009,
            "VDW_A2": 5.0293,
        },
        "B3LYP-D2-Ne": {
            "LHFCALC": True,
            "GGA": "B5",
            "AEXX": 0.2,
            "AGGAX": 0.72,
            "AGGAC": 0.81,
            "ALDAC": 0.19,
            "IVDW": 1,
            "VDW_S6": 1.05,
        },
    }

    # Test reading in the results as if all calculations have finished
    dft_ensemble_results = dft_ensemble_flow(
        xc_ensemble=xc_ensemble_params,
        calc_dir=FILE_DIR / "mocked_vasp_runs" / "dft_calc_dir",
        vib_xc_ensemble=["PBE-D2-Ne", "revPBE-D4", "vdW-DF", "rev-vdW-DF2"],
        geom_error_xc="revPBE-D4",
    )

    pbe_d2_ne_energies = [
        dft_ensemble_results[key]["PBE-D2-Ne"]["results"]["energy"]
        for key in dft_ensemble_results
        if dft_ensemble_results[key]["PBE-D2-Ne"] is not None
    ]

    pbe0_d4_energies = [
        dft_ensemble_results[key]["PBE0-D4"]["results"]["energy"]
        for key in dft_ensemble_results
        if dft_ensemble_results[key]["PBE0-D4"] is not None
    ]

    assert_allclose(
        pbe_d2_ne_energies + pbe0_d4_energies,
        [
            -14.79708994,
            -48.04639982,
            -375.65096691,
            -390.67585416,
            -390.67011508,
            -14.79129876,
            -390.80916499,
            -14.79622472,
            -375.78007192,
            -19.28971632,
            -60.34891922,
            -472.32144052,
            -491.84538959,
            -490.97879953,
            -19.27593454,
            -471.45312374,
        ],
        rtol=1e-05,
        atol=1e-07,
    )

    assert dft_ensemble_results["07-slab_vib"]["PBE-D2-Ne"] is None
    assert dft_ensemble_results["05-adsorbate_slab_vib"]["PBE-D2-Ne"] is not None
    assert dft_ensemble_results["05-adsorbate_slab_vib"]["PBE0-D4"] is None

    # Test whether errors from giving wrong inputs
    # in vib_xc_ensemble
    with pytest.raises(
        ValueError,
        match="The functional asdf in vib_xc_ensemble is not in the xc_ensemble.",
    ):
        dft_ensemble_results = dft_ensemble_flow(
            xc_ensemble=xc_ensemble_params,
            calc_dir=FILE_DIR / "mocked_vasp_runs" / "dft_calc_dir",
            vib_xc_ensemble=["asdf"],
            geom_error_xc="revPBE-D4",
        )
    # in geom_error_xc
    with pytest.raises(
        ValueError,
        match="The functional asdf in geom_error_xc is not in the xc_ensemble.",
    ):
        dft_ensemble_results = dft_ensemble_flow(
            xc_ensemble=xc_ensemble_params,
            calc_dir=FILE_DIR / "mocked_vasp_runs" / "dft_calc_dir",
            vib_xc_ensemble=["PBE-D2-Ne", "revPBE-D4", "vdW-DF", "rev-vdW-DF2"],
            geom_error_xc="asdf",
        )
    # in job_params
    with pytest.raises(
        ValueError,
        match="The asdf key in job_params is not valid. Please choose from the following: '01-molecule', '02-unit_cell', '03-slab', '04-adsorbate_slab', '05-adsorbate_slab_vib', '06-molecule_vib', '07-slab_vib', '08-eint_adsorbate_slab', '09-eint_adsorbate', '10-eint_slab'.",
    ):
        dft_ensemble_results = dft_ensemble_flow(
            xc_ensemble=xc_ensemble_params,
            calc_dir=FILE_DIR / "mocked_vasp_runs" / "dft_calc_dir",
            vib_xc_ensemble=["PBE-D2-Ne", "revPBE-D4", "vdW-DF", "rev-vdW-DF2"],
            geom_error_xc="revPBE-D4",
            job_params={"asdf": {}},
        )

    # Test if starting calculations from scratch. We will need to define the functions for generating the slabs and adsorbate_slabs

    def slab_gen_func(unit_cell):
        surface_cell = sort(
            surface(unit_cell, (0, 0, 1), 2, vacuum=7.5, periodic=True) * (2, 2, 1)
        )

        fix_list = []
        for atom_idx in surface_cell:
            if atom_idx.position[2] < (np.max(surface_cell.get_positions()[:, 2]) - 3):
                fix_list += [atom_idx.index]

        c = FixAtoms(indices=fix_list)
        surface_cell.set_constraint(c)
        return surface_cell

    def adsorbate_slab_gen_func(adsorbate, slab):
        maxzpos = np.max(slab.get_positions()[:, 2])
        top_Mg_index = next(
            atom.index
            for atom in slab
            if (abs(atom.position[2] - maxzpos) < 0.1 and atom.symbol == "Mg")
        )
        adsorbate.set_cell(slab.get_cell())
        adsorbate.set_pbc(slab.get_pbc())
        adsorbate.translate(
            slab[top_Mg_index].position
            - adsorbate.get_positions()[0]
            + np.array([0, 0, 2])
        )

        adsorbate_slab = adsorbate + slab
        slab_indices = slab.constraints[0].__dict__["index"]

        c = FixAtoms(indices=len(adsorbate) + slab_indices)
        adsorbate_slab.set_constraint(c)

        return adsorbate_slab

    unit_cell = read(FILE_DIR / "mocked_vasp_runs" / "POSCAR_unit_cell")
    adsorbate = read(FILE_DIR / "mocked_vasp_runs" / "POSCAR_adsorbate")

    dft_ensemble_results = dft_ensemble_flow(
        xc_ensemble=xc_ensemble_params,
        slab_gen_func=slab_gen_func,
        adsorbate_slab_gen_func=adsorbate_slab_gen_func,
        adsorbate=adsorbate,
        unit_cell=unit_cell,
        calc_dir=tmpdir,
        vib_xc_ensemble=["PBE-D2-Ne", "revPBE-D4", "vdW-DF", "rev-vdW-DF2"],
        geom_error_xc="revPBE-D4",
    )

    pbe_d2_ne_energies = [
        dft_ensemble_results[key]["PBE-D2-Ne"]["results"]["energy"]
        for key in dft_ensemble_results
        if dft_ensemble_results[key]["PBE-D2-Ne"] is not None
    ]

    pbe0_d4_energies = [
        dft_ensemble_results[key]["PBE0-D4"]["results"]["energy"]
        for key in dft_ensemble_results
        if dft_ensemble_results[key]["PBE0-D4"] is not None
    ]

    assert_allclose(
        pbe_d2_ne_energies + pbe0_d4_energies,
        [
            -14.79708994,
            -48.04639982,
            -375.65096691,
            -390.67585416,
            -390.67011508,
            -14.79129876,
            -390.80916499,
            -14.79622472,
            -375.78007192,
            -19.28971632,
            -60.34891922,
            -472.32144052,
            -491.84538959,
            -490.97879953,
            -19.27593454,
            -471.45312374,
        ],
        rtol=1e-05,
        atol=1e-07,
    )

    ref_slab = slab_gen_func(dft_ensemble_results["02-unit_cell"]["PBE-D2-Ne"]["atoms"])

    assert_allclose(
        ref_slab.get_positions(),
        dft_ensemble_results["03-slab"]["PBE-D2-Ne"]["input_atoms"][
            "atoms"
        ].get_positions(),
        rtol=1e-05,
        atol=1e-07,
    )

    ref_adsorbate_slab = adsorbate_slab_gen_func(
        dft_ensemble_results["01-molecule"]["PBE-D2-Ne"]["atoms"],
        dft_ensemble_results["03-slab"]["PBE-D2-Ne"]["atoms"],
    )

    assert_allclose(
        ref_adsorbate_slab.get_positions(),
        dft_ensemble_results["04-adsorbate_slab"]["PBE-D2-Ne"]["input_atoms"][
            "atoms"
        ].get_positions(),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_allclose(
        dft_ensemble_results["05-adsorbate_slab_vib"]["PBE-D2-Ne"]["results"][
            "real_vib_freqs"
        ]
        + dft_ensemble_results["05-adsorbate_slab_vib"]["PBE-D2-Ne"]["results"][
            "imag_vib_freqs"
        ],
        [261.986723, 28.241108, 28.221562, 14.191815, 3.62909, 3.817755],
        rtol=1e-05,
        atol=1e-07,
    )
    assert all(
        dft_ensemble_results["04-adsorbate_slab"]["PBE-D2-Ne"]["input_atoms"]["atoms"]
        .constraints[0]
        .__dict__["index"]
        == [
            2,
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            18,
            19,
            20,
            21,
            26,
            27,
            28,
            29,
            34,
            35,
            36,
            37,
            42,
            43,
            44,
            45,
            50,
            51,
            52,
            53,
            58,
            59,
            60,
            61,
        ]
    )
    assert all(
        dft_ensemble_results["05-adsorbate_slab_vib"]["PBE-D2-Ne"]["input_atoms"][
            "atoms"
        ]
        .constraints[0]
        .__dict__["index"]
        == [
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
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
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
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
        ]
    )


def test_adsorbate_slab_rss_flow(tmpdir):
    def slab_gen_func(unit_cell):
        surface_cell = sort(
            surface(unit_cell, (0, 0, 1), 2, vacuum=7.5, periodic=True) * (2, 2, 1)
        )

        fix_list = []
        for atom_idx in surface_cell:
            if atom_idx.position[2] < (np.max(surface_cell.get_positions()[:, 2]) - 3):
                fix_list += [atom_idx.index]

        c = FixAtoms(indices=fix_list)
        surface_cell.set_constraint(c)
        return surface_cell

    unit_cell = read(FILE_DIR / "mocked_vasp_runs" / "POSCAR_unit_cell")
    adsorbate = read(FILE_DIR / "mocked_vasp_runs" / "POSCAR_adsorbate")
    slab = slab_gen_func(unit_cell)

    rss_results = adsorbate_slab_rss_flow(
        slab=slab,
        adsorbate=adsorbate,
        num_rss=5,
        additional_fields={"calc_results_dir": tmpdir},
    )
    assert list(rss_results.keys()) == [
        "RSS_00001",
        "RSS_00002",
        "RSS_00003",
        "RSS_00004",
        "RSS_00005",
    ]
