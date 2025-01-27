from __future__ import annotations

import numpy as np
from ase.build import sort, surface
from ase.constraints import FixAtoms
from ase.io import read

OniomInfo = {
    "Bulk MP2": {
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

xc_ensemble = {
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

adsorbate = read("data/POSCAR_CO")
unit_cell = read("data/POSCAR_MgO")


# Define the function which can convert the unit cell into a MgO(001) surface.
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


# Define the function which can add the adsorbate and slab to create a CO molecule adsorbed on top of a Mg site.
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
        slab[top_Mg_index].position - adsorbate.get_positions()[0] + np.array([0, 0, 2])
    )

    adsorbate_slab = adsorbate + slab
    slab_indices = slab.constraints[0].__dict__["index"]

    c = FixAtoms(indices=len(adsorbate) + slab_indices)
    adsorbate_slab.set_constraint(c)

    return adsorbate_slab
