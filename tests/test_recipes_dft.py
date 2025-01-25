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
from ase.build import surface, sort
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_equal

from autoSKZCAM.embed import CreateEmbeddedCluster
from autoSKZCAM.oniom import Prepare
from autoSKZCAM.recipes_dft import (
    dft_ensemble_flow
)

FILE_DIR = Path(__file__).parent


def test_dft_ensemble_flow(tmpdir):
    xc_ensemble_params = {
        'PBE-D2-Ne': {"GGA": "PE", "IVDW": 1},
        'revPBE-D4': {"GGA": "RE", "IVDW": 13, "VDW_S8": 1.7468, "VDW_A1": 0.5363, "VDW_A2": 3.0726},
        'vdW-DF': {"GGA": "RE", "AGGAC": 0.0, "LUSE_VDW": True, "LASPH": True},
        'rev-vdW-DF2': {"GGA": "MK", "LUSE_VDW": True, "PARAM1": 0.1234, "PARAM2": 0.711357, "ZAB_VDW": -1.8867, "AGGAC": 0.0},
        'PBE0-D4': {"LHFCALC": True, "GGA": "PE", "IVDW": 13, "VDW_S8": 1.2007, "VDW_A1": 0.4009, "VDW_A2": 5.0293},
        'B3LYP-D2-Ne': {"LHFCALC": True, "GGA": "B5", "AEXX": 0.2, "AGGAX": 0.72, "AGGAC": 0.81, "ALDAC": 0.19, "IVDW": 1, "VDW_S6": 1.05}
    }

    def slab_gen_func(unit_cell):
        surface_cell = sort(surface(unit_cell,(0,0,1),2,vacuum=7.5,periodic=True)*(2,2,1))

        fix_list = []
        for atom_idx in surface_cell:
            if atom_idx.position[2] < (np.max(surface_cell.get_positions()[:,2]) -  3):
                fix_list += [atom_idx.index]

        c = FixAtoms(indices=fix_list)
        surface_cell.set_constraint(c)
        return surface_cell
    
    def adsorbate_slab_gen_func(adsorbate, slab):
        maxzpos = np.max(slab.get_positions()[:,2]) 
        top_Mg_index = [atom.index for atom in slab if (abs(atom.position[2]- maxzpos) < 0.1 and atom.symbol == 'Mg')][0]
        adsorbate.set_cell(slab.get_cell())
        adsorbate.set_pbc(slab.get_pbc())
        adsorbate.translate(slab[top_Mg_index].position - adsorbate.get_positions()[0] + np.array([0,0,2]))
        adsorbate_slab = adsorbate + slab
        return adsorbate_slab


        

    # dft_ensemble_results = dft_ensemble_flow(xc_ensemble=xc_ensemble_params, calc_dir=FILE_DIR / 'skzcam_files' /'dft_calc_dir', vib_xc_ensemble=['PBE-D2-Ne','revPBE-D4','vdW-DF','rev-vdW-DF2'], geom_error_xc = 'revPBE-D4')

    unit_cell = read(FILE_DIR / 'mocked_vasp_runs' / 'POSCAR_unit_cell')
    adsorbate = read(FILE_DIR / 'mocked_vasp_runs' / 'POSCAR_adsorbate')

    dft_ensemble_results = dft_ensemble_flow(xc_ensemble=xc_ensemble_params, slab_gen_func= slab_gen_func, adsorbate_slab_gen_func= adsorbate_slab_gen_func, adsorbate=adsorbate, unit_cell = unit_cell, calc_dir= tmpdir, vib_xc_ensemble=['PBE-D2-Ne','revPBE-D4','vdW-DF','rev-vdW-DF2'], geom_error_xc = 'revPBE-D4')



    # dft_ensemble_results['04-adsorbate_slab']['PBE-D2-Ne']['atoms'].write('PBE-D2-Ne_adsorbate_slab.xyz')