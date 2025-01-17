from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io.orca import write_orca
from quacc.calculators.mrcc.io import write_mrcc

from 

if TYPE_CHECKING:
    from autoSKZCAM.types import CalculatorInfo




def skzcam_eint_flow():
    pass

def skzcam_generate_job():
    pass
    
def skzcam_calculate_job():
    pass

def skzcam_analysis():
    pass

def skzcam_write_inputs():
    pass

def chemshell_run_job():

    pass




def write_inputs(
    skzcam_cluster_calculators: CalculatorInfo, input_dir: str | Path
) -> None:
    """
    Generate the input files for the SKZCAM calculations.

    Parameters
    ----------
    skzcam_cluster_calculators
        A dictionary containing the cluster number as key and a dictionary of ASE calculators for the calculations that need to performed on each cluster.
    input_dir
        The directory where the input files will be written.

    Returns
    -------
    None
    """

    for cluster_num in skzcam_cluster_calculators:
        for calculation_label in skzcam_cluster_calculators[cluster_num]:
            code = calculation_label.split()[0]
            method = calculation_label.split()[1]
            frozen_core = calculation_label.split()[2]
            basis_set = calculation_label.split()[3]
            for structure in ["adsorbate", "slab", "adsorbate_slab"]:
                system_path = Path(
                    input_dir,
                    str(cluster_num),
                    code,
                    f"{method}_{basis_set}_{frozen_core}",
                    structure,
                )
                system_path.mkdir(parents=True, exist_ok=True)
                # Write MRCC input files
                if code == "mrcc":
                    calc_parameters = deepcopy(
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ].calc.parameters
                    )
                    if "genbas" in calc_parameters:
                        if calc_parameters["genbas"] is not None:
                            with Path.open(Path(system_path, "GENBAS"), "w") as f:
                                f.write(calc_parameters["genbas"])
                        del calc_parameters["genbas"]

                    write_mrcc(
                        Path(system_path, "MINP"),
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ],
                        calc_parameters,
                    )

                # Write ORCA input files
                elif code == "orca":
                    calc_parameters = deepcopy(
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ].calc.parameters
                    )

                    if "pointcharges" in calc_parameters:
                        if calc_parameters["pointcharges"] is not None:
                            with Path.open(Path(system_path, "orca.pc"), "w") as f:
                                f.write(calc_parameters["pointcharges"])
                        del calc_parameters["pointcharges"]

                    write_orca(
                        Path(system_path, "orca.inp"),
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ],
                        calc_parameters,
                    )
