from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ase.atoms import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io.orca import write_orca

from quacc import get_settings
from quacc.calculators.mrcc.io import write_mrcc
from quacc.calculators.mrcc.mrcc import MRCC, MrccProfile

from autoSKZCAM.data import (
    capped_ecp_defaults,
    code_calculation_defaults,
    frozen_core_defaults,
)
from autoSKZCAM.io import MRCCInputGenerator, ORCAInputGenerator

if TYPE_CHECKING:

    from autoSKZCAM.types import (
        CalculatorInfo,
        ElementInfo,
        ElementStr,
        ONIOMLayerInfo
    )


def write_inputs(
    self, skzcam_cluster_calculators: CalculatorInfo, input_dir: str | Path
) -> None:
    """
    Generates the SKZCAM input for the MRCC and ORCA ASE calculators.

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
                    code,
                    f"{method}_{basis_set}_{frozen_core}",
                    structure,
                )
                system_path.mkdir(parents=True, exist_ok=True)
                # Write MRCC input files
                if code == "mrcc":
                    write_mrcc(
                        Path(system_path, "MINP"),
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ],
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ].calc.parameters,
                    )
                # Write ORCA input files
                elif code == "orca":
                    write_orca(
                        Path(system_path, "orca.inp"),
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ],
                        skzcam_cluster_calculators[cluster_num][calculation_label][
                            structure
                        ].calc.parameters,
                    )