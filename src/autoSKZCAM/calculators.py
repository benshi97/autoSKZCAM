from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.calculators.orca import OrcaProfile, OrcaTemplate
from ase.io.orca import write_orca
from ase.units import Hartree
from quacc.calculators.mrcc.io import write_mrcc
from quacc.calculators.mrcc.mrcc import MrccProfile, MrccTemplate

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from autoSKZCAM.types import EnergyInfo


class SkzcamOrcaTemplate(OrcaTemplate):
    """
    The ORCA calculator template class to be used for (auto)SKZCAM calculations.
    """

    def write_input(
        self,
        profile: OrcaProfile,  # noqa: ARG002
        directory: Path | str,
        atoms: Atoms,
        parameters: dict[str, str],
        properties: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """
        Write the MRCC input file.

        Parameters
        ----------
        profile
            The OrcaProfile class.
        directory
            The directory in which to write the input file.
        atoms
            The Atoms object.
        parameters
            The parameters for the calculation.
        properties
            The properties to calculate.

        Returns
        -------
        None
        """
        parameters = dict(parameters)

        kw = {
            "charge": 0,
            "mult": 1,
            "orcasimpleinput": "B3LYP def2-TZVP",
            "orcablocks": "%pal nprocs 1 end",
        }
        kw.update(parameters)
        if "pointcharges" in parameters and parameters["pointcharges"] is not None:
            with Path.open(directory / "orca.pc", "w") as pc_file:
                pc_file.write(parameters["pointcharges"])
            # Remove 'pointcharges' from kw, as it is not an ORCA keyword
            del kw["pointcharges"]

        write_orca(directory / self.inputname, atoms, kw)


class SkzcamMrccTemplate(MrccTemplate):
    """
    The MRCC calculator template class to be used for (auto)SKZCAM calculations.
    """

    def write_input(
        self,
        profile: MrccProfile,  # noqa: ARG002
        directory: Path | str,
        atoms: Atoms,
        parameters: dict[str, str],
        properties: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """
        Write the MRCC input file.

        Parameters
        ----------
        profile
            The MrccProfile class.
        directory
            The directory in which to write the input file.
        atoms
            The Atoms object.
        parameters
            The parameters for the calculation.
        properties
            The properties to calculate.

        Returns
        -------
        None
        """
        parameters = dict(parameters)

        kw = {"charge": 0, "mult": 1, "calc": "PBE", "basis": "def2-SVP"}
        kw.update(parameters)

        if "genbas" in parameters and parameters["genbas"] is not None:
            with Path.open(directory / "GENBAS", "w") as genbas_file:
                genbas_file.write(parameters["genbas"])
            # Remove 'genbas' from kw, as it is not an MRCC keyword
            del kw["genbas"]

        write_mrcc(directory / self.inputname, atoms, kw)


class MRCC(GenericFileIOCalculator):
    """
    Class for performing MRCC calculations.
    """

    def __init__(
        self, *, profile: MrccProfile = None, directory: str | Path = ".", **kwargs
    ) -> None:
        """
        Construct MRCC-calculator object.

        Parameters
        ----------
        profile: MrccProfile
            The MRCC profile to use.
        directory: str
            The directory in which to run the calculation.
        **kwargs
            The parameters for the MRCC calculation.

        Examples
        --------
        Use default values:

        >>> from quacc.calculators.mrcc.mrcc import MRCC, MrccProfile
        >>> from ase.build import molecule
        >>> from quacc import get_settings

        >>> calc = MRCC(
        ...     profile=MrccProfile(command=get_settings().MRCC_CMD),
        ...     charge=0,
        ...     mult=1,
        ...     basis="def2-SVP",
        ...     calc="PBE",
        ... )
        >>> h = molecule("H2")
        >>> h.set_calculator(calc)
        >>> h.get_total_energy()

        Returns
        -------
        None
        """

        super().__init__(
            template=SkzcamMrccTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )


class ORCA(GenericFileIOCalculator):
    """
    Class for performing ORCA calculations.
    """

    def __init__(self, *, profile=None, directory=".", **kwargs):
        """
        Construct ORCA-calculator object.

        Parameters
        ----------
        profile: OrcaProfile
            The ORCA profile to use.
        directory: str
            The directory in which to run the calculation.
        **kwargs
            The parameters for the ORCA calculation.


        Examples
        --------
        Use default values:

        >>> from ase.calculators.orca import ORCA
        >>> h = Atoms(
        ...     "H",
        ...     calculator=ORCA(
        ...         charge=0,
        ...         mult=1,
        ...         directory="water",
        ...         orcasimpleinput="B3LYP def2-TZVP",
        ...         orcablocks="%pal nprocs 16 end",
        ...     ),
        ... )

        Returns
        -------
        None

        """

        super().__init__(
            template=SkzcamOrcaTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )


def read_orca_energy(lines: list[str]) -> EnergyInfo:
    """
    Reads the energy components (SCF energy, MP2 correlation energy, CCSD correlation energy, CCSD(T) correlation energy) from the ORCA output file where available.

    Parameters
    ----------
    lines : list[str]
        List of lines read from the ORCA output file.

    Returns
    -------
    EnergyInfo
        Dictionary with the energy components. The keys are:

        - **energy** (`float`):
          Total energy (not computed in this function).

        - **scf_energy** (`float`):
          SCF energy.

        - **mp2_corr_energy** (`float`):
          MP2 correlation energy.

        - **ccsd_corr_energy** (`float`):
          CCSD correlation energy.

        - **ccsdt_corr_energy** (`float`):
          CCSD(T) correlation energy.
    """

    energy_dict = {
        "energy": None,
        "scf_energy": None,
        "mp2_corr_energy": None,
        "ccsd_corr_energy": None,
        "ccsdt_corr_energy": None,
    }

    for line in lines:
        if "Total Energy       :" in line:
            energy_dict["scf_energy"] = float(line.split()[-4]) * Hartree
        elif (
            "MP2 CORRELATION ENERGY:" in line
            or "E(MP2)" in line
            or "E(RI-MP2)" in line
            or "E(L-MP2)" in line
            or "E(SL-MP2)" in line
        ):
            energy_dict["mp2_corr_energy"] = (
                float(line.replace("Eh", "").split()[-1]) * Hartree
            )
        elif "E(CORR)" in line:
            energy_dict["ccsd_corr_energy"] = float(line.split()[-1]) * Hartree
        elif "Final correlation energy" in line:
            energy_dict["ccsdt_corr_energy"] = float(line.split()[-1]) * Hartree

    return energy_dict


def read_orca_outputs(output_file_path: Path | str) -> EnergyInfo:
    """
    Reads the energy components (SCF energy, MP2 correlation energy, CCSD correlation energy, CCSD(T) correlation energy) from the ORCA output file where available and calculates the total energy (based on the highest level of theory)

    Parameters
    ----------
    output_file_path : Path | str
        Path to the ORCA output file.

    Returns
    -------
    EnergyInfo
        Dictionary with the energy components. The keys are:

        - **energy** (`float`):
          Total energy (not computed in this function).

        - **scf_energy** (`float`):
          SCF energy.

        - **mp2_corr_energy** (`float`):
          MP2 correlation energy.

        - **ccsd_corr_energy** (`float`):
          CCSD correlation energy.

        - **ccsdt_corr_energy** (`float`):
          CCSD(T) correlation energy.
    """
    with Path.open(output_file_path) as output_textio:
        lines = output_textio.readlines()

    energy_dict = read_orca_energy(lines)

    # Raise error if scf_energy is None
    if energy_dict["scf_energy"] is None:
        raise ValueError("SCF energy not found in ORCA output file")

    if energy_dict["ccsdt_corr_energy"] is not None:
        energy_dict["energy"] = (
            energy_dict["scf_energy"] + energy_dict["ccsdt_corr_energy"]
        )
    elif energy_dict["ccsd_corr_energy"] is not None:
        energy_dict["energy"] = (
            energy_dict["scf_energy"] + energy_dict["ccsd_corr_energy"]
        )
    elif energy_dict["mp2_corr_energy"] is not None:
        energy_dict["energy"] = (
            energy_dict["scf_energy"] + energy_dict["mp2_corr_energy"]
        )
    else:
        energy_dict["energy"] = energy_dict["scf_energy"]

    return energy_dict
