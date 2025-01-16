from __future__ import annotations

from typing import TYPE_CHECKING


from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.calculators.orca import OrcaProfile, OrcaTemplate
from ase.io.orca import write_orca
from quacc.calculators.mrcc.io import write_mrcc
from quacc.calculators.mrcc.mrcc import MrccProfile, MrccTemplate
from pathlib import Path


if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms


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
    """Class for doing ORCA calculations.

    Example:

      calc = ORCA(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')
    """

    def __init__(self, *, profile=None, directory=".", **kwargs):
        """Construct ORCA-calculator object.

        Parameters
        ==========
        charge: int

        mult: int

        orcasimpleinput : str

        orcablocks: str


        Examples
        ========
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

        """

        super().__init__(
            template=SkzcamOrcaTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )
