"""Core recipes for MRCC and ORCA taken from quacc."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.calculators.orca import OrcaProfile
from quacc import get_settings, job
from quacc.calculators.mrcc.mrcc import MrccProfile
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize

from autoSKZCAM.calculators import MRCC, ORCA

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from quacc.types import Filenames, RunSchema, SourceDirectory


@job
def static_job_mrcc(
    atoms: Atoms,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> RunSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    method
        The method [e.g., PBE or CCSD(T)] to use, this is the value for the calc keyword.
    basis
        Basis set
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Custom kwargs for the Mrcc calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.Summarize][]
    """

    # Convert a charge string to an integer
    settings = get_settings()

    calc = MRCC(profile=MrccProfile(command=settings.MRCC_CMD), **calc_kwargs)

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        charge_and_multiplicity=(0, 1), additional_fields={"name": "MRCC Static"}
    ).run(final_atoms, atoms)


@job
def static_job_orca(
    atoms: Atoms,
    orcasimpleinput: str = '',
    orcablocks: str = '',
    pointcharges: str | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> RunSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    orcasimpleinput
        List of `orcasimpleinput` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcasimpleinput`.
    orcablocks
        List of `orcablocks` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcablocks`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.

    Returns
    -------
    RunSchema
        Dictionary of results
    """

    settings = get_settings()

    calc = ORCA(
        profile=OrcaProfile(command=settings.ORCA_CMD),
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        pointcharges=pointcharges,
    )

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        charge_and_multiplicity=(0, 1), additional_fields={"name": "ORCA Static"}
    ).run(final_atoms, atoms)
