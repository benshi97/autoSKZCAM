"""Core recipes for MRCC and ORCA taken from quacc."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any
from quacc.calculators.mrcc.mrcc import MrccProfile
from ase.calculators.orca import ORCA, OrcaProfile
import psutil
from quacc.utils.dicts import recursive_dict_merge
from quacc.utils.lists import merge_list_params
from quacc.runners.ase import Runner
from quacc import job, get_settings
from quacc.schemas.ase import Summarize

from autoSKZCAM.calculators import ORCA, MRCC

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quacc.types import Filenames, RunSchema, SourceDirectory

@job
def static_job_mrcc(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    method: str = "pbe",
    basis: str = "def2-tzvp",
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
        Custom kwargs for the Gaussian calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.Summarize][]
    """
    default_inputs = {"calc": method, "basis": basis, "symm": "off"}

    return run_and_summarize_mrcc(
        atoms,
        charge,
        spin_multiplicity,
        default_inputs=default_inputs,
        input_swaps=calc_kwargs,
        additional_fields={"name": "MRCC Static"},
        copy_files=copy_files,
    )


@job
def static_job_orca(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    xc: str = "wb97x-d3bj",
    basis: str = "def2-tzvp",
    orcasimpleinput: list[str] | None = None,
    orcablocks: list[str] | None = None,
    nprocs: int | Literal["max"] = "max",
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
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
    xc
        Exchange-correlation functional
    basis
        Basis set
    orcasimpleinput
        List of `orcasimpleinput` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcasimpleinput`.
    orcablocks
        List of `orcablocks` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcablocks`.
    nprocs
        Number of processors to use. Defaults to the number of physical cores.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.

    Returns
    -------
    RunSchema
        Dictionary of results
    """
    additional_fields = {"name": "ORCA Static"} | (additional_fields or {})
    nprocs = psutil.cpu_count(logical=False) if nprocs == "max" else nprocs
    default_inputs = [xc, basis, "engrad", "normalprint"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    return run_and_summarize_orca(
        atoms,
        charge,
        spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=orcasimpleinput,
        block_swaps=orcablocks,
        additional_fields=additional_fields,
        copy_files=copy_files,
    )

def run_and_summarize_orca(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> RunSchema:
    """
    Base job function for ORCA recipes.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    RunSchema
        Dictionary of results
    """
    calc = prep_calculator_orca(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=input_swaps,
        block_swaps=block_swaps,
        **calc_kwargs,
    )

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields=additional_fields,
    ).run(final_atoms, atoms)

def prep_calculator_orca(
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    **calc_kwargs,
) -> ORCA:
    """
    Prepare the ORCA calculator.

    Parameters
    ----------
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    ORCA
        The ORCA calculator
    """
    inputs = merge_list_params(default_inputs, input_swaps)
    blocks = merge_list_params(default_blocks, block_swaps)
    if "xyzfile" not in inputs:
        inputs.append("xyzfile")
    orcasimpleinput = " ".join(inputs)
    orcablocks = "\n".join(blocks)
    settings = get_settings()

    return ORCA(
        profile=OrcaProfile(command=settings.ORCA_CMD),
        charge=charge,
        mult=spin_multiplicity,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        **calc_kwargs,
    )

def run_and_summarize_mrcc(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: dict[str, str] | None = None,
    input_swaps: dict[str, str] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> RunSchema:
    """
    Base job function for MRCC recipes.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.Summarize][]
    """
    calc = prep_calculator_mrcc(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        input_swaps=input_swaps,
    )

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields=additional_fields,
    ).run(final_atoms, atoms)


def prep_calculator_mrcc(
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: dict[str, str] | None = None,
    input_swaps: dict[str, str] | None = None,
) -> MRCC:
    """
    Prepare the MRCC calculator.

    Parameters
    ----------
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    input_swaps
        Dictionary of mrccinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.

    Returns
    -------
    MRCC
        The MRCC calculator
    """
    mrccinput = recursive_dict_merge(
        default_inputs, input_swaps, {"charge": charge, "mult": spin_multiplicity}
    )

    # If spin_multiplicity bigger than 1, check if scftype is in either mrccinput or blocks
    if spin_multiplicity > 1:
        if "scftype" not in mrccinput:
            raise ValueError(
                "For spin_multiplicity > 1, scftype keyword must be specified in mrccinput"
            )
        if mrccinput["scftype"].lower() not in ["uhf", "uks", "rohf", "roks"]:
            raise ValueError(
                "For spin_multiplicity > 1, scftype must not be set to RHF or RKS"
            )

    settings = get_settings()

    return MRCC(profile=MrccProfile(command=settings.MRCC_CMD), **mrccinput)