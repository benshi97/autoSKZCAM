from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ase.calculators.vasp.create_input import count_symbols
from ase.io import read
from quacc import change_settings, flow, job
from quacc.recipes.vasp._base import run_and_summarize
from quacc.recipes.vasp.core import relax_job, static_job
from quacc.schemas.ase import Summarize

# from quacc.recipes.vasp.core import relax_job, double_relax_flow, static_job

if TYPE_CHECKING:
    from ase import Atoms
    from quacc.types import Filenames, SourceDirectory, VaspSchema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the level explicitly

# Add a stream handler to ensure output to console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@flow
def dft_ensemble_flow(
    xc_ensemble: dict[str, dict[str, Any]],
    vib_xc_ensemble: list[str] | None = None,
    geom_error_xc: str | None = None,
    freeze_surface_vib: bool = True,
    job_params: dict[str, dict[str, Any]] | None = None,
    adsorbate: Atoms | None = None,
    unit_cell: Atoms | None = None,
    calc_dir: str | Path = "./dft_calc_dir",
    slab_gen_func: Callable[[Atoms], Atoms] | None = None,
    adsorbate_slab_gen_func: Callable[[Atoms], Atoms] | None = None,
):
    """
    Workflow to perform the DFT ensemble calculations to obtain the geometry error and get the DeltaH contribution.
    The workflow consists of the following steps:

    1. Relax the gas-phase molecule for each functional in the ensemble.

    2. Relax the unit cell for each functional in the ensemble.

    3. Generate and relax the slab from the relaxed solid for each functional in the ensemble.

    4. Generate and relax the adsorbate-slab complex from the relaxed adsorbate and slab for each functional in the ensemble.

    5. Perform the vibrational calculation for each functional in the ensemble.

    6. Perform the eint calculation on the chosen functional for each functional in the ensemble.

    Parameters
    ----------
    xc_ensemble : dict[str, dict[str, Any]]
        A dictionary containing the xc functionals to be used as keys and the corresponding settings as values.
    job_params : dict[str, dict[str, Any]], optional
        A dictionary containing the job parameters to be used for each functional in the ensemble. If not provided, the default job parameters will be used.
    adsorbate : Atoms, optional
        The adsorbate molecule to be used for the calculations. If not provided, will attempt to read in the adsorbate from the calc_dir.
    unit_cell : Atoms, optional
        The unit cell of the solid to be used for the calculations. If not provided, will attempt to read in the unit cell from the calc_dir.
    calc_dir : str or Path, optional
        The directory where the calculations will be performed. Defaults to './calc_dir'.
    slab_gen_func : Callable[[Atoms], Atoms]
        The function to generate the slab from the unit cell.
    adsorbate_gen_func : Callable[[Atoms], Atoms]
        The function to generate the adsorbate molecule. It is important that the indices of the adsorbates are always the first indices in the Atoms object, followed by the slab Atoms object.

    Returns
    -------
    dict[str, dic[str,VaspSchema]]
        A dictionary containing the results of the DFT ensemble calculations for each functional in the ensemble.

    """

    if vib_xc_ensemble is None:
        vib_xc_ensemble = []
    dft_ensemble_results = {
        job_name: {xc_func: None for xc_func in xc_ensemble}
        for job_name in [
            "01-molecule",
            "02-unit_cell",
            "03-slab",
            "04-adsorbate_slab",
            "05-adsorbate_slab_vib",
            "06-molecule_vib",
            "07-slab_vib",
            "08-eint_adsorbate_slab",
            "09-eint_adsorbate",
            "10-eint_slab",
        ]
    }

    # Try to read in completed calculations from the calc_dir
    for xc_func in xc_ensemble:
        for job_name in dft_ensemble_results:
            vasp_dir = Path(calc_dir) / job_name / xc_func
            outcar_filename = Path(calc_dir) / job_name / xc_func / "OUTCAR"
            poscar_filename = Path(calc_dir) / job_name / xc_func / "POSCAR"
            if poscar_filename.exists():
                input_atoms = read(poscar_filename, format="vasp")
            else:
                input_atoms = None
            if outcar_filename.exists():
                with open(outcar_filename, encoding="ISO-8859-1") as file:
                    final_atoms = read(file, format="vasp-out")

                if "vib" in job_name:
                    real_vib_freqs, imag_vib_freqs = read_vib_freq(outcar_filename)
                    final_atoms.calc.results["real_vib_freqs"] = real_vib_freqs
                    final_atoms.calc.results["imag_vib_freqs"] = imag_vib_freqs
                dft_ensemble_results[job_name][xc_func] = Summarize(
                    directory=vasp_dir
                ).run(final_atoms, input_atoms)

    for xc_func in xc_ensemble:
        # relax molecule
        # Set no constraints on the molecule
        if dft_ensemble_results["01-molecule"][xc_func] is None:
            with change_settings(
                {
                    "RESULTS_DIR": Path(calc_dir) / "01-molecule" / xc_func,
                    "CREATE_UNIQUE_DIR": False,
                    "GZIP_FILES": False,
                }
            ):
                dft_ensemble_results["01-molecule"][xc_func] = relax_job(
                    adsorbate,
                    additional_fields={"job_type": "01-molecule", "xc_func": xc_func},
                )
                dft_ensemble_results["01-molecule"][xc_func]["atoms"] = resort_atoms(
                    adsorbate, dft_ensemble_results["01-molecule"][xc_func]["atoms"]
                )

        # relax solid
        if dft_ensemble_results["02-unit_cell"][xc_func] is None:
            with change_settings(
                {
                    "RESULTS_DIR": Path(calc_dir) / "02-unit_cell" / xc_func,
                    "CREATE_UNIQUE_DIR": False,
                    "GZIP_FILES": False,
                }
            ):
                dft_ensemble_results["02-unit_cell"][xc_func] = relax_job(
                    unit_cell,
                    additional_fields={"job_type": "02-unit_cell", "xc_func": xc_func},
                )
                dft_ensemble_results["02-unit_cell"][xc_func]["atoms"] = resort_atoms(
                    unit_cell, dft_ensemble_results["02-unit_cell"][xc_func]["atoms"]
                )

        # bulk to slab
        if dft_ensemble_results["03-slab"][xc_func] is None:
            initial_slab = slab_gen_func(
                dft_ensemble_results["02-unit_cell"][xc_func]["atoms"]
            )
            with change_settings(
                {
                    "RESULTS_DIR": Path(calc_dir) / "03-slab" / xc_func,
                    "CREATE_UNIQUE_DIR": False,
                    "GZIP_FILES": False,
                }
            ):
                dft_ensemble_results["03-slab"][xc_func] = relax_job(
                    initial_slab,
                    additional_fields={"job_type": "03-slab", "xc_func": xc_func},
                )
                dft_ensemble_results["03-slab"][xc_func]["atoms"] = resort_atoms(
                    initial_slab, dft_ensemble_results["03-slab"][xc_func]["atoms"]
                )

        # slab to ads_slab
        if dft_ensemble_results["04-adsorbate_slab"][xc_func] is None:
            initial_adsorbate_slab = adsorbate_slab_gen_func(
                dft_ensemble_results["01-molecule"][xc_func]["atoms"],
                dft_ensemble_results["03-slab"][xc_func]["atoms"],
            )
            # sorted_
            with change_settings(
                {
                    "RESULTS_DIR": Path(calc_dir) / "04-adsorbate_slab" / xc_func,
                    "CREATE_UNIQUE_DIR": False,
                    "GZIP_FILES": False,
                }
            ):
                dft_ensemble_results["04-adsorbate_slab"][xc_func] = relax_job(
                    initial_adsorbate_slab,
                    additional_fields={
                        "job_type": "04-adsorbate_slab",
                        "xc_func": xc_func,
                    },
                )
                dft_ensemble_results["04-adsorbate_slab"][xc_func]["atoms"] = (
                    resort_atoms(
                        initial_adsorbate_slab,
                        dft_ensemble_results["04-adsorbate_slab"][xc_func]["atoms"],
                    )
                )

        if xc_func in vib_xc_ensemble:
            # vibrational calculation
            if dft_ensemble_results["05-adsorbate_slab_vib"][xc_func] is None:
                with change_settings(
                    {
                        "RESULTS_DIR": Path(calc_dir)
                        / "05-adsorbate_slab_vib"
                        / xc_func,
                        "CREATE_UNIQUE_DIR": False,
                        "GZIP_FILES": True,
                    }
                ):
                    dft_ensemble_results["05-adsorbate_slab_vib"][xc_func] = freq_job(
                        dft_ensemble_results["04-adsorbate_slab"][xc_func]["atoms"],
                        additional_fields={
                            "job_type": "05-adsorbate_slab_vib",
                            "xc_func": xc_func,
                        },
                    )
                    dft_ensemble_results["05-adsorbate_slab_vib"][xc_func]["atoms"] = (
                        resort_atoms(
                            dft_ensemble_results["04-adsorbate_slab"][xc_func]["atoms"],
                            dft_ensemble_results["05-adsorbate_slab_vib"][xc_func][
                                "atoms"
                            ],
                        )
                    )

            if dft_ensemble_results["06-molecule_vib"][xc_func] is None:
                with change_settings(
                    {
                        "RESULTS_DIR": Path(calc_dir) / "06-molecule_vib" / xc_func,
                        "CREATE_UNIQUE_DIR": False,
                        "GZIP_FILES": False,
                    }
                ):
                    dft_ensemble_results["06-molecule_vib"][xc_func] = freq_job(
                        dft_ensemble_results["01-molecule"][xc_func]["atoms"],
                        additional_fields={
                            "job_type": "06-molecule_vib",
                            "xc_func": xc_func,
                        },
                    )
                    dft_ensemble_results["06-molecule_vib"][xc_func]["atoms"] = (
                        resort_atoms(
                            dft_ensemble_results["01-molecule"][xc_func]["atoms"],
                            dft_ensemble_results["06-molecule_vib"][xc_func]["atoms"],
                        )
                    )

            if (
                dft_ensemble_results["07-slab_vib"][xc_func] is None
                and freeze_surface_vib is False
            ):
                with change_settings(
                    {
                        "RESULTS_DIR": Path(calc_dir) / "07-slab_vib" / xc_func,
                        "CREATE_UNIQUE_DIR": False,
                        "GZIP_FILES": False,
                    }
                ):
                    dft_ensemble_results["07-slab_vib"][xc_func] = freq_job(
                        dft_ensemble_results["03-slab"][xc_func]["atoms"],
                        additional_fields={
                            "job_type": "07-slab_vib",
                            "xc_func": xc_func,
                        },
                    )
                    dft_ensemble_results["07-slab_vib"][xc_func]["atoms"] = (
                        resort_atoms(
                            dft_ensemble_results["03-slab"][xc_func]["atoms"],
                            dft_ensemble_results["07-slab_vib"][xc_func]["atoms"],
                        )
                    )

    if geom_error_xc is not None:
        adsorbate_len = len(dft_ensemble_results["01-molecule"][geom_error_xc]["atoms"])
        adsorbate_slab_atoms = dft_ensemble_results["04-adsorbate_slab"][geom_error_xc][
            "atoms"
        ]
        fixed_adsorbate_atoms = dft_ensemble_results["04-adsorbate_slab"][
            geom_error_xc
        ]["atoms"][:adsorbate_len]
        fixed_slab_atoms = dft_ensemble_results["04-adsorbate_slab"][geom_error_xc][
            "atoms"
        ][adsorbate_len:]

        for xc_func in xc_ensemble:
            # eint on adsorbate_slab
            if dft_ensemble_results["08-eint_adsorbate_slab"][xc_func] is None:
                dft_ensemble_results["08-eint_adsorbate_slab"][xc_func] = static_job(
                    adsorbate_slab_atoms,
                    additional_fields={
                        "job_type": "08-eint_adsorbate_slab",
                        "xc_func": xc_func,
                    },
                )
                dft_ensemble_results["08-eint_adsorbate_slab"][xc_func]["atoms"] = (
                    resort_atoms(
                        adsorbate_slab_atoms,
                        dft_ensemble_results["08-eint_adsorbate_slab"][xc_func][
                            "atoms"
                        ],
                    )
                )

            # eint on adsorbate
            if dft_ensemble_results["09-eint_adsorbate"][xc_func] is None:
                dft_ensemble_results["09-eint_adsorbate"][xc_func] = static_job(
                    fixed_adsorbate_atoms,
                    additional_fields={
                        "job_type": "09-eint_adsorbate",
                        "xc_func": xc_func,
                    },
                )
                dft_ensemble_results["09-eint_adsorbate"][xc_func]["atoms"] = (
                    resort_atoms(
                        fixed_adsorbate_atoms,
                        dft_ensemble_results["09-eint_adsorbate"][xc_func]["atoms"],
                    )
                )

            # eint on slab
            if dft_ensemble_results["10-eint_slab"][xc_func] is None:
                dft_ensemble_results["10-eint_slab"][xc_func] = static_job(
                    fixed_slab_atoms,
                    additional_fields={"job_type": "10-eint_slab", "xc_func": xc_func},
                )
                dft_ensemble_results["10-eint_slab"][xc_func]["atoms"] = resort_atoms(
                    fixed_slab_atoms,
                    dft_ensemble_results["10-eint_slab"][xc_func]["atoms"],
                )

    return dft_ensemble_results


def read_vib_freq(filename):
    """
    Read vibrational frequencies from a file.

    Parameters
    ----------
    filename : str
        The name of the file to read vibrational frequencies from.

    Returns
    -------
    freq : list
        List of real vibrational frequencies.
    i_freq : list
        List of imaginary vibrational frequencies.

    Notes
    -----
    This function reads vibrational frequency information from a given file. It extracts both real and imaginary vibrational frequencies from the lines containing the frequency data. The frequencies are extracted based on the presence of "THz" in the data. Real frequencies are extracted unless the "f/i=" label is found, in which case imaginary frequencies are extracted. The function returns two lists containing the real and imaginary frequencies respectively.
    """

    freq = []
    i_freq = []

    with open(filename, encoding="ISO-8859-1") as f:
        lines = f.readlines()

    for line in lines:
        data = line.split()
        if "THz" in data:
            if "f/i=" not in data:
                freq.append(float(data[-2]))  # Append real frequency to the freq list
            else:
                i_freq.append(
                    float(data[-2])
                )  # Append imaginary frequency to the i_freq list
    return freq, i_freq


@job
def freq_job(
    atoms: Atoms,
    preset: str | None = "BulkSet",
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][].

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run][].
        See the type-hint for the data structure.
    """
    calc_defaults = {"ediff": 1e-7, "isym": 0, "lcharg": False, "lwave": True, "nsw": 0}

    results_dict = run_and_summarize(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP Freq"} | (additional_fields or {}),
        copy_files=copy_files,
    )

    real_freqs, imag_freqs = read_vib_freq(results_dict["dir_name"] / "OUTCAR")
    results_dict["results"]["real_vib_freqs"] = real_freqs
    results_dict["results"]["imag_vib_freqs"] = imag_freqs

    return results_dict


def resort_atoms(initial_atoms: Atoms, final_atoms: Atoms) -> Atoms:
    """
    Resorts the atoms in the final_atoms object to match the order of the initial_atoms object.

    Parameters
    ----------
    initial_atoms : Atoms
        The initial Atoms object prior to entering the ASE calculator.
    final_atoms : Atoms
        The final Atoms object after being run by the ASE calculator.

    """
    symbols, _ = count_symbols(initial_atoms, exclude=())

    # Create sorting list
    srt = []  # type: List[int]

    for symbol in symbols:
        for m, atom in enumerate(initial_atoms):
            if atom.symbol == symbol:
                srt.append(m)
    # Create the resorting list
    resrt = list(range(len(srt)))
    for n in range(len(resrt)):
        resrt[srt[n]] = n

    resorted_final_atoms = final_atoms.copy()[srt]
    if hasattr(final_atoms, "calc"):
        resorted_final_atoms.calc = final_atoms.calc

    return resorted_final_atoms
