from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.vasp.create_input import count_symbols
from ase.constraints import FixAtoms
from ase.io import read
from quacc import change_settings, flow, job
from quacc.recipes.vasp._base import run_and_summarize
from quacc.schemas.ase import Summarize

from autoSKZCAM.analysis import get_quasi_rrho

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase import Atoms
    from quacc.types import Filenames, SourceDirectory, VaspSchema


def dft_ensemble_analyse(
    calc_dir: Path | str,
    xc_ensemble: list[str] | dict[str, str],
    geom_error_xc: str,
    vib_xc_ensemble: list[str],
    freeze_surface_vib: bool,
    temperature: float = 200.0,
) -> dict[str, list[float]]:
    """
    Analyses the completed DFT ensemble calculations.

    Parameters
    ----------
    calc_dir : Path or str
        The directory where the calculations were performed.
    xc_ensemble : dict[str, str]
        A dictionary containing the xc functionals to be used as keys and the corresponding settings as values.
    geom_error_xc : str
        The xc functional to be used for the geometry error calculation.
    vib_xc_ensemble : list[str]
        A list of xc functionals for which the vibrational calculations were performed.
    freeze_surface_vib : bool
        True if the vibrational calculations on the slab should be skipped.
    temperature : float
        The temperature to get the vibrational contributions to.

    Returns
    -------
    dict[str, list[float]]
        A dictionary containing the relaxation energy (and its geometry error) and DeltaH contributions from the DFT ensemble.

    """

    # Ensure that all the functionals in vib_xc_ensemble are also in xc_ensemble
    for vib_xc in vib_xc_ensemble:
        if vib_xc not in xc_ensemble:
            raise ValueError(
                f"The functional {vib_xc} in vib_xc_ensemble is not in the xc_ensemble."
            )

    # Ensure that the geom_error_xc is in the xc_ensemble
    if geom_error_xc is not None and geom_error_xc not in xc_ensemble:
        raise ValueError(
            f"The functional {geom_error_xc} in geom_error_xc is not in the xc_ensemble."
        )

    dft_ensemble_results = read_completed_calculations(
        calc_dir, xc_ensemble, vib_xc_ensemble, freeze_surface_vib
    )

    # Confirm that all the calculations have been completed
    for job_type in dft_ensemble_results:
        for xc_func in xc_ensemble:
            if "vib" in job_type and (
                xc_func not in vib_xc_ensemble
                or (job_type == "07-slab_vib" and freeze_surface_vib)
            ):
                continue
            if dft_ensemble_results[job_type][xc_func] is None or (
                "results" in dft_ensemble_results[job_type]
                and "energy" not in dft_ensemble_results[job_type][xc_func]["results"]
            ):
                raise ValueError(
                    f"The {job_type} calculation for the functional {xc_func} has not been completed."
                )

    xc_eads_dict = {xc_func: 0 for xc_func in xc_ensemble}
    xc_eint_dict = {xc_func: 0 for xc_func in xc_ensemble}
    xc_vib_dict = {xc_func: 0 for xc_func in vib_xc_ensemble}

    for xc_func in xc_ensemble:
        xc_eads_dict[xc_func] = (
            dft_ensemble_results["04-adsorbate_slab"][xc_func]["results"]["energy"]
            - dft_ensemble_results["03-slab"][xc_func]["results"]["energy"]
            - dft_ensemble_results["01-molecule"][xc_func]["results"]["energy"]
        )
        xc_eint_dict[xc_func] = (
            dft_ensemble_results["08-eint_adsorbate_slab"][xc_func]["results"]["energy"]
            - dft_ensemble_results["09-eint_adsorbate"][xc_func]["results"]["energy"]
            - dft_ensemble_results["10-eint_slab"][xc_func]["results"]["energy"]
        )

        if xc_func in vib_xc_ensemble:
            adsorbate_slab_dU, _, _, _ = get_quasi_rrho(
                dft_ensemble_results["05-adsorbate_slab_vib"][xc_func]["results"][
                    "real_vib_freqs"
                ],
                dft_ensemble_results["05-adsorbate_slab_vib"][xc_func]["results"][
                    "imag_vib_freqs"
                ],
                temperature,
            )
            adsorbate_dU, _, _, _ = get_quasi_rrho(
                dft_ensemble_results["06-molecule_vib"][xc_func]["results"][
                    "real_vib_freqs"
                ],
                dft_ensemble_results["06-molecule_vib"][xc_func]["results"][
                    "imag_vib_freqs"
                ],
                temperature,
            )

            if freeze_surface_vib is False:
                slab_dU, _, _, _ = get_quasi_rrho(
                    dft_ensemble_results["07-slab_vib"][xc_func]["results"][
                        "real_vib_freqs"
                    ],
                    dft_ensemble_results["07-slab_vib"][xc_func]["results"][
                        "imag_vib_freqs"
                    ],
                    temperature,
                )
            else:
                slab_dU = 0
            xc_vib_dict[xc_func] = adsorbate_slab_dU - adsorbate_dU - slab_dU

    erlx = xc_eads_dict[geom_error_xc] - xc_eint_dict[geom_error_xc]
    geom_error = 2 * np.sqrt(
        np.mean(
            [
                (xc_eads_dict[xc_func] - xc_eint_dict[xc_func] - erlx) ** 2
                for xc_func in xc_ensemble
                if xc_func != geom_error_xc
            ]
        )
    )
    return {
        "DFT Erlx": [erlx * 1000, geom_error * 1000],
        "DFT DeltaH": [
            np.mean([xc_vib_dict[xc_func] for xc_func in vib_xc_ensemble]),
            2 * np.std([xc_vib_dict[xc_func] for xc_func in vib_xc_ensemble]),
        ],
    }


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
    adsorbate_slab_gen_func : Callable[[Atoms], Atoms]
        The function to generate the adsorbate molecule. It is important that the indices of the adsorbates are always the first indices in the Atoms object, followed by the slab Atoms object.

    Returns
    -------
    dict[str, dic[str,VaspSchema]]
        A dictionary containing the results of the DFT ensemble calculations for each functional in the ensemble.

    """

    if vib_xc_ensemble is None:
        vib_xc_ensemble = []

    job_list = [
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

    # Ensure that all the functionals in vib_xc_ensemble are also in xc_ensemble
    for vib_xc in vib_xc_ensemble:
        if vib_xc not in xc_ensemble:
            raise ValueError(
                f"The functional {vib_xc} in vib_xc_ensemble is not in the xc_ensemble."
            )

    # Ensure that the geom_error_xc is in the xc_ensemble
    if geom_error_xc is not None and geom_error_xc not in xc_ensemble:
        raise ValueError(
            f"The functional {geom_error_xc} in geom_error_xc is not in the xc_ensemble."
        )

    if job_params is not None:
        for job_type in job_params:
            if job_type not in job_list:
                raise ValueError(
                    f"The {job_type} key in job_params is not valid. Please choose from the following: '01-molecule', '02-unit_cell', '03-slab', '04-adsorbate_slab', '05-adsorbate_slab_vib', '06-molecule_vib', '07-slab_vib', '08-eint_adsorbate_slab', '09-eint_adsorbate', '10-eint_slab'."
                )
    else:
        job_params = {}

    # Try to read in completed calculations from the calc_dir
    dft_ensemble_results = read_completed_calculations(
        calc_dir, xc_ensemble, vib_xc_ensemble, freeze_surface_vib
    )

    for xc_func, xc_func_kwargs in xc_ensemble.items():
        # relax molecule
        if dft_ensemble_results["01-molecule"][xc_func] is None:
            calc_kwargs = {**job_params.get("01-molecule", {}), **xc_func_kwargs}
            dft_ensemble_results["01-molecule"][xc_func] = relax_job(
                adsorbate,
                additional_fields={
                    "calc_results_dir": Path(calc_dir, "01-molecule", xc_func)
                },
                pmg_kpts={"kppvol": 1},
                **calc_kwargs,
            )

        # relax solid
        if dft_ensemble_results["02-unit_cell"][xc_func] is None:
            calc_kwargs = {**job_params.get("02-unit_cell", {}), **xc_func_kwargs}
            relax_job1 = relax_job(
                unit_cell,
                additional_fields={
                    "calc_results_dir": Path(calc_dir, "02-unit_cell", xc_func)
                },
                relax_cell=True,
                unique_dir=True,
                **calc_kwargs,
            )
            dft_ensemble_results["02-unit_cell"][xc_func] = relax_job(
                relax_job1["atoms"],
                relax_cell=True,
                additional_fields={
                    "calc_results_dir": Path(calc_dir, "02-unit_cell", xc_func),
                    "relax1": relax_job1,
                },
                **calc_kwargs,
            )

        # bulk to slab
        if dft_ensemble_results["03-slab"][xc_func] is None:
            calc_kwargs = {**job_params.get("03-slab", {}), **xc_func_kwargs}
            initial_slab = slab_gen_func(
                dft_ensemble_results["02-unit_cell"][xc_func]["atoms"]
            )
            dft_ensemble_results["03-slab"][xc_func] = relax_job(
                initial_slab,
                additional_fields={
                    "calc_results_dir": Path(calc_dir, "03-slab", xc_func)
                },
                pmg_kpts={"length_densities": [50, 50, 1]},
                **calc_kwargs,
            )

        # slab to ads_slab
        if dft_ensemble_results["04-adsorbate_slab"][xc_func] is None:
            calc_kwargs = {**job_params.get("04-adsorbate_slab", {}), **xc_func_kwargs}
            initial_adsorbate_slab = adsorbate_slab_gen_func(
                dft_ensemble_results["01-molecule"][xc_func]["atoms"],
                dft_ensemble_results["03-slab"][xc_func]["atoms"],
            )

            dft_ensemble_results["04-adsorbate_slab"][xc_func] = relax_job(
                initial_adsorbate_slab,
                additional_fields={
                    "calc_results_dir": Path(calc_dir, "04-adsorbate_slab", xc_func)
                },
                pmg_kpts={"length_densities": [50, 50, 1]},
                **calc_kwargs,
            )

        if xc_func in vib_xc_ensemble:
            # vibrational calculation
            if dft_ensemble_results["05-adsorbate_slab_vib"][xc_func] is None:
                calc_kwargs = {
                    **job_params.get("05-adsorbate_slab_vib", {}),
                    **xc_func_kwargs,
                }
                adsorbate_slab_vib_atoms = dft_ensemble_results["04-adsorbate_slab"][
                    xc_func
                ]["atoms"]

                if freeze_surface_vib:
                    adsorbate_len = len(
                        dft_ensemble_results["01-molecule"][xc_func]["atoms"]
                    )
                    slab_indices = list(range(len(adsorbate_slab_vib_atoms)))[
                        adsorbate_len:
                    ]
                    adsorbate_slab_vib_atoms.set_constraint(
                        FixAtoms(indices=slab_indices)
                    )

                dft_ensemble_results["05-adsorbate_slab_vib"][xc_func] = freq_job(
                    adsorbate_slab_vib_atoms,
                    additional_fields={
                        "calc_results_dir": Path(
                            calc_dir, "05-adsorbate_slab_vib", xc_func
                        )
                    },
                    pmg_kpts={"length_densities": [50, 50, 1]},
                    **calc_kwargs,
                )

            if dft_ensemble_results["06-molecule_vib"][xc_func] is None:
                calc_kwargs = {
                    **job_params.get("06-molecule_vib", {}),
                    **xc_func_kwargs,
                }
                dft_ensemble_results["06-molecule_vib"][xc_func] = freq_job(
                    dft_ensemble_results["01-molecule"][xc_func]["atoms"],
                    additional_fields={
                        "calc_results_dir": Path(calc_dir, "06-molecule_vib", xc_func)
                    },
                    pmg_kpts={"kppvol": 1},
                    **calc_kwargs,
                )

            if (
                dft_ensemble_results["07-slab_vib"][xc_func] is None
                and freeze_surface_vib is False
            ):
                calc_kwargs = {**job_params.get("07-slab_vib", {}), **xc_func_kwargs}
                dft_ensemble_results["07-slab_vib"][xc_func] = freq_job(
                    dft_ensemble_results["03-slab"][xc_func]["atoms"],
                    additional_fields={
                        "calc_results_dir": Path(calc_dir, "07-slab_vib", xc_func)
                    },
                    pmg_kpts={"length_densities": [50, 50, 1]},
                    **calc_kwargs,
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
                calc_kwargs = {
                    **job_params.get("08-eint_adsorbate_slab", {}),
                    **xc_func_kwargs,
                }
                dft_ensemble_results["08-eint_adsorbate_slab"][xc_func] = static_job(
                    adsorbate_slab_atoms,
                    additional_fields={
                        "calc_results_dir": Path(
                            calc_dir, "08-eint_adsorbate_slab", xc_func
                        )
                    },
                    pmg_kpts={"length_densities": [50, 50, 1]},
                    **calc_kwargs,
                )

            # eint on adsorbate
            if dft_ensemble_results["09-eint_adsorbate"][xc_func] is None:
                calc_kwargs = {
                    **job_params.get("09-eint_adsorbate", {}),
                    **xc_func_kwargs,
                }
                dft_ensemble_results["09-eint_adsorbate"][xc_func] = static_job(
                    fixed_adsorbate_atoms,
                    additional_fields={
                        "calc_results_dir": Path(calc_dir, "09-eint_adsorbate", xc_func)
                    },
                    pmg_kpts={"length_densities": [50, 50, 1]},
                    **calc_kwargs,
                )

            # eint on slab
            if dft_ensemble_results["10-eint_slab"][xc_func] is None:
                calc_kwargs = {**job_params.get("10-eint_slab", {}), **xc_func_kwargs}
                dft_ensemble_results["10-eint_slab"][xc_func] = static_job(
                    fixed_slab_atoms,
                    additional_fields={
                        "calc_results_dir": Path(calc_dir, "10-eint_slab", xc_func)
                    },
                    pmg_kpts={"length_densities": [50, 50, 1]},
                    **calc_kwargs,
                )

    return dft_ensemble_results


def read_completed_calculations(
    calc_dir: Path | str,
    xc_ensemble: list[str] | dict[str, str],
    vib_xc_ensemble: list[str],
    freeze_surface_vib: bool,
) -> dict[str, dict[str, VaspSchema]]:
    """
    Read in the completed calculations from the calc_dir.

    Parameters
    ----------
    calc_dir : Path or str
        The directory where the calculations were performed.
    xc_ensemble : dict[str, str]
        A dictionary containing the xc functionals to be used as keys and the corresponding settings as values.
    vib_xc_ensemble : list[str]
        A list of xc functionals for which the vibrational calculations were performed.
    freeze_surface_vib : bool
        True if the vibrational calculations on the slab should be skipped.

    Returns
    -------
    dict[str, dict[str,VaspSchema]]
        A dictionary containing the results of the DFT ensemble calculations for each functional in the ensemble.

    """
    job_list = [
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

    dft_ensemble_results = {
        job_type: {xc_func: None for xc_func in xc_ensemble} for job_type in job_list
    }
    for xc_func in xc_ensemble:
        for job_type in job_list:
            vasp_dir = Path(calc_dir) / job_type / xc_func
            outcar_filename = Path(calc_dir) / job_type / xc_func / "OUTCAR"
            if outcar_filename.exists():
                if "vib" in job_type:
                    if xc_func not in vib_xc_ensemble or (
                        freeze_surface_vib and "07-slab_vib" in job_type
                    ):
                        continue
                    with Path.open(outcar_filename, encoding="ISO-8859-1") as file:
                        final_atoms = read(file, format="vasp-out")
                        real_vib_freqs, imag_vib_freqs = read_vib_freq(outcar_filename)
                        final_atoms.calc.results["real_vib_freqs"] = real_vib_freqs
                        final_atoms.calc.results["imag_vib_freqs"] = imag_vib_freqs
                else:
                    with Path.open(outcar_filename, encoding="ISO-8859-1") as file:
                        final_atoms = read(file, format="vasp-out")

                dft_ensemble_results[job_type][xc_func] = Summarize(
                    directory=vasp_dir
                ).run(final_atoms, final_atoms)
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

    with Path.open(filename, encoding="ISO-8859-1") as f:
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
        keys, refer to [quacc.calculators.vasp.vasp.Vasp](https://quantum-accelerators.github.io/quacc/reference/quacc/calculators/vasp/vasp.html#quacc.calculators.vasp.vasp.Vasp).

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run](https://quantum-accelerators.github.io/quacc/reference/quacc/schemas/vasp.html#quacc.schemas.vasp.VaspSummarize).
        See the type-hint for the data structure.
    """
    calc_defaults = {
        "encut": 600,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 1e-7,
        "algo": "ALL",
        "istart": 0,
        "lreal": False,
        "ispin": 1,
        "nelm": 200,
        "nelmin": 8,
        "ibrion": 5,
        "potim": 0.01,
        "nfree": 2,
        "isym": 0,
        "lcharg": False,
        "lwave": False,
        "nsw": 0,
        "symprec": 1e-8,
    }

    if "calc_results_dir" in additional_fields:
        with change_settings(
            {
                "RESULTS_DIR": additional_fields["calc_results_dir"],
                "CREATE_UNIQUE_DIR": False,
                "GZIP_FILES": False,
            }
        ):
            results_dict = run_and_summarize(
                atoms,
                preset=preset,
                calc_defaults=calc_defaults,
                calc_swaps=calc_kwargs,
                additional_fields={"name": "VASP Freq"} | (additional_fields or {}),
                copy_files=copy_files,
            )
    else:
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
    results_dict["atoms"] = resort_atoms(atoms, results_dict["atoms"])
    results_dict["atoms"].set_constraint(atoms.constraints.copy())

    return results_dict


@job
def static_job(
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
        keys, refer to [quacc.calculators.vasp.vasp.Vasp](https://quantum-accelerators.github.io/quacc/reference/quacc/calculators/vasp/vasp.html#quacc.calculators.vasp.vasp.Vasp).

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run](https://quantum-accelerators.github.io/quacc/reference/quacc/schemas/vasp.html#quacc.schemas.vasp.VaspSummarize).
        See the type-hint for the data structure.
    """
    calc_defaults = {
        "encut": 600,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 5e-7,
        "algo": "ALL",
        "istart": 0,
        "lreal": False,
        "ispin": 1,
        "nelm": 200,
        "nelmin": 8,
        "ibrion": -1,
        "isym": 0,
        "lcharg": False,
        "lwave": False,
        "nsw": 0,
        "symprec": 1e-8,
    }

    if "calc_results_dir" in additional_fields:
        with change_settings(
            {
                "RESULTS_DIR": additional_fields["calc_results_dir"],
                "CREATE_UNIQUE_DIR": False,
                "GZIP_FILES": False,
            }
        ):
            results_dict = run_and_summarize(
                atoms,
                preset=preset,
                calc_defaults=calc_defaults,
                calc_swaps=calc_kwargs,
                additional_fields={"name": "VASP Static"} | (additional_fields or {}),
                copy_files=copy_files,
            )
    else:
        results_dict = run_and_summarize(
            atoms,
            preset=preset,
            calc_defaults=calc_defaults,
            calc_swaps=calc_kwargs,
            additional_fields={"name": "VASP Static"} | (additional_fields or {}),
            copy_files=copy_files,
        )

    results_dict["atoms"] = resort_atoms(atoms, results_dict["atoms"])
    results_dict["atoms"].set_constraint(atoms.constraints.copy())

    return results_dict


@job
def relax_job(
    atoms: Atoms,
    preset: str | None = "BulkSet",
    relax_cell: bool = False,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    unique_dir: bool = False,
    **calc_kwargs,
) -> VaspSchema:
    """
    Relax a structure.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    relax_cell
        True if a volume relaxation (ISIF = 3) should be performed. False if
        only the positions (ISIF = 2) should be updated.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [quacc.calculators.vasp.vasp.Vasp](https://quantum-accelerators.github.io/quacc/reference/quacc/calculators/vasp/vasp.html#quacc.calculators.vasp.vasp.Vasp) calculator.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run](https://quantum-accelerators.github.io/quacc/reference/quacc/schemas/vasp.html#quacc.schemas.vasp.VaspSummarize).
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "encut": 600,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 5e-7,
        "algo": "ALL",
        "istart": 0,
        "lreal": False,
        "ispin": 1,
        "nelm": 200,
        "nelmin": 8,
        "ediffg": -0.01,
        "isif": 3 if relax_cell else 2,
        "ibrion": 2,
        "isym": 0,
        "lcharg": False,
        "lwave": False,
        "nsw": 200,
        "symprec": 1e-8,
    }

    if "calc_results_dir" in additional_fields:
        with change_settings(
            {
                "RESULTS_DIR": additional_fields["calc_results_dir"],
                "CREATE_UNIQUE_DIR": unique_dir,
                "GZIP_FILES": False,
            }
        ):
            results_dict = run_and_summarize(
                atoms,
                preset=preset,
                calc_defaults=calc_defaults,
                calc_swaps=calc_kwargs,
                additional_fields={"name": "VASP Relax"} | (additional_fields or {}),
                copy_files=copy_files,
            )
    else:
        results_dict = run_and_summarize(
            atoms,
            preset=preset,
            calc_defaults=calc_defaults,
            calc_swaps=calc_kwargs,
            additional_fields={"name": "VASP Relax"} | (additional_fields or {}),
            copy_files=copy_files,
        )

    results_dict["atoms"] = resort_atoms(atoms, results_dict["atoms"])
    results_dict["atoms"].set_constraint(atoms.constraints.copy())

    return results_dict


@flow
def adsorbate_slab_rss_flow(
    adsorbate: Atoms,
    slab: Atoms,
    num_rss: int = 5,
    min_z: float = 3.0,
    max_z: float = 8.0,
    preset: str | None = "SlabSet",
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
    unique_dir: bool = False,
    **calc_kwargs,
) -> dict[str, VaspSchema]:
    """
    Perform a random structure search on a specified number of structures.

    Parameters
    ----------
    adsorbate
        Adsorbate Atoms object
    slab
        Slab Atoms object
    num_rss
        Number of random structure search calculations to perform.
    min_z
        Minimum z-height above the slab for the adsorbate.
    max_z
        Maximum z-height above the slab for the adsorbate.
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.
    unique_dir
        True if a unique directory should be created for each calculation.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [quacc.calculators.vasp.vasp.Vasp](https://quantum-accelerators.github.io/quacc/reference/quacc/calculators/vasp/vasp.html#quacc.calculators.vasp.vasp.Vasp) calculator.

    Returns
    -------
    dict[str,VaspSchema]
        Dictionary with RSS calculation number as key and the value is Dictionary of results from [quacc.schemas.vasp.VaspSummarize.run](https://quantum-accelerators.github.io/quacc/reference/quacc/schemas/vasp.html#quacc.schemas.vasp.VaspSummarize).
        See the type-hint for the data structure.
    """
    from ase import neighborlist

    rng = np.random.default_rng()

    calc_defaults = {
        "encut": 400,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 1e-5,
        "algo": "Fast",
        "istart": 0,
        "lreal": False,
        "ispin": 1,
        "nelm": 200,
        "nelmin": 8,
        "ediffg": -0.03,
        "isif": 2,
        "ibrion": 2,
        "isym": 0,
        "lcharg": False,
        "lwave": False,
        "nsw": 200,
        "symprec": 1e-8,
    }

    surface_max_z = np.max([atom.position[2] for atom in slab])
    adsorbate_num_clashes = len(neighborlist.neighbor_list("i", adsorbate, cutoff=1.4))
    slab_num_clashes = len(neighborlist.neighbor_list("i", slab, cutoff=1.4))
    rss_results_dict = {}

    rss_num = 0
    while rss_num < num_rss:
        rotated_adsorbate = adsorbate.copy()
        rotated_adsorbate.set_cell(slab.get_cell())
        random_angle = rng.random() * 360.0
        random_direction = [rng.random(), rng.random(), rng.random()]
        rotated_adsorbate.rotate(random_angle, random_direction, center="COM")
        random_displacement = (
            rng.random() * slab.get_cell()[0]
            + rng.random() * slab.get_cell()[1]
            + np.array(
                [0.0, 0.0, surface_max_z + rng.random() * (max_z - min_z) + min_z]
            )
            - rotated_adsorbate.get_center_of_mass()
        )
        rotated_adsorbate.translate(random_displacement)
        rss_adsorbate_slab = rotated_adsorbate + slab
        slab_indices = slab.constraints[0].__dict__["index"]
        c = FixAtoms(indices=len(adsorbate) + slab_indices)
        rss_adsorbate_slab.set_constraint(c)
        num_of_clashes = len(
            neighborlist.neighbor_list("i", rss_adsorbate_slab, cutoff=1.4)
        )
        if num_of_clashes - adsorbate_num_clashes - slab_num_clashes == 0:
            rss_num += 1
            if "calc_results_dir" in additional_fields:
                with change_settings(
                    {
                        "RESULTS_DIR": Path(
                            additional_fields["calc_results_dir"], f"RSS_{rss_num:05d}"
                        ),
                        "CREATE_UNIQUE_DIR": unique_dir,
                        "GZIP_FILES": False,
                    }
                ):
                    results_dict = run_and_summarize(
                        rss_adsorbate_slab,
                        preset=preset,
                        calc_defaults=calc_defaults,
                        calc_swaps=calc_kwargs,
                        additional_fields={
                            "name": f"VASP RSS {rss_num:05d}",
                            "calc_results_dir": Path(
                                additional_fields["calc_results_dir"],
                                "rss_calcs",
                                f"RSS_{rss_num:05d}",
                            ),
                        },
                        copy_files=copy_files,
                    )
            else:
                results_dict = run_and_summarize(
                    rss_adsorbate_slab,
                    preset=preset,
                    calc_defaults=calc_defaults,
                    calc_swaps=calc_kwargs,
                    additional_fields={
                        "name": f"VASP RSS {rss_num:05d}",
                        "calc_results_dir": Path(
                            additional_fields["calc_results_dir"],
                            "rss_calcs",
                            f"RSS_{rss_num:05d}",
                        ),
                    },
                    copy_files=copy_files,
                )
            results_dict["atoms"] = resort_atoms(
                rss_adsorbate_slab, results_dict["atoms"]
            )
            results_dict["atoms"].set_constraint(rss_adsorbate_slab.constraints.copy())
            rss_results_dict[f"RSS_{rss_num:05d}"] = results_dict
    return rss_results_dict


def resort_atoms(initial_atoms: Atoms, final_atoms: Atoms) -> Atoms:
    """
    Resorts the atoms in the final_atoms object to match the order of the initial_atoms object.

    Parameters
    ----------
    initial_atoms : Atoms
        The initial Atoms object prior to entering the ASE calculator.
    final_atoms : Atoms
        The final Atoms object after being run by the ASE calculator.

    Returns
    -------
    Atoms
        The final Atoms object with the atoms in the same order as the initial Atoms object.

    """
    symbols, _ = count_symbols(initial_atoms, exclude=())

    # Create sorting list
    srt = []  # type: list[int]

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
