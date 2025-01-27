from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io.orca import write_orca
from quacc import change_settings
from quacc.calculators.mrcc.io import write_mrcc

from autoSKZCAM.analysis import analyze_calculations, compute_skzcam_int_ene, parse_energy
from autoSKZCAM.embed import CreateEmbeddedCluster
from autoSKZCAM.oniom import Prepare
from autoSKZCAM.quacc import static_job_mrcc, static_job_orca

if TYPE_CHECKING:
    from autoSKZCAM.types import ElementStr, OniomLayerInfo, SkzcamOutput


def skzcam_analyse(
    calc_dir: str | Path,
    embedded_cluster_npy_path: Path | str | None = None,
    OniomInfo: dict[str, OniomLayerInfo] | None = None,
    EmbeddedCluster: CreateEmbeddedCluster | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Analyze the completed SKZCAM calculations and compute the final ONIOM contributions.

    Parameters
    ----------
    calc_dir
        The directory containing the calculations.
    embedded_cluster_npy_path
        The path to the embedded cluster .npy object.
    EmbeddedCluster
        The CreateEmbeddedCluster object containing the embedded cluster.
    OniomInfo
        A dictionary containing the information about the ONIOM layers.
    print_results
        If True, displays the results

    Returns
    -------
    dict[str, tuple[float, float]]
        A dictionary containing the ONIOM layer as key and a tuple containing the contribution to the final interaction energy as well as its error.
    """

    # Initialize the EmbeddedCluster object if it is not provided
    if EmbeddedCluster is None and embedded_cluster_npy_path is None:
        # Check if the embedded_cluster.npy file exists in the calc_dir
        if not Path(calc_dir, "embedded_cluster.npy").exists():
            raise ValueError(
                "Either the EmbeddedCluster object must be provided or embedded_cluster_npy_path is set or embedded_cluster.npy is provided in calc_dir."
            )
        EmbeddedCluster = np.load(
            Path(calc_dir, "embedded_cluster.npy"), allow_pickle=True
        ).item()

    elif EmbeddedCluster is None and embedded_cluster_npy_path is not None:
        EmbeddedCluster = np.load(embedded_cluster_npy_path, allow_pickle=True).item()

    # Load the OniomInfo dictionary if it is not provided
    if EmbeddedCluster.OniomInfo is None and OniomInfo is None:
        raise ValueError(
            "The OniomInfo dictionary must be provided in EmbeddedCluster or as an argument."
        )
    if EmbeddedCluster.OniomInfo is not None and OniomInfo is None:
        OniomInfo = EmbeddedCluster.OniomInfo

    skzcam_calcs_analysis = analyze_calculations(
        calc_dir=calc_dir,
        embedded_cluster_path=embedded_cluster_npy_path,
        EmbeddedCluster=EmbeddedCluster,
    )

    return compute_skzcam_int_ene(
        skzcam_calcs_analysis=skzcam_calcs_analysis, OniomInfo=OniomInfo
    )


def skzcam_eint_flow(
    EmbeddedCluster: CreateEmbeddedCluster,
    OniomInfo: dict[str, OniomLayerInfo],
    **kwargs,
):
    """
    The complete SKZCAM protocol to generate the embedded clusters, perform the calculations, and analyze the results.

    Parameters
    ----------
    EmbeddedCluster
        The CreateEmbeddedCluster object containing the embedded cluster. This is initialized using the skzcam_initialize() function.
    OniomInfo
        A dictionary containing the information about the ONIOM layers.
    **kwargs
        Additional keyword arguments to pass to the skzcam_generate_job() and skzcam_calculate_job() functions.

    Returns
    -------
    None
    """

    # Generate the skzcam embedded clusters
    skzcam_generate_job(EmbeddedCluster, **kwargs)

    # Perform the calculations on the embedded clusters
    skzcam_calculate_job(EmbeddedCluster, OniomInfo, **kwargs)


def skzcam_initialize(
    adsorbate_indices: list[int],
    slab_center_indices: list[int],
    atom_oxi_states: dict[ElementStr, int],
    adsorbate_slab_file: str | Path,
    pun_filepath: str | Path = "./ChemShell_EmbeddedCluster.pun",
    run_chemshell: bool = False,
    chemsh_radius_active: float = 40.0,
    chemsh_radius_cluster: float = 60.0,
    chemsh_bq_layer: float = 6.0,
    write_xyz_file: bool = False,
    **kwargs,  # noqa ARG001
) -> CreateEmbeddedCluster:
    """
    Parameters to initialize the SKZCAM protocol to generate the embedded clusters.

    Parameters
    ----------
    adsorbate_indices
        The indices of the atoms that make up the adsorbate molecule.
    slab_center_indices
        The indices of the atoms that make up the 'center' of the slab right beneath the adsorbate.
    slab_indices
        The indices of the atoms that make up the slab.
    atom_oxi_states
        A dictionary with the element symbol as the key and its oxidation state as the value.
    adsorbate_slab_file
        The path to the file containing the adsorbate molecule on the surface slab. It can be in any format that ASE can read.
    pun_filepath
        The path to the .pun file containing the atomic coordinates and charges of the adsorbate-slab complex if it has already been generated by ChemShell. If it is None, then ChemShell will need to be used to create this file.
    chemsh_radius_active
        The radius of the active region in Angstroms. This 'active' region is simply region where the charge fitting is performed to ensure correct Madelung potential; it can be a relatively large value.
    chemsh_radius_cluster
        The radius of the total embedded cluster in Angstroms.
    chemsh_bq_layer
        The height above the surface to place some additional fitting point charges in Angstroms; simply for better reproduction of the electrostatic potential close to the adsorbate.

    """

    EmbeddedCluster = CreateEmbeddedCluster(
        adsorbate_indices=adsorbate_indices,
        slab_center_indices=slab_center_indices,
        atom_oxi_states=atom_oxi_states,
        adsorbate_slab_file=adsorbate_slab_file,
        pun_filepath=pun_filepath,
    )

    # Check that pun_filepath exists if run_chemshell is False
    if run_chemshell is False and not Path(pun_filepath).exists():
        raise ValueError(
            "The path to the .pun file from ChemShell must be provided in EmbeddedCluster if run_chemshell is False."
        )

    if run_chemshell:
        # Create the ChemShell input file
        EmbeddedCluster.run_chemshell(
            filepath=pun_filepath,
            chemsh_radius_active=chemsh_radius_active,
            chemsh_radius_cluster=chemsh_radius_cluster,
            chemsh_bq_layer=chemsh_bq_layer,
            write_xyz_file=write_xyz_file,
        )

    return EmbeddedCluster


def skzcam_generate_job(
    EmbeddedCluster: CreateEmbeddedCluster,
    max_cluster_num: int = 10,
    shell_width: float = 0.1,
    bond_dist: float = 2.5,
    ecp_dist: float = 6.0,
    write_clusters: bool = False,
    write_clusters_path: str | Path = ".",
    write_include_ecp: bool = False,
    **kwargs,  # noqa ARG001
) -> SkzcamOutput:
    """
    Generates the set of clusters for the SKZCAM protocol. It will return the embedded cluster Atoms object and the indices of the atoms in the quantum clusters and the ECP region. The number of clusters created is controlled by the max_cluster_num parameter.

    Parameters
    ----------
    max_num_clusters
        The maximum number of quantum clusters to be created.
    shell_width
        Defines the distance between atoms within shells; this is the maximum distance between any two atoms within the shell.
    bond_dist
        The distance within which an anion is considered to be coordinating a cation.
    ecp_dist
        The distance from edges of the quantum cluster to define the ECP region.
    write_clusters
        If True, the quantum clusters will be written to a file.
    write_clusters_path
        The path to the file where the quantum clusters will be written.
    write_include_ecp
        If True, the ECP region will be included in the quantum clusters.
    """

    # Ensure that the pun_filepath has been provided in EmbeddedCluster
    if (
        not hasattr(EmbeddedCluster, "pun_filepath")
        or EmbeddedCluster.pun_filepath is None
    ):
        raise ValueError(
            "The path pun_filepath to the .pun file from ChemShell must be provided in EmbeddedCluster."
        )

    # Generate the embedded cluster
    EmbeddedCluster.run_skzcam(
        shell_max=max_cluster_num,
        shell_width=shell_width,
        bond_dist=bond_dist,
        ecp_dist=ecp_dist,
        write_clusters=write_clusters,
        write_clusters_path=write_clusters_path,
        write_include_ecp=write_include_ecp,
    )


def skzcam_calculate_job(
    EmbeddedCluster: CreateEmbeddedCluster,
    OniomInfo: dict[str, str],
    dryrun: bool = False,
    use_quacc: bool = False,
    calc_dir: str | Path = "calc_dir",
):
    """
    Perform the skzcam calculations on the embedded clusters.

    Parameters
    ----------
    EmbeddedCluster
        The CreateEmbeddedCluster object containing the embedded cluster.
    OniomInfo
        A dictionary containing the information about the ONIOM layers.

    Returns
    -------
    CalculatorInfo
        A dictionary containing the cluster number as key and a dictionary of ASE calculators for the calculations that have been performed.
    """

    # Prepare the embedded cluster for the calculations
    Prepare(EmbeddedCluster=EmbeddedCluster, OniomInfo=OniomInfo).create_cluster_calcs()

    # Set the OniomInfo attribute
    EmbeddedCluster.OniomInfo = OniomInfo

    if dryrun:
        skzcam_write_inputs(EmbeddedCluster, calc_dir)
    else:
        skzcam_write_inputs(EmbeddedCluster, calc_dir)
        for cluster_num in EmbeddedCluster.skzcam_calcs:
            for calculation_label in EmbeddedCluster.skzcam_calcs[cluster_num]:
                code = calculation_label.split()[0]
                method = calculation_label.split()[1]
                frozen_core = calculation_label.split()[2]
                basis_set = calculation_label.split()[3]
                for structure in ["adsorbate", "slab", "adsorbate_slab"]:
                    system_path = Path(
                        calc_dir,
                        str(cluster_num),
                        code,
                        f"{method}_{basis_set}_{frozen_core}",
                        structure,
                    )
                    EmbeddedCluster.skzcam_calcs[cluster_num][calculation_label][
                        structure
                    ].calc.directory = system_path
                    # Check whether the calculation has already been performed
                    if code == "mrcc":
                        # Simply read calculation if it has already been performed
                        if (
                            Path(system_path, "mrcc.out").exists()
                        ):
                            final_energy = parse_energy(Path(system_path, "mrcc.out"), code=code)
                            EmbeddedCluster.skzcam_calcs[cluster_num][calculation_label][
                                structure
                            ].results = final_energy             

                        else:
                            if use_quacc:
                                calc_parameters = EmbeddedCluster.skzcam_calcs[cluster_num][
                                    calculation_label
                                ][structure].calc.parameters
                                with change_settings(
                                    {
                                        "RESULTS_DIR": system_path,
                                        "CREATE_UNIQUE_DIR": False,
                                        "GZIP_FILES": False,
                                    }
                                ):
                                    static_job_mrcc(
                                        EmbeddedCluster.skzcam_calcs[cluster_num][
                                            calculation_label
                                        ][structure],
                                        **calc_parameters,
                                    )
                            else:
                                EmbeddedCluster.skzcam_calcs[cluster_num][calculation_label][
                                    structure
                                ].get_potential_energy()  
                    elif code == "orca":
                        if ( Path(system_path, "orca.out").exists() ):
                            final_energy = parse_energy(Path(system_path, "orca.out"), code=code)
                            EmbeddedCluster.skzcam_calcs[cluster_num][calculation_label][structure].results = final_energy
                        else:
                            if use_quacc:                             
                                static_job_orca(
                                    EmbeddedCluster.skzcam_calcs[cluster_num][
                                        calculation_label
                                    ][structure],
                                    **calc_parameters,
                                )
                            else:
                                EmbeddedCluster.skzcam_calcs[cluster_num][calculation_label][
                                    structure
                                ].get_potential_energy()


def skzcam_write_inputs(
    EmbeddedCluster: CreateEmbeddedCluster, input_dir: str | Path
) -> None:
    """
    Generate the input files for the SKZCAM calculations.

    Parameters
    ----------
    EmbeddedCluster
        The CreateEmbeddedCluster object containing the embedded cluster.
    input_dir
        The directory where the input files will be written.

    Returns
    -------
    None
    """

    # Confirm that skzcam_calcs is not None
    if EmbeddedCluster.skzcam_calcs is None:
        raise ValueError(
            "The EmbeddedCluster object must have the skzcam_calcs attribute set using oniom.Prepare."
        )

    skzcam_cluster_calculators = EmbeddedCluster.skzcam_calcs

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
    np.save(Path(input_dir, "embedded_cluster.npy"), EmbeddedCluster)
