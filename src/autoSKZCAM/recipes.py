from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io.orca import write_orca
from quacc.calculators.mrcc.io import write_mrcc

from autoSKZCAM.embed import CreateSkzcamClusters
from autoSKZCAM.oniom import Prepare

if TYPE_CHECKING:
    from autoSKZCAM.types import (
        CalculatorInfo,
        ElementStr,
        OniomLayerInfo,
        SkzcamOutput,
    )


def skzcam_eint_flow(
    EmbeddedCluster: CreateSkzcamClusters,
    OniomInfo: dict[str, OniomLayerInfo],
    **kwargs,
):
    """
    The complete SKZCAM protocol to generate the embedded clusters, perform the calculations, and analyze the results.

    Parameters
    ----------
    EmbeddedCluster
        The CreateSkzcamClusters object containing the embedded cluster. This is initialized using the skzcam_initialize() function.
    OniomInfo
        A dictionary containing the information about the ONIOM layers.
    **kwargs
        Additional keyword arguments to pass to the skzcam_generate_job() and skzcam_calculate_job() functions.

    Returns
    -------
    CalculatorInfo
        A dictionary containing the cluster number as key and a dictionary of ASE calculators for the calculations that have been performed.
    """

    # Generate the embedded clusters
    skzcam_generate_job(EmbeddedCluster, **kwargs)

    # Perform the calculations on the embedded clusters
    skzcam_calculate_job(EmbeddedCluster, OniomInfo, **kwargs)

    # Analyze the results
    skzcam_analysis()


def skzcam_initialize(
    adsorbate_indices: list[int],
    slab_center_indices: list[int],
    atom_oxi_states: dict[ElementStr, int],
    adsorbate_slab_file: str | Path,
    pun_file: str | Path | None = None,
) -> CreateSkzcamClusters:
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
    pun_file
        The path to the .pun file containing the atomic coordinates and charges of the adsorbate-slab complex if it has already been generated by ChemShell. If it is None, then ChemShell will need to be used to create this file.
    """

    EmbeddedCluster = CreateSkzcamClusters(
        adsorbate_indices=adsorbate_indices,
        slab_center_indices=slab_center_indices,
        atom_oxi_states=atom_oxi_states,
        adsorbate_slab_file=adsorbate_slab_file,
        pun_file=pun_file,
    )

    EmbeddedCluster.convert_slab_to_atoms()

    return EmbeddedCluster


def skzcam_generate_job(
    EmbeddedCluster: CreateSkzcamClusters,
    max_cluster_num: int = 10,
    shell_width: float = 0.1,
    bond_dist: float = 2.5,
    ecp_dist: float = 6.0,
    write_clusters: bool = False,
    write_clusters_path: str | Path = ".",
    write_include_ecp: bool = False,
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

    if EmbeddedCluster.adsorbate_slab is None:
        EmbeddedCluster.convert_slab_to_atoms()

    # Ensure that the pun_file has been provided in EmbeddedCluster
    if EmbeddedCluster.pun_file is None:
        raise ValueError(
            "The path ('pun_file') to the .pun file from ChemShell must be provided in EmbeddedCluster."
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
    EmbeddedCluster: CreateSkzcamClusters,
    OniomLayerInfo: dict[str, str],
    dryrun: bool = False,
    calc_folder: str | Path = "calc_dir",
):
    """
    Perform the skzcam calculations on the embedded clusters.

    Parameters
    ----------
    EmbeddedCluster
        The CreateSkzcamClusters object containing the embedded cluster.
    OniomLayerInfo
        A dictionary containing the information about the ONIOM layers.

    Returns
    -------
    CalculatorInfo
        A dictionary containing the cluster number as key and a dictionary of ASE calculators for the calculations that have been performed.
    """

    # Prepare the embedded cluster for the calculations
    prep_cluster = Prepare(
        adsorbate_slab_embedded_cluster=EmbeddedCluster.adsorbate_slab_embedded_cluster,
        quantum_cluster_indices_set=EmbeddedCluster.quantum_cluster_indices_set,
        ecp_region_indices_set=EmbeddedCluster.ecp_region_indices_set,
        oniom_layers=OniomLayerInfo,
    )

    # Create the ASE calculators for the embedded cluster
    skzcam_cluster_calculators = prep_cluster.create_cluster_calcs()

    if dryrun:
        skzcam_write_inputs(skzcam_cluster_calculators, calc_folder)
    else:
        skzcam_write_inputs(skzcam_cluster_calculators, calc_folder)
        for cluster_num in skzcam_cluster_calculators:
            for calculation_label in skzcam_cluster_calculators[cluster_num]:
                code = calculation_label.split()[0]
                method = calculation_label.split()[1]
                frozen_core = calculation_label.split()[2]
                basis_set = calculation_label.split()[3]
                for structure in ["adsorbate", "slab", "adsorbate_slab"]:
                    system_path = Path(
                        calc_folder,
                        str(cluster_num),
                        code,
                        f"{method}_{basis_set}_{frozen_core}",
                        structure,
                    )
                    skzcam_cluster_calculators[cluster_num][calculation_label][
                        structure
                    ].calc.directory = system_path
                    skzcam_cluster_calculators[cluster_num][calculation_label][
                        structure
                    ].get_potential_energy()

    return skzcam_cluster_calculators


def skzcam_analysis():
    pass


def chemshell_run_job(
    SkzcamCluster: CreateSkzcamClusters,
    pun_filepath: str | Path = "chemshell_run",
    chemsh_radius_active: float = 40.0,
    chemsh_radius_cluster: float = 60.0,
    chemsh_bq_layer: float = 6.0,
    write_xyz_file: bool = False,
) -> None:
    """
    Recipe to run a ChemShell calculation on the adsorbate-slab system.

    Parameters
    ----------
    pun_filepath
        The path to write the ChemShell .pun files.
    chemsh_radius_active
        The radius of the active region in Angstroms. This 'active' region is simply region where the charge fitting is performed to ensure correct Madelung potential; it can be a relatively large value.
    chemsh_radius_cluster
        The radius of the total embedded cluster in Angstroms.
    chemsh_bq_layer
        The height above the surface to place some additional fitting point charges in Angstroms; simply for better reproduction of the electrostatic potential close to the adsorbate.

    """

    SkzcamCluster.convert_slab_to_atoms()

    # Create the ChemShell input file
    SkzcamCluster.run_chemshell(
        filepath=pun_filepath,
        chemsh_radius_active=chemsh_radius_active,
        chemsh_radius_cluster=chemsh_radius_cluster,
        chemsh_bq_layer=chemsh_bq_layer,
        write_xyz_file=write_xyz_file,
    )


def skzcam_write_inputs(
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
