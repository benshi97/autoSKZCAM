from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

# from ase.io.orca import write_orca
from ase.units import Bohr
from monty.dev import requires
from monty.io import zopen
from monty.os.path import zpath

# from quacc.calculators.mrcc.io import write_mrcc

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from autoSKZCAM.types import SKZCAMInfo, SKZCAMOutput


element_from_atomic_num_dict = {v: k for k, v in atomic_numbers.items()}

semicore_frozen_core_dict = {
    element_from_atomic_num_dict[x]: (
        0
        if x <= 4
        else 2
        if x <= 12
        else 10
        if x <= 30
        else 18
        if x <= 38
        else 28
        if x <= 48
        else 36
        if x <= 71
        else 46
        if x <= 80
        else 68
        if x <= 103
        else None  # You can choose a default value or None for x > 103
    )
    for x in range(1, 104)  # Adjust the range to include up to 103
    if x
    in element_from_atomic_num_dict  # Ensure the atomic number exists in the dictionary
}

valence_frozen_core_dict = {
    element_from_atomic_num_dict[x]: (
        0
        if x <= 2
        else 2
        if x <= 10
        else 10
        if x <= 18
        else 18
        if x <= 30
        else 28
        if x <= 36
        else 36
        if x <= 48
        else 46
        if x <= 54
        else 54
        if x <= 70
        else 68
        if x <= 80
        else 78
        if x <= 86
        else 86
        if x <= 102
        else 100
        if x <= 103
        else None  # You can choose a default value or None for x > 103
    )
    for x in range(1, 104)  # Adjust the range to include up to 103
    if x
    in element_from_atomic_num_dict  # Ensure the atomic number exists in the dictionary
}

skzcam_cation_cap_ecp = {
    "orca": {
        "Ti": """NewECP
N_core 0
  lmax f
  s 2
   1      0.860000       9.191690  2
   2      0.172000       0.008301  2
  p 2
   1      0.860000      17.997720  2
   2      0.172000      -0.032600  2
  d 2
   1      1.600000      -9.504310  2
   2      0.320000      -0.151370  2
  f 1
   1      1.000000000    0.000000000 2
end""",
        "Mg": """NewECP
N_core 0
lmax f
s 1
1      1.732000000   14.676000000 2
p 1
1      1.115000000    5.175700000 2
d 1
1      1.203000000   -1.816000000 2
f 1
1      1.000000000    0.000000000 2
end""",
    },
    "mrcc": {
        "Ti": """
Ti:capECP
Stuttgart RSC 1997 ECP
*
    NCORE = 22    LMAX = 3
f
    0.000000000  2     1.000000000
s-f
    9.191690000  2     0.860000000
    0.008301000  2     0.172000000
p-f
   17.997720000  2     0.860000000
   -0.032600000  2     0.172000000
d-f
   -9.504310000  2     1.600000000
   -0.151370000  2     0.320000000
*
""",
        "Mg": """
Mg:capECP
Stuttgart RLC ECP
*
    NCORE = 12    LMAX = 3
f
    0.000000000  2     1.000000000
s-f
   14.676000000  2     1.732000000
p-f
    5.175700000  2     1.115000000
d-f
   -1.816000000  2     1.203000000
*
""",
    },
}

has_chemshell = find_spec("chemsh") is not None


class PrepareautoSKZCAM:
    """
    From the set of ONIOM calculations, this class generates the calculations needed to be performed for each cluster from the series of clusters generated by the [quacc.atoms.skzcam.CreateSKZCAMClusters][] class.
    """

    def __init__(
        self,
        adsorbate_slab_embedded_cluster: Atoms,
        quantum_cluster_indices_set: list[list[int]],
        ecp_region_indices_set: list[list[int]],
        oniom_layers: dict[str, SKZCAMInfo],
    ) -> None:
        """
        Parameters
        ----------
        adsorbate_slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates and atomic charges from the .pun file, as well as the atom type. This object is created by the [quacc.atoms.skzcam.CreateSKZCAMClusters][] class.
        quantum_cluster_indices_set
            A list of lists containing the indices of the atoms of a set of quantum clusters. These indices are provided by the [quacc.atoms.skzcam.CreateSKZCAMClusters][] class.
        ecp_region_indices_set
            A list of lists containing the indices of the atoms in the ECP region of a set of quantum clusters. These indices are provided by the [quacc.atoms.skzcam.CreateSKZCAMClusters][] class.
        oniom_layers
            A dictionary containing the information for each of the calculations for each of the ONIOM layers. An arbitrary number of ONIOM layers can be included.

        Returns
        -------
        None
        """

        self.adsorbate_slab_embedded_cluster = adsorbate_slab_embedded_cluster
        self.quantum_cluster_indices_set = quantum_cluster_indices_set
        self.ecp_region_indices_set = ecp_region_indices_set
        self.oniom_layers = oniom_layers

        # Check that the quantum_cluster_indices_set and ecp_region_indices_set are the same length
        if len(self.quantum_cluster_indices_set) != len(self.ecp_region_indices_set):
            raise ValueError(
                "The quantum_cluster_indices_set and ecp_region_indices_set must be the same length."
            )

        # Check that all of the necessary keywords are included in each oniom layer

        max_cluster = 0
        for oniom_layer in self.oniom_layers:
            for level in ["ll", "hl"]:
                if self.oniom_layers[oniom_layer][level] is not None:
                    oniom_layer_parameters = self.oniom_layers[oniom_layer][level]
                    # Check that all the required parameters are provided
                    for parameter in [
                        "method",
                        "max_cluster_num",
                        "basis",
                        "frozen_core",
                    ]:
                        if parameter not in oniom_layer_parameters:
                            raise ValueError(
                                f"The {parameter} parameter must be provided for all ONIOM layers specified."
                            )
                    # Check that the maximum cluster number is below the number of quantum clusters for all ONIOM layers
                    if (
                        oniom_layer_parameters["max_cluster_num"]
                        > len(self.quantum_cluster_indices_set)
                        or oniom_layer_parameters["max_cluster_num"] < 1
                    ):
                        raise ValueError(
                            "The maximum cluster number for all ONIOM layers must be bigger than 0 and less than or equal to the number of quantum clusters generated by autoSKZCAM."
                        )
                    if oniom_layer_parameters["max_cluster_num"] > max_cluster:
                        max_cluster = oniom_layer_parameters["max_cluster_num"]

                    # Check that the frozencore is either 'valence' or 'semicore'
                    if oniom_layer_parameters["frozen_core"] not in [
                        "valence",
                        "semicore",
                    ]:
                        raise ValueError(
                            "The frozen_core must be specified as either 'valence' or 'semicore'."
                        )
        self.max_cluster = max_cluster

    def initialize_clusters(self) -> dict[int, dict]:
        """
        Initialize the set of calculations needed to be performed for each cluster from the series of clusters generated by the [quacc.atoms.skzcam.CreateSKZCAMClusters][] class.

        Returns
        -------
        dict[int, dict]
            A dictionary containing the information for each cluster. The information includes the indices of the atoms in the cluster, the indices of the atoms in the ECP region, the calculations to be performed, and the code specific kwargs for each calculation.

        """
        # Set up the dictionary to store the information for each cluster
        skzcam_clusters_information_dict = {
            cluster_num: {
                "cluster indices": self.quantum_cluster_indices_set[cluster_num - 1],
                "ecp_region indices": self.ecp_region_indices_set[cluster_num - 1],
                "cluster calculations": [],
                "cluster calculations code kwargs": [],
            }
            for cluster_num in range(1, self.max_cluster + 1)
        }
        for oniom_layer in self.oniom_layers:
            for level in ["ll", "hl"]:
                if self.oniom_layers[oniom_layer][level] is not None:
                    oniom_layer_parameters = self.oniom_layers[oniom_layer][level]
                    frozen_core = oniom_layer_parameters["frozen_core"]
                    method = oniom_layer_parameters["method"]
                    if "CBS" in oniom_layer_parameters["basis"]:
                        basis_sets = [
                            oniom_layer_parameters["basis"][4:-1].split("/")[0],
                            oniom_layer_parameters["basis"][4:-1].split("/")[1],
                        ]
                    else:
                        basis_sets = [oniom_layer_parameters["basis"]]
                    for basis_set in basis_sets:
                        calculation_label = f"{method} {frozen_core} {basis_set}"
                        for cluster_num in range(
                            1, oniom_layer_parameters["max_cluster_num"] + 1
                        ):
                            if (
                                calculation_label
                                not in skzcam_clusters_information_dict[cluster_num][
                                    "cluster calculations"
                                ]
                            ):
                                skzcam_clusters_information_dict[cluster_num][
                                    "cluster calculations"
                                ].append(calculation_label)
                                skzcam_clusters_information_dict[cluster_num][
                                    "cluster calculations code kwargs"
                                ].append(oniom_layer_parameters["code kwargs"])
        return skzcam_clusters_information_dict


class CreateSKZCAMClusters:
    """
    A class to create the quantum clusters and ECP regions for the SKZCAM protocol.

    Attributes
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
        The path to the .pun file containing the atomic coordinates and charges of the adsorbate-slab complex. This file should be generated by ChemShell. If it is None, then ChemShell wil be used to create this file.
    adsorbate
        The ASE Atoms object containing the atomic coordinates of the adsorbate.
    slab
        The ASE Atoms object containing the atomic coordinates of the slab.
    adsorbate_slab
        The ASE Atoms object containing the atomic coordinates of the adsorbate-slab complex.
    adsorbate_slab_embedded_cluster
        The ASE Atoms object containing the atomic coordinates, atomic charges and atom type (i.e., point charge or cation/anion) from the .pun file for the embedded cluster of the adsorbate-slab complex.
    slab_embedded_cluster
        The ASE Atoms object containing the atomic coordinates, atomic charges and atom type (i.e., point charge or cation/anion) from the .pun file for the embedded cluster of the slab.
    quantum_cluster_indices_set
        A list of lists of indices of the atoms in the set of quantum clusters created by the SKZCAM protocol
    ecp_region_indices_set
        A list of lists of indices of the atoms in the ECP region for the set of quantum clusters created by the SKZCAM protocol
    """

    def __init__(
        self,
        adsorbate_indices: list[int],
        slab_center_indices: list[int],
        atom_oxi_states: dict[str, int],
        adsorbate_slab_file: str | Path | None = None,
        pun_file: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        adsorbate_indices
            The indices of the atoms that make up the adsorbate molecule.
        slab_center_indices
            The indices of the atoms that make up the 'center' of the slab right beneath the adsorbate.
        atom_oxi_states
            A dictionary with the element symbol as the key and its oxidation state as the value.
        adsorbate_slab_file
            The path to the file containing the adsorbate molecule on the surface slab. It can be in any format that ASE can read.
        pun_file
            The path to the .pun file containing the atomic coordinates and charges of the adsorbate-slab complex. This file should be generated by ChemShell. If it is None, then ChemShell wil be used to create this file.

        Returns
        -------
        None
        """

        self.adsorbate_indices = adsorbate_indices
        self.slab_center_indices = slab_center_indices
        self.slab_indices = None  # This will be set later
        self.atom_oxi_states = atom_oxi_states
        self.adsorbate_slab_file = adsorbate_slab_file
        self.pun_file = pun_file

        # Check that the adsorbate_indices and slab_center_indices are not the same
        if any(x in self.adsorbate_indices for x in self.slab_center_indices):
            raise ValueError(
                "The adsorbate and slab center indices cannot be the same."
            )

        # Check that the adsorbate_slab_file and pun_file are not both None
        if self.adsorbate_slab_file is None and self.pun_file is None:
            raise ValueError(
                "Either the adsorbate_slab_file or pun_file must be provided."
            )

        # Initialize the adsorbate, slab and adsorbate_slab Atoms object which contains the adsorbate, slab and adsorbate-slab complex respectively
        self.adsorbate: Atoms | None
        self.slab: Atoms | None
        self.adsorbate_slab: Atoms | None

        # Initialize the embedded_adsorbate_slab_cluster, and embedded_slab_cluster Atoms object which are the embedded cluster for the adsorbate-slab complex and slab respectively
        self.adsorbate_slab_embedded_cluster: Atoms | None = None
        self.slab_embedded_cluster: Atoms | None = None

        # Initialize the quantum cluster indices and ECP region indices
        self.quantum_cluster_indices_set: list[list[int]] | None = None
        self.ecp_region_indices_set: list[list[int]] | None = None

    def convert_slab_to_atoms(self) -> None:
        """
        Read the file containing the periodic slab and adsorbate (geometry optimized) and format the resulting Atoms object to be used to create a .pun file in ChemShell.

        Returns
        -------
        None
        """

        # Get the necessary information for the cluster from a provided slab file (in any format that ASE can read)
        adsorbate_slab = read(self.adsorbate_slab_file)

        # Find indices (within adsorbate_slab) of the slab
        slab_indices = self.slab_center_indices + [
            i
            for i, _ in enumerate(adsorbate_slab)
            if i not in (self.adsorbate_indices + self.slab_center_indices)
        ]

        # Create adsorbate and slab from adsorbate_slab
        slab = adsorbate_slab[slab_indices]
        adsorbate = adsorbate_slab[self.adsorbate_indices]

        adsorbate.translate(-slab[0].position)
        slab.translate(-slab[0].position)

        # Get the relative distance of the adsorbate from the first center atom of the slab as defined in the slab_center_indices
        adsorbate_vector_from_slab = adsorbate[0].position - slab[0].position

        # Get the center of the cluster from the slab_center_indices
        slab_center_position = slab[
            : len(self.slab_center_indices)
        ].get_positions().sum(axis=0) / len(self.slab_center_indices)

        # Add the height of the adsorbate from the slab along the z-direction relative to the slab_center
        adsorbate_com_z_disp = (
            adsorbate.get_center_of_mass()[2] - slab_center_position[2]
        )

        center_position = (
            np.array([0.0, 0.0, adsorbate_com_z_disp]) + slab_center_position
        )

        self.adsorbate = adsorbate
        self.slab = slab
        self.adsorbate_slab = adsorbate_slab
        self.adsorbate_vector_from_slab = adsorbate_vector_from_slab
        self.center_position = center_position

    @requires(has_chemshell, "ChemShell is not installed")
    def run_chemshell(
        self,
        filepath: str | Path,
        chemsh_radius_active: float = 40.0,
        chemsh_radius_cluster: float = 60.0,
        chemsh_bq_layer: float = 6.0,
        write_xyz_file: bool = False,
    ) -> None:
        """
        Run ChemShell to create an embedded cluster from a slab.

        Parameters
        ----------
        filepath
            The location where the ChemShell output files will be written.
        chemsh_radius_active
            The radius of the active region in Angstroms. This 'active' region is simply region where the charge fitting is performed to ensure correct Madelung potential; it can be a relatively large value.
        chemsh_radius_cluster
            The radius of the total embedded cluster in Angstroms.
        chemsh_bq_layer
            The height above the surface to place some additional fitting point charges in Angstroms; simply for better reproduction of the electrostatic potential close to the adsorbate.
        write_xyz_file
            Whether to write an XYZ file of the cluster for visualisation.

        Returns
        -------
        None
        """
        from chemsh.io.tools import convert_atoms_to_frag

        # Convert ASE Atoms to ChemShell Fragment object
        slab_frag = convert_atoms_to_frag(self.slab, connect_mode="ionic", dim="2D")

        # Add the atomic charges to the fragment
        slab_frag.addCharges(self.atom_oxi_states)

        # Create the chemshell cluster (i.e., add electrostatic fitting charges) from the fragment
        chemsh_slab_embedded_cluster = slab_frag.construct_cluster(
            origin=0,
            radius_cluster=chemsh_radius_cluster / Bohr,
            radius_active=chemsh_radius_active / Bohr,
            bq_layer=chemsh_bq_layer / Bohr,
            adjust_charge="coordination_scaled",
        )

        # Save the final cluster to a .pun file
        chemsh_slab_embedded_cluster.save(
            filename=Path(filepath).with_suffix(".pun"), fmt="pun"
        )
        self.pun_file = Path(filepath).with_suffix(".pun")

        if write_xyz_file:
            # XYZ for visualisation
            chemsh_slab_embedded_cluster.save(
                filename=Path(filepath).with_suffix(".xyz"), fmt="xyz"
            )

    def run_skzcam(
        self,
        shell_max: int = 10,
        shell_width: float = 0.1,
        bond_dist: float = 2.5,
        ecp_dist: float = 6.0,
        write_clusters: bool = False,
        write_clusters_path: str | Path = ".",
        write_include_ecp: bool = False,
    ) -> SKZCAMOutput:
        """
        From a provided .pun file (generated by ChemShell), this function creates quantum clusters using the SKZCAM protocol. It will return the embedded cluster Atoms object and the indices of the atoms in the quantum clusters and the ECP region. The number of clusters created is controlled by the rdf_max parameter.

        Parameters
        ----------
        shell_max
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

        Returns
        -------
        dict[str, Atoms]
            A dictionary containing the embedded cluster Atoms object of the adsorbate-slab complex, accessed by the key 'adsorbate_slab_embedded_cluster', and the indices of the atoms in the quantum clusters and the ECP region, accessed by the keys 'quantum_cluster_indices_set' and 'ecp_region_indices_set' respectively.
        """

        # Read the .pun file and create the embedded_cluster Atoms object
        self.slab_embedded_cluster = self._convert_pun_to_atoms(pun_file=self.pun_file)

        # Get distances of all atoms from the cluster center
        atom_center_distances = _get_atom_distances(
            atoms=self.slab_embedded_cluster, center_position=self.center_position
        )

        # Determine the cation shells from the center of the embedded cluster
        _, cation_shells_idx = self._find_cation_shells(
            slab_embedded_cluster=self.slab_embedded_cluster,
            distances=atom_center_distances,
            shell_width=shell_width,
        )

        # Create the distance matrix for the embedded cluster
        slab_embedded_cluster_all_dist = self.slab_embedded_cluster.get_all_distances()

        # Create the anion coordination list for each cation shell
        anion_coord_idx = []
        for shell_idx in range(shell_max):
            shell_indices = cation_shells_idx[shell_idx]
            anion_coord_idx += [
                self._get_anion_coordination(
                    slab_embedded_cluster=self.slab_embedded_cluster,
                    cation_shell_indices=shell_indices,
                    dist_matrix=slab_embedded_cluster_all_dist,
                    bond_dist=bond_dist,
                )
            ]

        # Create the quantum clusters by summing up the indices of the cations and their coordinating anions
        slab_quantum_cluster_indices_set = []
        dummy_cation_indices = []
        dummy_anion_indices = []
        for shell_idx in range(shell_max):
            dummy_cation_indices += cation_shells_idx[shell_idx]
            dummy_anion_indices += anion_coord_idx[shell_idx]
            slab_quantum_cluster_indices_set += [
                list(set(dummy_cation_indices + dummy_anion_indices))
            ]

        # Get the ECP region for each quantum cluster
        slab_ecp_region_indices_set = self._get_ecp_region(
            slab_embedded_cluster=self.slab_embedded_cluster,
            quantum_cluster_indices_set=slab_quantum_cluster_indices_set,
            dist_matrix=slab_embedded_cluster_all_dist,
            ecp_dist=ecp_dist,
        )

        # Create the adsorbate_slab_embedded_cluster from slab_embedded_cluster and adsorbate atoms objects. This also sets the final quantum_cluster_indices_set and ecp_region_indices_set for the adsorbate_slab_embedded_cluster
        self._create_adsorbate_slab_embedded_cluster(
            quantum_cluster_indices_set=slab_quantum_cluster_indices_set,
            ecp_region_indices_set=slab_ecp_region_indices_set,
        )

        # Write the quantum clusters to files
        if write_clusters:
            for idx in range(len(self.quantum_cluster_indices_set)):
                quantum_atoms = self.adsorbate_slab_embedded_cluster[
                    self.quantum_cluster_indices_set[idx]
                ]
                if write_include_ecp:
                    ecp_atoms = self.adsorbate_slab_embedded_cluster[
                        self.ecp_region_indices_set[idx]
                    ]
                    ecp_atoms.set_chemical_symbols(np.array(["U"] * len(ecp_atoms)))
                    cluster_atoms = quantum_atoms + ecp_atoms
                else:
                    cluster_atoms = quantum_atoms
                write(
                    Path(write_clusters_path, f"SKZCAM_cluster_{idx}.xyz"),
                    cluster_atoms,
                )

        return {
            "adsorbate_slab_embedded_cluster": self.adsorbate_slab_embedded_cluster,
            "quantum_cluster_indices_set": self.quantum_cluster_indices_set,
            "ecp_region_indices_set": self.ecp_region_indices_set,
        }

    def _convert_pun_to_atoms(self, pun_file: str | Path) -> Atoms:
        """
        Reads a .pun file and returns an ASE Atoms object containing the atomic coordinates,
        point charges/oxidation states, and atom types.

        Parameters
        ----------
        pun_file
            The path to the .pun file created by ChemShell to be read.

        Returns
        -------
        Atoms
            The ASE Atoms object containing the atomic coordinates and atomic charges from the .pun file.
            The `oxi_states` array contains the atomic charges, and the `atom_type` array contains the
            atom types (cation, anion, neutral).
        """

        # Create a dictionary containing the atom types and whether they are cations or anions
        atom_type_dict = {
            atom: "cation" if oxi_state > 0 else "anion" if oxi_state < 0 else "neutral"
            for atom, oxi_state in self.atom_oxi_states.items()
        }

        # Load the pun file as a list of strings
        with zopen(zpath(str(Path(pun_file)))) as f:
            raw_pun_file = [
                line.rstrip().decode("utf-8")
                if isinstance(line, bytes)
                else line.rstrip()
                for line in f
            ]

        # Get the number of atoms and number of atomic charges in the .pun file
        n_atoms = int(raw_pun_file[3].split()[-1])
        n_charges = int(raw_pun_file[4 + n_atoms - 1 + 3].split()[-1])

        # Check if number of atom charges same as number of atom positions
        if n_atoms != n_charges:
            raise ValueError(
                "Number of atomic positions and atomic charges in the .pun file are not the same."
            )

        raw_atom_positions = raw_pun_file[4 : 4 + n_atoms]
        raw_charges = raw_pun_file[7 + n_atoms : 7 + 2 * n_atoms]
        charges = [float(charge) for charge in raw_charges]

        # Add the atomic positions the embedded_cluster Atoms object (converting from Bohr to Angstrom)
        atom_types = []
        atom_numbers = []
        atom_positions = []
        for _, line in enumerate(raw_atom_positions):
            line_info = line.split()

            # Add the atom type to the atom_type_list
            if line_info[0] in atom_type_dict:
                atom_types.append(atom_type_dict[line_info[0]])
            elif line_info[0] == "F":
                atom_types.append("pc")
            else:
                atom_types.append("unknown")

            # Add the atom number to the atom_number_list and position to the atom_position_list
            atom_numbers += [atomic_numbers[line_info[0]]]
            atom_positions += [
                [
                    float(line_info[1]) * Bohr,
                    float(line_info[2]) * Bohr,
                    float(line_info[3]) * Bohr,
                ]
            ]

        slab_embedded_cluster = Atoms(numbers=atom_numbers, positions=atom_positions)

        # Center the embedded cluster so that atom index 0 is at the [0, 0, 0] position
        slab_embedded_cluster.translate(-slab_embedded_cluster[0].position)

        # Add the `oxi_states` and `atom_type` arrays to the Atoms object
        slab_embedded_cluster.set_array("oxi_states", np.array(charges))
        slab_embedded_cluster.set_array("atom_type", np.array(atom_types))

        return slab_embedded_cluster

    def _create_adsorbate_slab_embedded_cluster(
        self,
        quantum_cluster_indices_set: list[list[int]] | None = None,
        ecp_region_indices_set: list[list[int]] | None = None,
    ) -> None:
        """
        Insert the adsorbate into the embedded cluster and update the quantum cluster and ECP region indices.

        Parameters
        ----------
        quantum_cluster_indices_set
            A list of lists containing the indices of the atoms in each quantum cluster.
        ecp_region_indices_set
            A list of lists containing the indices of the atoms in the ECP region for each quantum cluster.

        Returns
        -------
        None
        """

        # Remove PBC from the adsorbate
        self.adsorbate.set_pbc(False)

        # Translate the adsorbate to the correct position relative to the slab
        self.adsorbate.translate(
            self.slab_embedded_cluster[0].position
            - self.adsorbate[0].position
            + self.adsorbate_vector_from_slab
        )

        # Set oxi_state and atom_type arrays for the adsorbate
        self.adsorbate.set_array("oxi_states", np.array([0.0] * len(self.adsorbate)))
        self.adsorbate.set_array(
            "atom_type", np.array(["adsorbate"] * len(self.adsorbate))
        )

        # Add the adsorbate to the embedded cluster
        self.adsorbate_slab_embedded_cluster = (
            self.adsorbate + self.slab_embedded_cluster
        )

        # Update the quantum cluster and ECP region indices
        if quantum_cluster_indices_set is not None:
            quantum_cluster_indices_set = [
                list(range(len(self.adsorbate)))
                + [idx + len(self.adsorbate) for idx in cluster]
                for cluster in quantum_cluster_indices_set
            ]
        if ecp_region_indices_set is not None:
            ecp_region_indices_set = [
                [idx + len(self.adsorbate) for idx in cluster]
                for cluster in ecp_region_indices_set
            ]

        self.quantum_cluster_indices_set = quantum_cluster_indices_set
        self.ecp_region_indices_set = ecp_region_indices_set

    def _find_cation_shells(
        self, slab_embedded_cluster: Atoms, distances: NDArray, shell_width: float = 0.1
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Returns a list of lists containing the indices of the cations in each shell, based on distance from the embedded cluster center.
        This is achieved by clustering the data based on the DBSCAN clustering algorithm.

        Parameters
        ----------
        slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates AND the atom types (i.e. cation or anion).
        distances
            The distance of atoms from the cluster center.
        shell_width
            Defines the distance between atoms within shells; this is the maximum distance between any two atoms within the shell

        Returns
        -------
        list[list[int]]
            A list of lists containing the distance of the cation in each shell from the adsorbate.
        list[list[int]]
            A list of lists containing the indices of the cations in each shell.
        """

        # Define the empty list to store the cation shells
        shells_distances = []
        shells_indices = []

        # Sort the points by distance from the cluster center for the cations only
        distances_sorted = []
        distances_sorted_indices = []
        for i in np.argsort(distances):
            if slab_embedded_cluster.get_array("atom_type")[i] == "cation":
                distances_sorted.append(distances[i])
                distances_sorted_indices.append(i)

        current_point = distances_sorted[0]
        current_shell = [current_point]
        current_shell_idx = [distances_sorted_indices[0]]

        for idx, point in enumerate(distances_sorted[1:]):
            if point <= current_point + shell_width:
                current_shell.append(point)
                current_shell_idx.append(distances_sorted_indices[idx + 1])
            else:
                shells_distances.append(current_shell)
                shells_indices.append(current_shell_idx)
                current_shell = [point]
                current_shell_idx = [distances_sorted_indices[idx + 1]]
            current_point = point
        shells_distances.append(current_shell)
        shells_indices.append(current_shell_idx)

        return shells_distances, shells_indices

    def _get_anion_coordination(
        self,
        slab_embedded_cluster: Atoms,
        cation_shell_indices: list[int],
        dist_matrix: NDArray,
        bond_dist: float = 2.5,
    ) -> list[int]:
        """
        Returns a list of lists containing the indices of the anions coordinating the cation indices provided.

        Parameters
        ----------
        slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates AND the atom types (i.e. cation or anion).
        cation_shell_indices
            A list of the indices of the cations in the cluster.
        dist_matrix
            A matrix containing the distances between each pair of atoms in the embedded cluster.
        bond_dist
            The distance within which an anion is considered to be coordinating a cation.

        Returns
        -------
        list[int]
            A list containing the indices of the anions coordinating the cation indices.
        """

        # Define the empty list to store the anion coordination
        anion_coord_indices = []

        # Iterate over the cation shell indices and find the atoms within the bond distance of each cation
        for atom_idx in cation_shell_indices:
            anion_coord_indices += [
                idx
                for idx, dist in enumerate(dist_matrix[atom_idx])
                if (
                    dist < bond_dist
                    and slab_embedded_cluster.get_array("atom_type")[idx] == "anion"
                )
            ]

        return list(set(anion_coord_indices))

    def _get_ecp_region(
        self,
        slab_embedded_cluster: Atoms,
        quantum_cluster_indices_set: list[int],
        dist_matrix: NDArray,
        ecp_dist: float = 6.0,
    ) -> list[list[int]]:
        """
        Returns a list of lists containing the indices of the atoms in the ECP region of the embedded cluster for each quantum cluster

        Parameters
        ----------
        slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates AND the atom types (i.e. cation or anion).
        quantum_cluster_indices_set
            A list of lists containing the indices of the atoms in each quantum cluster.
        dist_matrix
            A matrix containing the distances between each pair of atoms in the embedded cluster.
        ecp_dist
            The distance from edges of the quantum cluster to define the ECP region.

        Returns
        -------
        list[list[int]]
            A list of lists containing the indices of the atoms in the ECP region for each quantum cluster.
        """

        ecp_region_indices_set = []
        dummy_cation_indices = []

        # Iterate over the quantum clusters and find the atoms within the ECP distance of each quantum cluster
        for cluster in quantum_cluster_indices_set:
            dummy_cation_indices += cluster
            cluster_ecp_region_idx = []
            for atom_idx in dummy_cation_indices:
                for idx, dist in enumerate(dist_matrix[atom_idx]):
                    # Check if the atom is within the ecp_dist region and is not in the quantum cluster and is a cation
                    if (
                        dist < ecp_dist
                        and idx not in dummy_cation_indices
                        and slab_embedded_cluster.get_array("atom_type")[idx]
                        == "cation"
                    ):
                        cluster_ecp_region_idx += [idx]

            ecp_region_indices_set += [list(set(cluster_ecp_region_idx))]

        return ecp_region_indices_set


def _get_atom_distances(atoms: Atoms, center_position: NDArray) -> NDArray:
    """
    Returns the distance of all atoms from the center position of the embedded cluster

    Parameters
    ----------
    embedded_cluster
        The ASE Atoms object containing the atomic coordinates of the embedded cluster.
    center_position
        The position of the center of the embedded cluster (i.e., position of the adsorbate).

    Returns
    -------
    NDArray
        An array containing the distances of each atom in the Atoms object from the cluster center.
    """

    return np.array([np.linalg.norm(atom.position - center_position) for atom in atoms])
