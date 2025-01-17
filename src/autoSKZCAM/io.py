from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# from ase.io.orca import write_orca

# from quacc.calculators.mrcc.io import write_mrcc

if TYPE_CHECKING:
    from ase.atoms import Atom, Atoms

    from autoSKZCAM.types import (
        BlockInfo,
        ElementInfo,
        ElementStr,
        MrccInputDict,
        MultiplicityDict,
    )


class MRCCInputGenerator:
    """
    A class to generate the SKZCAM input for the MRCC ASE calculator.
    """

    def __init__(
        self,
        adsorbate_slab_embedded_cluster: Atoms,
        quantum_cluster_indices: list[int],
        ecp_region_indices: list[int],
        element_info: dict[ElementStr, ElementInfo],
        include_cp: bool = True,
        multiplicities: MultiplicityDict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        adsorbate_slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates and atomic charges from the .pun file, as well as the atom type. This object is created within the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        quantum_cluster_indices
            A list containing the indices of the atoms in one quantum cluster. These indices are created within the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        ecp_region_indices
            A list containing the indices of the atoms in the corresponding ECP region of one quantum cluster. These indices are provided by the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        element_info
            A dictionary with elements as keys which gives the (1) number of core electrons as 'core', (2) basis set as 'basis', (3) effective core potential as 'ecp', (4) resolution-of-identity/density-fitting auxiliary basis set for DFT/HF calculations as 'ri_scf_basis' and (5) resolution-of-identity/density-fitting for correlated wave-function methods as 'ri_cwft_basis'.
        include_cp
            If True, the coords strings will include the counterpoise correction (i.e., ghost atoms) for the adsorbate and slab.
        multiplicities
            The multiplicity of the adsorbate-slab complex, adsorbate and slab respectively, with the keys 'adsorbate_slab', 'adsorbate', and 'slab'.

        Returns
        -------
        None
        """

        self.adsorbate_slab_embedded_cluster = adsorbate_slab_embedded_cluster
        self.quantum_cluster_indices = quantum_cluster_indices
        self.ecp_region_indices = ecp_region_indices
        self.element_info = element_info
        self.include_cp = include_cp
        self.multiplicities = (
            {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1}
            if multiplicities is None
            else multiplicities
        )

        # Check that none of the indices in quantum_cluster_indices are in ecp_region_indices
        if not np.all(
            [x not in self.ecp_region_indices for x in self.quantum_cluster_indices]
        ):
            raise ValueError(
                "An atom in the quantum cluster is also in the ECP region."
            )

        # Create the adsorbate-slab complex quantum cluster and ECP region cluster
        self.adsorbate_slab_cluster: Atoms = self.adsorbate_slab_embedded_cluster[
            self.quantum_cluster_indices
        ]
        self.ecp_region: Atoms = self.adsorbate_slab_embedded_cluster[
            self.ecp_region_indices
        ]

        # Get the indices of the adsorbates from the quantum cluster
        self.adsorbate_indices: list[int] = [
            i
            for i in range(len(self.adsorbate_slab_cluster))
            if self.adsorbate_slab_cluster.get_array("atom_type")[i] == "adsorbate"
        ]
        # Get the indices of the slab from the quantum cluster
        self.slab_indices: list[int] = [
            i
            for i in range(len(self.adsorbate_slab_cluster))
            if self.adsorbate_slab_cluster.get_array("atom_type")[i] != "adsorbate"
        ]

        # Create the adsorbate and slab quantum clusters
        self.adsorbate_cluster: Atoms = self.adsorbate_slab_cluster[
            self.adsorbate_indices
        ]
        self.slab_cluster: Atoms = self.adsorbate_slab_cluster[self.slab_indices]

        # Initialize the SKZCAM MRCC input strings for the adsorbate-slab complex, adsorbate, and slab in the same fashion as for ORCAInputGenerator.orcablocks
        self.skzcam_input_str: BlockInfo = {
            "adsorbate_slab": "",
            "adsorbate": "",
            "slab": "",
        }

        # Initialize the dictionary with keyword and values pairs for MRCC input
        self.skzcam_input_dict: MrccInputDict = {
            "adsorbate_slab": {},
            "adsorbate": {},
            "slab": {},
        }

    def generate_input(self) -> MrccInputDict:
        """
        Creates the mrccinput input for the MRCC ASE calculator.

        Returns
        -------
        MrccInputDict
            A dictionary of key-value pairs (to be put in 'mrccinput' parameter) for the adsorbate-slab complex, the adsorbate, and the slab.
        """

        def _convert_input_str_to_dict(input_str: str) -> dict[str, str]:
            """
            Convert the SKZCAM input string to a dictionary.

            Parameters
            ----------
            input_str
                The SKZCAM input string containing all the input parameters for the SKZCAM protocol (i.e., basis, ecp, geometry, point charges)

            Returns
            -------
            dict[str,str]
                The SKZCAM input as a dictionary where each key is the input parameter and the value is the value of that parameter.
            """

            input_dict = {}

            key = None

            for line in input_str.split("\n"):
                if "=" in line:
                    key = line.split("=")[0]
                    input_dict[key] = line.split("=")[1]
                elif key is not None:
                    input_dict[key] += "\n" + line

            return input_dict

        # Create the blocks for the basis sets (basis, basis_sm, dfbasis_scf, dfbasis_cor, ecp)
        self._generate_basis_ecp_block()

        # Create the blocks for the coordinates
        self._generate_coords_block()

        # Create the point charge block and add it to the adsorbate-slab complex and slab blocks
        point_charge_block = self._generate_point_charge_block()
        self.skzcam_input_str["adsorbate_slab"] += point_charge_block
        self.skzcam_input_str["slab"] += point_charge_block

        # Convert the input string to a dictionary
        self.skzcam_input_dict["adsorbate_slab"] = _convert_input_str_to_dict(
            self.skzcam_input_str["adsorbate_slab"]
        )
        self.skzcam_input_dict["adsorbate"] = _convert_input_str_to_dict(
            self.skzcam_input_str["adsorbate"]
        )
        self.skzcam_input_dict["slab"] = _convert_input_str_to_dict(
            self.skzcam_input_str["slab"]
        )

        return self.skzcam_input_dict

    def create_genbas_file(self) -> str:
        """
        Create a GENBAS file that can be read by MRCC. This contains the capped ECP in CFOUR format as well as empty basis sets for the capped ECP.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The text for the GENBAS file for the electrostatic embedding.
        """
        # Get set of element symbols from the ecp_region
        element_symbols = list(set(self.ecp_region.get_chemical_symbols()))

        genbas_file = ""
        for element in element_symbols:
            genbas_file += f"{element}:cappedECP\nINSERT_cappedECP\n\n"
            genbas_file += f"{element}:no-basis-set\nno basis set\n\n    0\n    0\n    0\n    0\n\n"
            genbas_file += f"{element}:no-basis-set-ri-jk\nno basis set\n\n    0\n    0\n    0\n    0\n\n"

        return genbas_file

    def _generate_basis_ecp_block(self) -> None:
        """
        Generates the basis and ECP block for the MRCC input file.

        Returns
        -------
        None
        """

        # Helper to generate basis strings for MRCC
        def _create_basis_block(quantum_region, ecp_region=None):
            atomtype_ecp = "ecp=special\n"
            for atom in quantum_region:
                if (
                    "ecp" in self.element_info[atom.symbol]
                    and self.element_info[atom.symbol]["ecp"] is not None
                ):
                    atomtype_ecp += f"{self.element_info[atom.symbol]['ecp']}\n"
                else:
                    atomtype_ecp += "none\n"
            if ecp_region is not None:
                atomtype_ecp += "cappedECP\n" * len(ecp_region)

            atomtype_ecp += "\n"

            return (
                f"""
basis_sm=special
{self._create_atomtype_basis(quantum_region=quantum_region, ecp_region=ecp_region, element_basis_info={element: 'def2-SVP' for element in self.element_info})}

basis=special
{self._create_atomtype_basis(quantum_region=quantum_region, ecp_region=ecp_region, element_basis_info={element: self.element_info[element]['basis'] for element in self.element_info})}

dfbasis_scf=special
{self._create_atomtype_basis(quantum_region=quantum_region, ecp_region=ecp_region, element_basis_info={element: self.element_info[element]['ri_scf_basis'] for element in self.element_info})}

dfbasis_cor=special
{self._create_atomtype_basis(quantum_region=quantum_region, ecp_region=ecp_region, element_basis_info={element: self.element_info[element]['ri_cwft_basis'] for element in self.element_info})}

"""
                + atomtype_ecp
            )

        if self.include_cp:
            self.skzcam_input_str["adsorbate_slab"] += _create_basis_block(
                quantum_region=self.adsorbate_slab_cluster, ecp_region=self.ecp_region
            )
            self.skzcam_input_str["slab"] += _create_basis_block(
                quantum_region=self.adsorbate_slab_cluster, ecp_region=self.ecp_region
            )
            self.skzcam_input_str["adsorbate"] += _create_basis_block(
                quantum_region=self.adsorbate_slab_cluster, ecp_region=None
            )
        else:
            self.skzcam_input_str["adsorbate_slab"] += _create_basis_block(
                quantum_region=self.adsorbate_slab_cluster, ecp_region=self.ecp_region
            )
            self.skzcam_input_str["slab"] += _create_basis_block(
                quantum_region=self.slab_cluster, ecp_region=self.ecp_region
            )
            self.skzcam_input_str["adsorbate"] += _create_basis_block(
                quantum_region=self.adsorbate_cluster, ecp_region=None
            )

    def _create_atomtype_basis(
        self,
        quantum_region: Atoms,
        element_basis_info: dict[ElementStr, str],
        ecp_region: Atoms | None = None,
    ) -> str:
        """
        Creates a column for the basis set for each atom in the Atoms object, given by element_info.

        Parameters
        ----------
        quantum_region
            The ASE Atoms object containing the atomic coordinates of the quantum cluster region (could be the adsorbate-slab complex, slab or adsorbate by itself).
        element_basis_info
            A dictionary with elements as keys which gives the basis set for each element.
        ecp_region
            The ASE atoms object containing the atomic coordinates of the capped ECP region.

        Returns
        -------
        str
            The basis set for each atom in the Atoms object given as a column (of size N, where N is the number of atoms).
        """

        basis_str = ""
        for atom in quantum_region:
            basis_str += f"{element_basis_info[atom.symbol]}\n"
        if ecp_region is not None:
            basis_str += "no-basis-set\n" * len(ecp_region)

        return basis_str

    def _generate_coords_block(self) -> None:
        """
        Generates the coordinates block for the MRCC input file. This includes the coordinates of the quantum cluster, the ECP region, and the point charges. It will return three strings for the adsorbate-slab complex, adsorbate and slab.

        Returns
        -------
        None
        """

        # Get the charge of the quantum cluster
        charge = int(sum(self.adsorbate_slab_cluster.get_array("oxi_states")))

        # Get the total number of core electrons for the quantum cluster
        core = {
            "adsorbate_slab": sum(
                [
                    self.element_info[atom.symbol]["core"]
                    for atom in self.adsorbate_slab_cluster
                ]
            ),
            "adsorbate": sum(
                [
                    self.element_info[atom.symbol]["core"]
                    for atom in self.adsorbate_cluster
                ]
            ),
            "slab": sum(
                [self.element_info[atom.symbol]["core"] for atom in self.slab_cluster]
            ),
        }

        # Add the charge and core electron information to skzcam_input_str
        self.skzcam_input_str["adsorbate_slab"] += f"""charge={charge}
mult={self.multiplicities['adsorbate_slab']}
core={int(core['adsorbate_slab']/2)}
unit=angs
geom=xyz
"""
        self.skzcam_input_str["adsorbate"] += f"""charge=0
mult={self.multiplicities['adsorbate']}
core={int(core['adsorbate']/2)}
unit=angs
geom=xyz
"""
        self.skzcam_input_str["slab"] += f"""charge={charge}
mult={self.multiplicities['slab']}
core={int(core['slab']/2)}
unit=angs
geom=xyz
"""
        # Create the atom coordinates block for the adsorbate-slab cluster, ECP region
        adsorbate_slab_coords_block = ""
        for atom in self.adsorbate_slab_cluster:
            adsorbate_slab_coords_block += create_atom_coord_string(atom=atom)

        ecp_region_block = ""
        for ecp_atom in self.ecp_region:
            ecp_region_block += create_atom_coord_string(atom=ecp_atom)

        # Set the number of atoms for each system. This would be the number of atoms in the quantum cluster plus the number of atoms in the ECP region. If include_cp is True, then the number of atoms in the quantum cluster is the number of atoms in the adsorbate-slab complex for both the adsorbate and slab.
        if self.include_cp:
            self.skzcam_input_str["adsorbate_slab"] += (
                f"{len(self.adsorbate_slab_cluster) + len(self.ecp_region)}\n\n"
            )
            self.skzcam_input_str["adsorbate_slab"] += (
                adsorbate_slab_coords_block + ecp_region_block
            )

            self.skzcam_input_str["adsorbate"] += (
                f"{len(self.adsorbate_slab_cluster)}\n\n"
            )
            self.skzcam_input_str["adsorbate"] += adsorbate_slab_coords_block

            self.skzcam_input_str["slab"] += (
                f"{len(self.adsorbate_slab_cluster) + len(self.ecp_region)}\n\n"
            )
            self.skzcam_input_str["slab"] += (
                adsorbate_slab_coords_block + ecp_region_block
            )

            for system in ["adsorbate_slab", "adsorbate", "slab"]:
                self.skzcam_input_str[system] += "\nghost=serialno\n"
                # Add the ghost atoms for the counterpoise correction in the adsorbate and slab
                if system == "adsorbate":
                    self.skzcam_input_str[system] += ",".join(
                        [str(atom_idx + 1) for atom_idx in self.slab_indices]
                    )
                elif system == "slab":
                    self.skzcam_input_str[system] += ",".join(
                        [str(atom_idx + 1) for atom_idx in self.adsorbate_indices]
                    )
                self.skzcam_input_str[system] += "\n\n"
        else:
            self.skzcam_input_str["adsorbate_slab"] += (
                f"{len(self.adsorbate_slab_cluster) + len(self.ecp_region)}\n\n"
            )
            self.skzcam_input_str["adsorbate_slab"] += (
                adsorbate_slab_coords_block + ecp_region_block
            )

            self.skzcam_input_str["adsorbate"] += f"{len(self.adsorbate_cluster)}\n\n"
            for atom in self.adsorbate_cluster:
                self.skzcam_input_str["adsorbate"] += create_atom_coord_string(
                    atom=atom
                )

            self.skzcam_input_str["slab"] += (
                f"{len(self.slab_cluster) + len(self.ecp_region)}\n\n"
            )
            for atom in self.slab_cluster:
                self.skzcam_input_str["slab"] += create_atom_coord_string(atom=atom)
            self.skzcam_input_str["slab"] += ecp_region_block

    def _generate_point_charge_block(self) -> str:
        """
        Create the point charge block for the MRCC input file. This requires the embedded_cluster Atoms object containing both atom_type and oxi_states arrays, as well as the indices of the quantum cluster and ECP region. Such arrays are created by the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.

        Returns
        -------
        str
            The point charge block for the MRCC input file.
        """

        # Get the oxi_states arrays from the embedded_cluster
        oxi_states = self.adsorbate_slab_embedded_cluster.get_array("oxi_states")

        # Get the number of point charges for this system. There is a point charge associated with each capped ECP as well.
        pc_region_indices = [
            atom.index
            for atom in self.adsorbate_slab_embedded_cluster
            if atom.index not in self.quantum_cluster_indices
        ]

        num_pc = len(pc_region_indices)
        pc_block = f"qmmm=Amber\npointcharges\n{num_pc}\n"

        # Add the ecp_region indices
        for i in pc_region_indices:
            position = self.adsorbate_slab_embedded_cluster[i].position
            pc_block += f"  {position[0]:-16.11f} {position[1]:-16.11f} {position[2]:-16.11f} {oxi_states[i]:-16.11f}\n"

        return pc_block


class ORCAInputGenerator:
    """
    A class to generate the SKZCAM input for the ORCA ASE calculator.
    """

    def __init__(
        self,
        adsorbate_slab_embedded_cluster: Atoms,
        quantum_cluster_indices: list[int],
        ecp_region_indices: list[int],
        element_info: dict[ElementStr, ElementInfo],
        include_cp: bool = True,
        multiplicities: MultiplicityDict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        adsorbate_slab_embedded_cluster
            The ASE Atoms object containing the atomic coordinates and atomic charges from the .pun file, as well as the atom type. This object is created by the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        quantum_cluster_indices
            A list containing the indices of the atoms in each quantum cluster. These indices are provided by the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        ecp_region_indices
            A list containing the indices of the atoms in each ECP region. These indices are provided by the [quacc.atoms.skzcam.CreateSkzcamClusters][] class.
        element_info
            A dictionary with elements as keys which gives the (1) number of core electrons as 'core', (2) basis set as 'basis', (3) effective core potential as 'ecp', (4) resolution-of-identity/density-fitting auxiliary basis set for DFT/HF calculations as 'ri_scf_basis' and (5) resolution-of-identity/density-fitting for correlated wave-function methods as 'ri_cwft_basis'.
        include_cp
            If True, the coords strings will include the counterpoise correction (i.e., ghost atoms) for the adsorbate and slab.
        multiplicities
            The multiplicity of the adsorbate-slab complex, adsorbate and slab respectively, with the keys 'adsorbate_slab', 'adsorbate', and 'slab'.

        Returns
        -------
        None
        """

        self.adsorbate_slab_embedded_cluster = adsorbate_slab_embedded_cluster
        self.quantum_cluster_indices = quantum_cluster_indices
        self.ecp_region_indices = ecp_region_indices
        self.element_info = element_info
        self.include_cp = include_cp
        self.multiplicities = (
            {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1}
            if multiplicities is None
            else multiplicities
        )

        # Check that none of the indices in quantum_cluster_indices are in ecp_region_indices
        if not np.all(
            [x not in self.ecp_region_indices for x in self.quantum_cluster_indices]
        ):
            raise ValueError(
                "An atom in the quantum cluster is also in the ECP region."
            )

        # Create the adsorbate-slab complex quantum cluster and ECP region cluster
        self.adsorbate_slab_cluster: Atoms = self.adsorbate_slab_embedded_cluster[
            self.quantum_cluster_indices
        ]
        self.ecp_region: Atoms = self.adsorbate_slab_embedded_cluster[
            self.ecp_region_indices
        ]

        # Get the indices of the adsorbates from the quantum cluster
        self.adsorbate_indices: list[int] = [
            i
            for i in range(len(self.adsorbate_slab_cluster))
            if self.adsorbate_slab_cluster.get_array("atom_type")[i] == "adsorbate"
        ]
        # Get the indices of the slab from the quantum cluster
        self.slab_indices: list[int] = [
            i
            for i in range(len(self.adsorbate_slab_cluster))
            if self.adsorbate_slab_cluster.get_array("atom_type")[i] != "adsorbate"
        ]

        # Create the adsorbate and slab quantum clusters
        self.adsorbate_cluster: Atoms = self.adsorbate_slab_cluster[
            self.adsorbate_indices
        ]
        self.slab_cluster: Atoms = self.adsorbate_slab_cluster[self.slab_indices]

        # Initialize the orcablocks input strings for the adsorbate-slab complex, adsorbate, and slab
        self.orcablocks: BlockInfo = {"adsorbate_slab": "", "adsorbate": "", "slab": ""}

    def generate_input(self) -> BlockInfo:
        """
        Creates the orcablocks input for the ORCA ASE calculator.

        Returns
        -------
        BlockInfo
            The ORCA input block (to be put in 'orcablocks' parameter) as a string for the adsorbate-slab complex, the adsorbate, and the slab in a dictionary with the keys 'adsorbate_slab', 'adsorbate', and 'slab' respectively.
        """

        # First generate the preamble block
        self._generate_preamble_block()

        # Create the blocks for the coordinates
        self._generate_coords_block()

        # Combine the blocks
        return self.orcablocks

    def create_point_charge_file(self) -> str:
        """
        Create a point charge file that can be read by ORCA. This requires the embedded_cluster Atoms object containing both atom_type and oxi_states arrays, as well as the indices of the quantum cluster and ECP region.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The text for the point charge file for the electrostatic embedding.
        """

        # Get the oxi_states arrays from the embedded_cluster
        oxi_states = self.adsorbate_slab_embedded_cluster.get_array("oxi_states")

        # Get the number of point charges for this system
        total_indices = self.quantum_cluster_indices + self.ecp_region_indices
        num_pc = len(self.adsorbate_slab_embedded_cluster) - len(total_indices)
        counter = 0
        pc_file = f"{num_pc}\n"
        for i in range(len(self.adsorbate_slab_embedded_cluster)):
            if i not in total_indices:
                counter += 1
                position = self.adsorbate_slab_embedded_cluster[i].position
                if counter != num_pc:
                    pc_file += f"{oxi_states[i]:-16.11f} {position[0]:-16.11f} {position[1]:-16.11f} {position[2]:-16.11f}\n"
                else:
                    pc_file += f"{oxi_states[i]:-16.11f} {position[0]:-16.11f} {position[1]:-16.11f} {position[2]:-16.11f}"
        return pc_file

    def _generate_coords_block(self) -> None:
        """
        Generates the coordinates block for the ORCA input file. This includes the coordinates of the quantum cluster, the ECP region, and the point charges. It will return three strings for the adsorbate-slab complex, adsorbate and slab.

        Returns
        -------
        None
        """

        # Get the charge of the adsorbate_slab cluster
        charge = int(sum(self.adsorbate_slab_cluster.get_array("oxi_states")))

        # Add the coords strings for the adsorbate-slab complex, adsorbate, and slab
        self.orcablocks["adsorbate_slab"] += f"""%coords
CTyp xyz
Mult {self.multiplicities['adsorbate_slab']}
Units angs
Charge {charge}
coords
"""
        self.orcablocks["adsorbate"] += f"""%coords
CTyp xyz
Mult {self.multiplicities['adsorbate']}
Units angs
Charge 0
coords
"""
        self.orcablocks["slab"] += f"""%coords
CTyp xyz
Mult {self.multiplicities['slab']}
Units angs
Charge {charge}
coords
"""

        for i, atom in enumerate(self.adsorbate_slab_cluster):
            # Create the coords section for the adsorbate-slab complex
            self.orcablocks["adsorbate_slab"] += create_atom_coord_string(atom=atom)

            # Create the coords section for the adsorbate and slab
            if i in self.adsorbate_indices:
                self.orcablocks["adsorbate"] += create_atom_coord_string(atom=atom)
                if self.include_cp:
                    self.orcablocks["slab"] += create_atom_coord_string(
                        atom=atom, is_ghost_atom=True
                    )
            elif i in self.slab_indices:
                self.orcablocks["slab"] += create_atom_coord_string(atom=atom)
                if self.include_cp:
                    self.orcablocks["adsorbate"] += create_atom_coord_string(
                        atom=atom, is_ghost_atom=True
                    )

        # Create the coords section for the ECP region
        ecp_region_coords_section = ""
        for i, atom in enumerate(self.ecp_region):
            ecp_region_coords_section += create_atom_coord_string(
                atom=atom,
                is_capped_ecp=True,
                pc_charge=self.ecp_region.get_array("oxi_states")[i],
            )

        # Add the ECP region coords section to the ads_slab_coords string
        self.orcablocks["adsorbate_slab"] += f"{ecp_region_coords_section}end\nend\n"
        self.orcablocks["slab"] += f"{ecp_region_coords_section}end\nend\n"
        self.orcablocks["adsorbate"] += "end\nend\n"

    def _generate_preamble_block(self) -> str:
        """
        From the quantum cluster Atoms object, generate the ORCA input preamble for the basis, method, pal, and scf blocks.

        Returns
        -------
        None
        """

        # Get the set of element symbols from the quantum cluster
        element_symbols = list(set(self.adsorbate_slab_cluster.get_chemical_symbols()))
        element_symbols.sort()

        # Check all element symbols are provided in element_info keys
        if not all(element in self.element_info for element in element_symbols):
            raise ValueError(
                "Not all element symbols are provided in the element_info dictionary."
            )

        # Initialize preamble_input
        preamble_input = ""

        # Add pointcharge file to read. It will be assumed that it is in the same folder as the input file
        preamble_input += '%pointcharges "orca.pc"\n'

        # Make the method block
        preamble_input += "%method\n"
        # Iterate over the core value and ECP for each element (if it has been given)
        for element in element_symbols:
            if "core" in self.element_info[element]:
                preamble_input += (
                    f"NewNCore {element} {self.element_info[element]['core']} end\n"
                )
        preamble_input += "end\n"

        # Make the basis block for basis, ri_scf_basis and ri_cwft_basis (and ecp if it has been provided)
        preamble_input += "%basis\n"
        for element in element_symbols:
            # basis
            element_basis = self.element_info[element]["basis"]
            preamble_input += f"""NewGTO {element} "{element_basis}" end\n"""
            # ri_scf_basis
            element_basis = self.element_info[element]["ri_scf_basis"]
            preamble_input += f'NewAuxJGTO {element} "{element_basis}" end\n'
            # ri_cwft_basis
            element_basis = self.element_info[element]["ri_cwft_basis"]
            preamble_input += f"""NewAuxCGTO {element} "{element_basis}" end\n"""
            # ecp
            if (
                "ecp" in self.element_info[element]
                and self.element_info[element]["ecp"] is not None
            ):
                element_ecp = self.element_info[element]["ecp"]
                preamble_input += f'NewECP {element} "{element_ecp}" end\n'

        preamble_input += "end\n"

        # Add preamble_input to the orcablocks for the adsorbate-slab complex, adsorbate, and slab
        self.orcablocks["adsorbate_slab"] += preamble_input
        self.orcablocks["adsorbate"] += (
            "\n".join(
                [line for line in preamble_input.splitlines() if "orca.pc" not in line]
            )
            + "\n"
        )
        self.orcablocks["slab"] += preamble_input


def create_atom_coord_string(
    atom: Atom,
    is_ghost_atom: bool = False,
    is_capped_ecp: bool = False,
    pc_charge: float | None = None,
) -> str:
    """
    Creates a string containing the Atom symbol and coordinates for both MRCC and ORCA, with additional information for atoms in the ECP region as well as ghost atoms.

    Parameters
    ----------
    atom
        The ASE Atom (not Atoms) object containing the atomic coordinates.
    is_ghost_atom
        If True, then the atom is a ghost atom.
    is_capping_ecp
        If True, then the atom is a capped_ECP.
    pc_charge
        The point charge value for the ECP region atom.

    Returns
    -------
    str
        The atom symbol and coordinates in the ORCA input file format.
    """

    # If ecp_info is not None and ghost_atom is True, raise an error
    if is_capped_ecp and is_ghost_atom:
        raise ValueError("Capped ECP cannot be a ghost atom.")

    # Check that pc_charge is a float if atom_ecp_info is not None
    if is_capped_ecp and pc_charge is None:
        raise ValueError("Point charge value must be given for atoms with ECP info.")

    if is_ghost_atom:
        atom_coord_str = f"{(atom.symbol + ':').ljust(3)} {' '*16} {atom.position[0]:-16.11f} {atom.position[1]:-16.11f} {atom.position[2]:-16.11f}\n"
    elif is_capped_ecp:
        atom_coord_str = f"{(atom.symbol + '>').ljust(3)} {pc_charge:-16.11f} {atom.position[0]:-16.11f} {atom.position[1]:-16.11f} {atom.position[2]:-16.11f}\ncappedECP\n"
    else:
        atom_coord_str = f"{atom.symbol.ljust(3)} {' '*16} {atom.position[0]:-16.11f} {atom.position[1]:-16.11f} {atom.position[2]:-16.11f}\n"

    return atom_coord_str
