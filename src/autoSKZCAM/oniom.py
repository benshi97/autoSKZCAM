from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from ase.calculators.orca import OrcaProfile
from quacc import get_settings
from quacc.calculators.mrcc.mrcc import MrccProfile

from autoSKZCAM.calculators import MRCC, ORCA
from autoSKZCAM.data import (
    capped_ecp_defaults,
    code_calculation_defaults,
    frozen_core_defaults,
)
from autoSKZCAM.io import MRCCInputGenerator, ORCAInputGenerator

if TYPE_CHECKING:
    from autoSKZCAM.embed import CreateEmbeddedCluster
    from autoSKZCAM.types import CalculatorInfo, ElementInfo, ElementStr, OniomLayerInfo


class Prepare:
    """
    From the set of ONIOM calculations, this class generates the calculations needed to be performed for each cluster from the series of clusters generated by the [quacc.atoms.skzcam.CreateEmbeddedCluster][] class.
    """

    def __init__(
        self,
        EmbeddedCluster: CreateEmbeddedCluster,
        OniomInfo: dict[str, OniomLayerInfo],
        capped_ecp: dict[Literal["mrcc", "orca"], str] | None = None,
        multiplicities: dict[str, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        EmbeddedCluster
            The CreateEmbeddedCluster object containing the atomic coordinates and atomic charges from the .pun file, as well as the atom type. This assumes the run_skzcam function has been run.
        OniomInfo
            A dictionary containing the information for each of the calculations for each of the ONIOM layers. An arbitrary number of ONIOM layers can be included.
        capped_ecp
            A dictionary containing the capped ECPs for each element in the quantum cluster. The dictionary should be in the form of {code: "capped ecp info"}. The code should be either 'mrcc' or 'orca'. Refer to the [autoSKZCAM.data.capped_ecp_defaults][] for the default values and how it should be formatted.
        multiplicities
            A dictionary containing the multiplicities for the adsorbate, slab and adsorbate-slab complex. The dictionary should be in the form of {"adsorbate_slab": int, "adsorbate": int, "slab": int}.

        Returns
        -------
        None
        """

        if multiplicities is None:
            multiplicities = {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1}
        self.adsorbate_slab_embedded_cluster = (
            EmbeddedCluster.adsorbate_slab_embedded_cluster
        )
        self.quantum_cluster_indices_set = EmbeddedCluster.quantum_cluster_indices_set
        self.ecp_region_indices_set = EmbeddedCluster.ecp_region_indices_set
        self.OniomInfo = OniomInfo
        if capped_ecp is None:
            unformatted_capped_ecp = capped_ecp_defaults
        else:
            unformatted_capped_ecp = capped_ecp
        self.multiplicities = multiplicities
        self.EmbeddedCluster = EmbeddedCluster

        # Check that adso_slab_embedded_cluster, quantum_cluster_indices_set and ecp_region_indices_set are not None
        if (
            self.adsorbate_slab_embedded_cluster is None
            or self.quantum_cluster_indices_set is None
            or self.ecp_region_indices_set is None
        ):
            raise ValueError(
                "The adsorbate_slab_embedded_cluster, quantum_cluster_indices_set and ecp_region_indices_set must be provided."
            )

        # Check that the quantum_cluster_indices_set and ecp_region_indices_set are the same length
        if len(self.quantum_cluster_indices_set) != len(self.ecp_region_indices_set):
            raise ValueError(
                "The quantum_cluster_indices_set and ecp_region_indices_set must be the same length."
            )

        # Raise an error if the capped_ecp dictionary is not formatted correctly
        formatted_capped_ecp = {}
        for code, ecp in unformatted_capped_ecp.items():
            if code.lower() == "mrcc" or code.lower() == "orca":
                formatted_capped_ecp[code.lower()] = ecp
            else:
                raise ValueError(
                    "The keys in the capped_ecp dictionary must be either 'mrcc' or 'orca' in the corresponding code format."
                )
        self.capped_ecp = formatted_capped_ecp

        # Raise an error if multiplicities is not formatted correctly
        for structure in ["adsorbate_slab", "adsorbate", "slab"]:
            if structure not in self.multiplicities:
                raise ValueError(
                    "The multiplicities must be provided for all three structures: adsorbate_slab, adsorbate, and slab."
                )

        # Check that all of the necessary keywords are included in each oniom layer
        max_cluster = 0
        for layer_name, oniom_layer in self.OniomInfo.items():
            for level in ["ll", "hl"]:
                if oniom_layer[level] is not None:
                    oniom_layer_parameters = oniom_layer[level]
                    # Check that all the required parameters are provided
                    for parameter in [
                        "method",
                        "max_cluster_num",
                        "basis",
                        "frozen_core",
                        "code",
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

                    # Check that the code is either 'mrcc' or 'orca'
                    if oniom_layer_parameters["code"] not in ["mrcc", "orca"]:
                        raise ValueError(
                            "The code must be specified as either 'mrcc' or 'orca'."
                        )

                    # Ensure that basis is specified as either DZ, TZ, QZ, 5Z or CBS(X/Y)
                    if oniom_layer_parameters["basis"] not in [
                        "DZ",
                        "TZ",
                        "QZ",
                        "5Z",
                        "CBS(DZ//TZ)",
                        "CBS(TZ//QZ)",
                        "CBS(QZ//5Z)",
                    ]:
                        raise ValueError(
                            "The basis must be specified as either DZ, TZ, QZ, 5Z, CBS(DZ//TZ), CBS(TZ//QZ) or CBS(QZ//5Z)."
                        )

                    # Some checks of the element_info in the case where the layer involves a CBS calculation
                    # If element_info is provided, check that it is a dictionary
                    if (
                        "element_info" in oniom_layer_parameters
                        and oniom_layer_parameters["element_info"] is not None
                    ):
                        # Check that the keys in element_info are valid element symbols
                        if not all(
                            key in frozen_core_defaults["semicore"]
                            for key in oniom_layer_parameters["element_info"]
                        ):
                            raise ValueError(
                                "The keys in the element_info dictionary must be valid element symbols."
                            )
                        # If the basis set is a CBS basis set, ensure that the basis set is split into two
                        if is_valid_cbs_format(oniom_layer_parameters["basis"])[0]:
                            for element in oniom_layer_parameters["element_info"]:
                                for basis_type in [
                                    "basis",
                                    "ri_scf_basis",
                                    "ri_cwft_basis",
                                ]:
                                    if (
                                        basis_type
                                        in oniom_layer_parameters["element_info"][
                                            element
                                        ]
                                        and not is_valid_cbs_format(
                                            oniom_layer_parameters["element_info"][
                                                element
                                            ][basis_type]
                                        )[0]
                                    ):
                                        raise ValueError(
                                            f"The {basis_type} parameter must be provided in the element_info dictionary as format CBS(X//Y), where X and Y are the two basis sets."
                                        )

                    # If code_inputs is provided and the code is orca, check that the orcasimpleinput and orcablocks are provided
                    if (
                        oniom_layer_parameters["code"] == "orca"
                        and "code_inputs" in oniom_layer_parameters
                        and oniom_layer_parameters["code_inputs"] is not None
                    ):
                        for key in oniom_layer_parameters["code_inputs"]:
                            if key not in ["orcasimpleinput", "orcablocks"]:
                                raise ValueError(
                                    "If the code is orca, the code_inputs dictionary can only contain the orcasimpleinput and orcablocks keys."
                                )

            if "bulk" in layer_name.lower():
                if (
                    OniomInfo[layer_name]["hl"] is None
                    or OniomInfo[layer_name]["ll"] is not None
                ):
                    raise ValueError(
                        f"For the {layer_name} layer, only high-level portion should be supplied."
                    )
            elif "fse" in layer_name.lower():
                # Ensure both hl and ll are provided
                if (
                    OniomInfo[layer_name]["hl"] is None
                    or OniomInfo[layer_name]["ll"] is None
                ):
                    raise ValueError(
                        f"For the {layer_name} layer, both high-level and low-level portions should be supplied."
                    )
                # Ensure that the only parameter which is different is the max_cluster_num
                if not all(
                    [
                        OniomInfo[layer_name]["hl"][key]
                        == OniomInfo[layer_name]["ll"][key]
                        for key in OniomInfo[layer_name]["hl"]
                        if key != "max_cluster_num"
                    ]
                    + [
                        OniomInfo[layer_name]["hl"]["max_cluster_num"]
                        > OniomInfo[layer_name]["ll"]["max_cluster_num"]
                    ]
                ):
                    raise ValueError(
                        f"The only parameter which should be different between the high-level and low-level calculations is the max_cluster_num, which should be the high-level for the {layer_name} layer."
                    )
            elif "delta" in layer_name.lower():
                # Ensure both hl and ll are provided
                if (
                    OniomInfo[layer_name]["hl"] is None
                    or OniomInfo[layer_name]["ll"] is None
                ):
                    raise ValueError(
                        f"Both high-level and low-level portions should be supplied for the {layer_name} layer."
                    )
                # Ensure that max_cluster_num is the same for both hl and ll
                if (
                    OniomInfo[layer_name]["hl"]["max_cluster_num"]
                    != OniomInfo[layer_name]["ll"]["max_cluster_num"]
                ):
                    raise ValueError(
                        f"The {layer_name} layer should have max_cluster_num that is same for both the high-level and low-level calculations."
                    )
            else:
                raise ValueError(
                    f"The {layer_name} layer should contain the keywords: 'bulk', 'fse' or 'delta'."
                )
        self.max_cluster = max_cluster

    def initialize_calculator(
        self,
        oniom_layer_parameters: OniomLayerInfo,
        quantum_cluster_indices: list[int],
        ecp_region_indices: list[int],
        element_info: dict[ElementStr, ElementInfo],
    ) -> CalculatorInfo:
        """
        Initialize the ASE calculator for the quantum cluster with the necessary inputs.

        Parameters
        ----------
        code
            The code to use for the quantum cluster. This could be either mrcc or orca.
        quantum_cluster_indices
            The indices of the atoms in the quantum cluster.
        ecp_region_indices
            The indices of the atoms in the ECP region.
        element_info
            A dictionary with elements as keys which gives the (1) number of core electrons as 'core', (2) basis set as 'basis', (3) effective core potential as 'ecp', (4) resolution-of-identity/density-fitting auxiliary basis set for DFT/HF calculations as 'ri_scf_basis' and (5) resolution-of-identity/density-fitting for correlated wave-function methods as 'ri_cwft_basis'.

        Returns
        -------
        CalculatorInfo
            A dictionary containing the ASE Atoms object for the 'adsorbate', 'slab' and 'adsorbate_slab' structures, and the ASE calculator for each structure.
        """
        code = oniom_layer_parameters["code"]
        method = oniom_layer_parameters["method"]
        calculators = {
            structure: deepcopy(self.adsorbate_slab_embedded_cluster)
            for structure in ["adsorbate", "slab", "adsorbate_slab"]
        }

        # Depending on the code, set the calculator and inputs
        if code == "mrcc":
            # Use MRCCInputGenerator to generate the necessary blocks for the SKZCAM protocol for the MRCC ASE calculator
            inputgenerator = MRCCInputGenerator(
                adsorbate_slab_embedded_cluster=self.adsorbate_slab_embedded_cluster,
                quantum_cluster_indices=quantum_cluster_indices,
                ecp_region_indices=ecp_region_indices,
                element_info=element_info,
                include_cp=True,
                multiplicities=self.multiplicities,
            )

            mrcc_skzcam_inputs = inputgenerator.generate_input()
            genbas_file = inputgenerator.create_genbas_file().replace(
                "INSERT_cappedECP", self.capped_ecp["mrcc"]
            )

            if method.upper() == "LNO-CCSD(T)":
                mrcc_default_method_inputs = code_calculation_defaults[code][
                    "LNO-CCSD(T)"
                ]
            elif method.upper() in ["MP2", "RI-MP2"]:
                mrcc_default_method_inputs = code_calculation_defaults[code]["MP2"]
            elif method.upper() == "LMP2":
                mrcc_default_method_inputs = code_calculation_defaults[code]["LMP2"]
            elif method.upper() == "CCSD(T)":
                mrcc_default_method_inputs = code_calculation_defaults[code]["CCSD(T)"]
            else:
                mrcc_default_method_inputs = code_calculation_defaults[code]["Other"]

            # Add default values to the mrcc_calc_inputs dictionary
            if (
                "code_inputs" in oniom_layer_parameters
                and oniom_layer_parameters["code_inputs"] is not None
            ):
                mrcc_calc_inputs = {
                    **mrcc_default_method_inputs,
                    **oniom_layer_parameters["code_inputs"],
                }
            else:
                mrcc_calc_inputs = {**mrcc_default_method_inputs}

            # Combine with the mrcc_block inputs
            inputs = {
                structure: {**mrcc_calc_inputs, **mrcc_skzcam_inputs[structure]}
                for structure in mrcc_skzcam_inputs
            }

            # Add "genbas" to the inputs dictionary
            inputs["adsorbate_slab"]["genbas"] = genbas_file
            inputs["slab"]["genbas"] = genbas_file
            inputs["adsorbate"]["genbas"] = genbas_file

            for structure, calculator in calculators.items():
                calculator.calc = MRCC(
                    profile=MrccProfile(command=get_settings().MRCC_CMD),
                    **inputs[structure],
                )

        elif code == "orca":
            # Use ORCAInputGenerator to generate the necessary orca_blocks for the ORCA ASE calculator
            inputgenerator = ORCAInputGenerator(
                adsorbate_slab_embedded_cluster=self.adsorbate_slab_embedded_cluster,
                quantum_cluster_indices=quantum_cluster_indices,
                ecp_region_indices=ecp_region_indices,
                element_info=element_info,
                include_cp=True,
                multiplicities=self.multiplicities,
            )

            orca_skzcam_inputs = inputgenerator.generate_input()
            pc_file = inputgenerator.create_point_charge_file()

            if method.upper() == "DLPNO-CCSD(T)":
                orcasimpleinput = code_calculation_defaults[code]["orcasimpleinput"][
                    "DLPNO-CCSD(T)"
                ]
            elif method.upper() == "DLPNO-MP2":
                orcasimpleinput = code_calculation_defaults[code]["orcasimpleinput"][
                    "DLPNO-MP2"
                ]
            elif method.upper() in ["MP2", "RI-MP2"]:
                orcasimpleinput = code_calculation_defaults[code]["orcasimpleinput"][
                    "MP2"
                ]
            elif method.upper() == "CCSD(T)":
                orcasimpleinput = code_calculation_defaults[code]["orcasimpleinput"][
                    "CCSD(T)"
                ]
            else:
                orcasimpleinput = code_calculation_defaults[code]["orcasimpleinput"][
                    "Other"
                ]

            orcablocks = code_calculation_defaults[code]["orcablocks"]

            if (
                "code_inputs" in oniom_layer_parameters
                and oniom_layer_parameters["code_inputs"] is not None
            ):
                if "orcasimpleinput" in oniom_layer_parameters["code_inputs"]:
                    orcasimpleinput = oniom_layer_parameters["code_inputs"][
                        "orcasimpleinput"
                    ]
                if "orcablocks" in oniom_layer_parameters["code_inputs"]:
                    orcablocks = oniom_layer_parameters["code_inputs"]["orcablocks"]

            # Add simpleinput and blocks to the orca_inputs dictionary
            inputs = {
                structure: {
                    "orcasimpleinput": orcasimpleinput,
                    "orcablocks": f"{orcablocks}\n{orca_skzcam_inputs[structure].replace('cappedECP', self.capped_ecp['orca'])}",
                }
                for structure in orca_skzcam_inputs
            }

            # Add "pointcharges" to the inputs dictionary
            inputs["adsorbate_slab"]["pointcharges"] = pc_file
            inputs["slab"]["pointcharges"] = pc_file
            inputs["adsorbate"]["pointcharges"] = None

            calculators = {
                structure: deepcopy(self.adsorbate_slab_embedded_cluster)
                for structure in inputs
            }
            for structure, calculator in calculators.items():
                calculator.calc = ORCA(
                    profile=OrcaProfile(command=get_settings().ORCA_CMD),
                    **inputs[structure],
                )

        return calculators

    def create_cluster_calcs(self) -> None:
        """
        Create the set of calculations needed to be performed for each cluster from the series of clusters generated by the [quacc.atoms.skzcam.CreateEmbeddedCluster][] class.

        Parameters
        ----------
        EmbeddedCluster
            The CreateEmbeddedCluster object containing the atomic coordinates and atomic charges from the .pun file, as well as the atom type. This assumes the run_skzcam function has been run.

        Returns
        -------
        None
        """
        # Set up the dictionary to store the information for each cluster
        skzcam_cluster_calculators: dict[int, dict[str, CalculatorInfo]] = {
            cluster_num: {} for cluster_num in range(1, self.max_cluster + 1)
        }
        for oniom_layer in self.OniomInfo.values():
            for level in ["ll", "hl"]:
                if oniom_layer[level] is not None:
                    oniom_layer_parameters = oniom_layer[level]
                    frozen_core = oniom_layer_parameters["frozen_core"]
                    method = oniom_layer_parameters["method"].replace(" ", "_")
                    code = oniom_layer_parameters["code"].lower()

                    # Continue if the code is MRCC and the method is (L)MP2 for the low-level and (LNO-)CCSD(T) for the high-level
                    if (
                        code == "mrcc"
                        and level == "ll"
                        and "MP2" in oniom_layer["ll"]["method"].upper()
                        and "CCSD(T)" in oniom_layer["hl"]["method"].upper()
                    ):
                        continue

                    # Continue if the code is ORCA and the method is MP2 for the low-level and CCSD(T) for the high-level
                    if (
                        code == "orca"
                        and level == "ll"
                        and oniom_layer["ll"]["method"].upper() == "MP2"
                        and oniom_layer["hl"]["method"].upper() == "CCSD(T)"
                    ):
                        continue

                    (is_cbs, basis_1, basis_2) = is_valid_cbs_format(
                        oniom_layer_parameters["basis"]
                    )
                    if is_cbs:
                        basis_sets = [basis_1, basis_2]
                    else:
                        basis_sets = [oniom_layer_parameters["basis"]]
                    for basis_idx, basis_set in enumerate(basis_sets):
                        default_element_info = self.create_element_info(
                            frozen_core=frozen_core, basis=basis_set, code=code
                        )
                        if (
                            "element_info" in oniom_layer_parameters
                            and oniom_layer_parameters["element_info"] is not None
                        ):
                            custom_element_info = {}
                            for key, value in oniom_layer_parameters[
                                "element_info"
                            ].items():
                                custom_element_info[key] = {}
                                for subkey, subvalue in value.items():
                                    if "basis" in subkey:
                                        is_element_basis_cbs = is_valid_cbs_format(
                                            subvalue
                                        )
                                        if is_cbs and is_element_basis_cbs[0]:
                                            custom_element_info[key][subkey] = (
                                                is_element_basis_cbs[basis_idx + 1]
                                            )
                                        else:
                                            custom_element_info[key][subkey] = subvalue
                                    else:
                                        custom_element_info[key][subkey] = subvalue

                            element_info = {
                                **default_element_info,
                                **custom_element_info,
                            }
                        else:
                            element_info = default_element_info

                        calculation_label = f"{code} {method} {frozen_core} {basis_set}"
                        for (
                            cluster_num,
                            cluster_calculators,
                        ) in skzcam_cluster_calculators.items():
                            if (
                                calculation_label not in cluster_calculators
                                and cluster_num
                                < oniom_layer_parameters["max_cluster_num"] + 1
                            ):
                                skzcam_cluster_calculators[cluster_num][
                                    calculation_label
                                ] = self.initialize_calculator(
                                    oniom_layer_parameters=oniom_layer_parameters,
                                    quantum_cluster_indices=self.quantum_cluster_indices_set[
                                        cluster_num - 1
                                    ],
                                    ecp_region_indices=self.ecp_region_indices_set[
                                        cluster_num - 1
                                    ],
                                    element_info=element_info,
                                )

        self.EmbeddedCluster.skzcam_calcs = skzcam_cluster_calculators

    def create_element_info(
        self,
        frozen_core: Literal["valence", "semicore"],
        basis: Literal["DZ", "TZ", "QZ", "5Z"],
        code: Literal["mrcc", "orca"],
        ecp: dict[ElementStr, str] | None = None,
    ) -> dict[ElementStr, ElementInfo]:
        """
        Creates the element info dictionary for the SKZCAM input across each oniom layer.

        Parameters
        ----------
        frozencore
            The frozen core to use for the quantum cluster. This could be specified as a string being either 'semicore' or 'valence'.
        basis
            The basis set to use for the quantum cluster. This could be either double-zeta, triple-zeta, quadruple-zeta, quintuple-zeta, denoted as 'DZ', 'TZ',' QZ' and '5Z' respectively.
        code
            The code to use for the quantum cluster. This could be either mrcc or orca.
        ecp
            The effective core potential to use for each element within the quantum cluster.
        ri_scf_basis
            The resolution-of-identity/density-fitting auxiliary basis set for DFT/HF calculations.
        ri_cwft_basis
            The resolution-of-identity/density-fitting for correlated wave-function methods.

        Returns
        -------
        dict[ElementStr, ElementInfo]
            A dictionary with elements as keys which gives the (1) number of core electrons as 'core', (2) basis set as 'basis', (3) effective core potential as 'ecp', (4) resolution-of-identity/density-fitting auxiliary basis set for DFT/HF calculations as 'ri_scf_basis' and (5) resolution-of-identity/density-fitting for correlated wave-function methods as 'ri_cwft_basis'.
        """

        # Create an adsorbate_slab_quantum_cluster object for the first SKZCAM cluster
        adsorbate_slab_quantum_cluster = self.adsorbate_slab_embedded_cluster[
            self.quantum_cluster_indices_set[0]
        ]

        element_info_dict = {}
        # If use_presets is True, use some preset inputs based on basis set and frozen core
        for atom_idx, atom in enumerate(adsorbate_slab_quantum_cluster):
            if atom.symbol in element_info_dict:
                continue
            if adsorbate_slab_quantum_cluster.get_array("atom_type")[atom_idx] in [
                "adsorbate",
                "anion",
            ]:
                element_info_dict[atom.symbol] = {
                    "core": frozen_core_defaults["valence"][atom.symbol],
                    "basis": f"aug-cc-pV{basis}",
                    "ecp": None if ecp is None else ecp.get(atom.symbol, None),
                    "ri_scf_basis": "def2-QZVPP-RI-JK" if code == "mrcc" else "def2/J",
                    "ri_cwft_basis": f"aug-cc-pV{basis}-RI"
                    if code == "mrcc"
                    else f"aug-cc-pV{basis}/C",
                }
            elif (
                adsorbate_slab_quantum_cluster.get_array("atom_type")[atom_idx]
                == "cation"
                and frozen_core == "valence"
            ):
                element_info_dict[atom.symbol] = {
                    "core": frozen_core_defaults["valence"][atom.symbol],
                    "basis": f"cc-pV{basis}",
                    "ecp": None if ecp is None else ecp.get(atom.symbol, None),
                    "ri_scf_basis": "def2-QZVPP-RI-JK" if code == "mrcc" else "def2/J",
                    "ri_cwft_basis": f"cc-pV{basis}-RI"
                    if code == "mrcc"
                    else f"cc-pV{basis}/C",
                }
            elif (
                adsorbate_slab_quantum_cluster.get_array("atom_type")[atom_idx]
                == "cation"
                and frozen_core == "semicore"
            ):
                element_info_dict[atom.symbol] = {
                    "core": frozen_core_defaults["semicore"][atom.symbol],
                    "basis": f"cc-pwCV{basis}",
                    "ecp": None if ecp is None else ecp.get(atom.symbol, None),
                    "ri_scf_basis": "def2-QZVPP-RI-JK" if code == "mrcc" else "def2/J",
                    "ri_cwft_basis": f"cc-pwCV{basis}-RI"
                    if code == "mrcc"
                    else "AutoAux",
                }

        return element_info_dict


def is_valid_cbs_format(string) -> list[bool, str | None, str | None]:
    """
    Returns True if the string is in the format of a CBS extrapolation when specified in element_info.

    Parameters
    ----------
    string
        The string to be checked.

    Returns
    -------
    bool
        True if the string is in the format of a CBS extrapolation, False otherwise.
    """
    string = string.replace(" ", "")
    # Define the regex pattern with capturing groups for the strings in between
    pattern = r"^CBS\((.+?)//(.+?)\)$"
    # Match the string against the pattern
    match = re.match(pattern, string)
    if match:
        # Return the captured groups as a tuple
        return True, match.group(1), match.group(2)
    return False, None, None
