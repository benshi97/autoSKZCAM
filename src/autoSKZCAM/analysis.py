from __future__ import annotations

from pathlib import Path


import numpy as np
from ase.units import Hartree
from typing import TYPE_CHECKING, Literal
from quacc.calculators.mrcc.io import read_mrcc_outputs
from autoSKZCAM.calculators import read_orca_outputs
from autoSKZCAM.embed import CreateEmbeddedCluster
from autoSKZCAM.oniom import is_valid_cbs_format

if TYPE_CHECKING:
    from autoSKZCAM.types import OniomLayerInfo, EnergyInfo

# def compute_skzcam_int_ene(
#         skzcam_calcs_analysis: dict[int, list[str]],
#         OniomInfo: dict[str, OniomLayerInfo]
# ) -> dict[str,list[float]]:
#     """
#     Computes the interaction energy contributions for each ONIOM layer in the SKZCAM protocol.

#     Parameters
#     ----------
#     skzcam_calcs_analysis
#         The dictionary containing the SKZCAM calculations analysis.
#     OniomInfo
#         The dictionary containing the ONIOM layer information.

#     Returns
#     -------
#     dict[str,list[float]]
#         The interaction energy and error contributions for each ONIOM layer in the SKZCAM protocol.
#     """

#     skzcam_int_ene = {layer_name: [0,0] for layer_name in OniomInfo.keys()}

#     for layer_name, oniom_layer in OniomInfo.items():
#         cluster_level_int_ene = {
#             'll': [],
#             'hl': []
#         }
#         for level in ["ll", "hl"]:
#             if oniom_layer[level] is not None:
#                 oniom_layer_parameters = oniom_layer[level]
#                 frozen_core = oniom_layer_parameters["frozen_core"]
#                 method = oniom_layer_parameters["method"].replace(" ", "_")
#                 code = oniom_layer_parameters["code"].lower()
#                 max_cluster_num = oniom_layer_parameters["max_cluster_num"]

#                 if "mp2" in method.lower():
#                     method_type = "mp2"
#                 elif "ccsd(t)" in method.lower():
#                     method_type = "ccsd(t)"
#                 elif "ccsd" in method.lower():
#                     method_type = "ccsd"

#                 (is_cbs, basis_1, basis_2) = is_valid_cbs_format(
#                     oniom_layer_parameters["basis"]
#                 )
#                 if is_cbs:
#                     basis_sets = [basis_1, basis_2]
#                 else:
#                     basis_sets = [oniom_layer_parameters["basis"]]

#                 for cluster_num in range(1, max_cluster_num + 1):
#                     basis_set_scf_int_ene_list = []
#                     basis_set_corr_int_ene_list = []
#                     for basis_idx, basis_set in enumerate(basis_sets):
#                         # Use the 
#                         if (
#                             code == "mrcc"
#                             and level == "ll"
#                             and "MP2" in oniom_layer["ll"]["method"].upper()
#                             and "CCSD(T)" in oniom_layer["hl"]["method"].upper()
#                         ) or (
#                             code == "orca"
#                             and level == "ll"
#                             and oniom_layer["ll"]["method"].upper() == "MP2"
#                             and oniom_layer["hl"]["method"].upper() == "CCSD(T)"
#                         ):
#                             calculation_label = f"{code} {oniom_layer['hl']['method']} {frozen_core} {basis_set}"
#                         else:
#                             calculation_label = f"{code} {method} {frozen_core} {basis_set}"

#                         if calculation_label in skzcam_calcs_analysis.keys():
#                             basis_set_scf_int_ene_list += [skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"]["scf_energy"]]
#                             basis_set_corr_int_ene_list += [_get_method_int_ene(energy_dict = skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"], method_type = method_type)]
#                     if is_cbs:
#                         cluster_level_int_ene[level] += [get_cbs_limit(basis_set_scf_int_ene_list[0], basis_set_corr_int_ene_list[0], basis_set_scf_int_ene_list[1], basis_set_corr_int_ene_list[1], X=basis_sets[0], Y=basis_sets[1], family='mixcc')]
#                     else:
#                         cluster_level_int_ene[level] += [basis_set_corr_int_ene_list[0] + basis_set_scf_int_ene_list[0]]

#         if "extrapolated bulk limit"

#         if 'bulk' in layer_name:

#         elif 'extrapolate' in layer_name:

#         elif 


# def _get_method_int_ene(
#         energy_dict: EnergyInfo,
#         method_type: Literal["mp2","ccsd","ccsd(t)","scf"]
# ) -> float:
#     """
#     Get the interaction energy for a given method.
    
#     Parameters
#     ----------
#     energy_dict
#         The dictionary containing the energy information.
#     method_type
#         The method to extract the interaction energy from.
        
#     Returns
#     -------
#     float
#         The interaction energy for the given method.
#     """

#     if method_type == "scf":
#         return energy_dict["scf_energy"]
#     elif method_type == "mp2":
#         return energy_dict["mp2_corr_energy"]
#     elif method_type == "ccsd":
#         return energy_dict["ccsd_corr_energy"]
#     elif method_type == "ccsd(t)":
#         return energy_dict["ccsdt_corr_energy"]


# def extrapolate_int_ene():
#     pass




def analyze_calculations(
        calc_dir: Path | str,
        embedded_cluster_path: Path | str | None = None,
        EmbeddedCluster: CreateEmbeddedCluster | None = None,
) -> dict[int, list[str]]:
    """
    Analyze the calculations performed in the calc_dir folder

    Parameters
    ----------
    calc_dir
        The directory containing the calculations to analyze.
    embedded_cluster_path
        The path to load the EmbeddedCluster object if it is not provided.
    EmbeddedCluster
        The EmbeddedCluster object containing the cluster information.

    Returns
    -------
    None
    
    """

    # If EmbeddedCluster is None, check that embedded_cluster_path is not None
    if EmbeddedCluster is None and embedded_cluster_path is None:
        embedded_cluster_path = Path(calc_dir, "embedded_cluster.npy")
        # Check that the embedded_cluster_path exists
        if not embedded_cluster_path.exists():
            raise ValueError("The embedded_cluster_path or EmbeddedCluster object must be provided.")
        else:
            EmbeddedCluster = np.load(embedded_cluster_path, allow_pickle=True).item()
    elif EmbeddedCluster is None and embedded_cluster_path is not None:
        EmbeddedCluster = np.load(embedded_cluster_path, allow_pickle=True).item()

    # Check that EmbeddedCluster.skzcam_calcs is not None
    if EmbeddedCluster.skzcam_calcs is None:
        raise ValueError("The skzcam_calcs attribute of the EmbeddedCluster object is None.")
    
    skzcam_calcs_analysis = {cluster_num: {calculation_label: {} for calculation_label in calculation_labels.keys()} for cluster_num, calculation_labels in EmbeddedCluster.skzcam_calcs.items()}
    
    for cluster_num, calculations_list in EmbeddedCluster.skzcam_calcs.items():
        for calculation_label in calculations_list:
            code = calculation_label.split(" ")[0]
            method = calculation_label.split(" ")[1]
            frozen_core = calculation_label.split(" ")[2]
            basis_set = calculation_label.split(" ")[3]
            for structure in ["adsorbate_slab", "adsorbate", "slab"]:
                system_path = Path(
                    calc_dir,
                    str(cluster_num),
                    code,
                    f"{method}_{basis_set}_{frozen_core}",
                    structure,
                    f"{code}.out"
                )
                skzcam_calcs_analysis[cluster_num][calculation_label][structure] = parse_energy(filename=system_path, code=code)
            energy_dict = {}
            for method in skzcam_calcs_analysis[cluster_num][calculation_label]['adsorbate'].keys():
                if skzcam_calcs_analysis[cluster_num][calculation_label]['adsorbate'][method] is not None:
                    energy_dict[method] = skzcam_calcs_analysis[cluster_num][calculation_label]['adsorbate_slab'][method] - skzcam_calcs_analysis[cluster_num][calculation_label]['slab'][method] - skzcam_calcs_analysis[cluster_num][calculation_label]['adsorbate'][method]
                else:
                    energy_dict[method] = None
            skzcam_calcs_analysis[cluster_num][calculation_label]['int_ene'] = energy_dict
    return skzcam_calcs_analysis


def parse_energy(filename, code="mrcc"):
    """
    Function to parse the energy from a MRCC or ORCA output file.
    
    Parameters
    ----------
    filename : str
        The location of the output file to read from.
    code : str
        The code format. Options are 'mrcc' and 'orca'
        
    Returns
    -------
    float
        The energy in the original units.
    """

    if code == "mrcc":
        energy_dict = read_mrcc_outputs(filename)

    elif code == "orca":
        energy_dict = read_orca_outputs(filename)

    return energy_dict