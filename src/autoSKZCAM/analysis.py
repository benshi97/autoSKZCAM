from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from quacc.calculators.mrcc.io import read_mrcc_outputs

from autoSKZCAM.calculators import read_orca_outputs
from autoSKZCAM.oniom import is_valid_cbs_format

if TYPE_CHECKING:
    from autoSKZCAM.embed import CreateEmbeddedCluster
    from autoSKZCAM.types import EnergyInfo, OniomLayerInfo


def compute_skzcam_int_ene(
    skzcam_calcs_analysis: dict[int, list[str]], OniomInfo: dict[str, OniomLayerInfo]
) -> dict[str, list[float]]:
    """
    Computes the interaction energy contributions for each ONIOM layer in the SKZCAM protocol.

    Parameters
    ----------
    skzcam_calcs_analysis
        The dictionary containing the SKZCAM calculations analysis.
    OniomInfo
        The dictionary containing the ONIOM layer information.

    Returns
    -------
    dict[str,list[float]]
        The interaction energy and error contributions for each ONIOM layer in the SKZCAM protocol.
    """

    skzcam_int_ene = {layer_name: [0, 0] for layer_name in OniomInfo}

    for layer_name, oniom_layer in OniomInfo.items():
        cluster_level_int_ene = {"ll": [], "hl": []}
        for level in ["ll", "hl"]:
            if oniom_layer[level] is not None:
                oniom_layer_parameters = oniom_layer[level]
                frozen_core = oniom_layer_parameters["frozen_core"]
                method = oniom_layer_parameters["method"].replace(" ", "_")
                code = oniom_layer_parameters["code"].lower()
                max_cluster_num = oniom_layer_parameters["max_cluster_num"]

                # Check the basis set family
                basis_set_family = "mixcc"
                if (
                    "element_info" in oniom_layer_parameters
                    and oniom_layer_parameters["element_info"] is not None
                ):
                    for element in oniom_layer_parameters["element_info"]:
                        if (
                            "def2"
                            in oniom_layer_parameters["element_info"][element]["basis"]
                        ):
                            basis_set_family = "def2"

                if "mp2" in method.lower():
                    method_type = "mp2"
                elif "ccsd(t)" in method.lower():
                    method_type = "ccsdt"
                elif "ccsd" in method.lower():
                    method_type = "ccsd"
                else:
                    raise ValueError("The method cannot be analysed automatically.")
                (is_cbs, basis_1, basis_2) = is_valid_cbs_format(
                    oniom_layer_parameters["basis"]
                )
                if is_cbs:
                    basis_sets = [basis_1, basis_2]
                else:
                    basis_sets = [oniom_layer_parameters["basis"]]

                for cluster_num in range(1, max_cluster_num + 1):
                    basis_set_scf_int_ene_list = []
                    basis_set_corr_int_ene_list = []
                    for _basis_idx, basis_set in enumerate(basis_sets):
                        # Use the
                        if (
                            (
                                code == "mrcc"
                                and level == "ll"
                                and "MP2" in oniom_layer["ll"]["method"].upper()
                                and "CCSD(T)" in oniom_layer["hl"]["method"].upper()
                            )
                            or (
                                code == "mrcc"
                                and level == "ll"
                                and "CCSD" in oniom_layer["ll"]["method"].upper()
                                and "CCSD(T)" in oniom_layer["hl"]["method"].upper()
                            )
                            or (
                                code == "orca"
                                and level == "ll"
                                and oniom_layer["ll"]["method"].upper() == "MP2"
                                and oniom_layer["hl"]["method"].upper() == "CCSD(T)"
                            )
                        ):
                            calculation_label = f"{code} {oniom_layer['hl']['method']} {frozen_core} {basis_set}"
                        else:
                            calculation_label = (
                                f"{code} {method} {frozen_core} {basis_set}"
                            )
                        if (
                            calculation_label in skzcam_calcs_analysis[cluster_num]
                            and calculation_label != "cluster_size"
                        ):
                            basis_set_scf_int_ene_list += [
                                skzcam_calcs_analysis[cluster_num][calculation_label][
                                    "int_ene"
                                ]["scf_energy"]
                            ]
                            basis_set_corr_int_ene_list += [
                                _get_method_int_ene(
                                    energy_dict=skzcam_calcs_analysis[cluster_num][
                                        calculation_label
                                    ]["int_ene"],
                                    method_type=method_type,
                                )
                            ]
                    if is_cbs:
                        cluster_level_int_ene[level] += [
                            get_cbs_extrapolation(
                                basis_set_scf_int_ene_list[0],
                                basis_set_corr_int_ene_list[0],
                                basis_set_scf_int_ene_list[1],
                                basis_set_corr_int_ene_list[1],
                                X_size=basis_sets[0],
                                Y_size=basis_sets[1],
                                family=basis_set_family,
                            )[-1]
                        ]
                    else:
                        cluster_level_int_ene[level] += [
                            basis_set_corr_int_ene_list[0]
                            + basis_set_scf_int_ene_list[0]
                        ]

        cluster_level_int_ene["ll"] = np.array(cluster_level_int_ene["ll"])
        cluster_level_int_ene["hl"] = np.array(cluster_level_int_ene["hl"])

        if "extrapolate" in layer_name.lower() and "bulk" in layer_name.lower():
            skzcam_int_ene[layer_name] = [
                extrapolate_to_bulk(
                    [
                        skzcam_calcs_analysis[cluster_num]["cluster_size"]
                        for cluster_num in range(
                            1, oniom_layer["hl"]["max_cluster_num"] + 1
                        )
                    ],
                    cluster_level_int_ene["hl"],
                ),
                0,
            ]
        elif "bulk" in layer_name.lower():
            skzcam_int_ene[layer_name] = [cluster_level_int_ene["hl"][-1], 0]
        elif "fse" in layer_name.lower():
            skzcam_int_ene[layer_name] = [
                0,
                abs(
                    extrapolate_to_bulk(
                        [
                            skzcam_calcs_analysis[cluster_num]["cluster_size"]
                            for cluster_num in range(
                                1, oniom_layer["hl"]["max_cluster_num"] + 1
                            )
                        ],
                        cluster_level_int_ene["hl"],
                    )
                    - extrapolate_to_bulk(
                        [
                            skzcam_calcs_analysis[cluster_num]["cluster_size"]
                            for cluster_num in range(
                                1, oniom_layer["ll"]["max_cluster_num"] + 1
                            )
                        ],
                        cluster_level_int_ene["ll"],
                    )
                ),
            ]
        elif "delta" in layer_name.lower():
            if max_cluster_num < 3:
                skzcam_int_ene[layer_name] = [
                    np.mean(cluster_level_int_ene["hl"] - cluster_level_int_ene["ll"]),
                    0,
                ]
            else:
                skzcam_int_ene[layer_name] = [
                    np.mean(cluster_level_int_ene["hl"] - cluster_level_int_ene["ll"]),
                    2
                    * np.std(cluster_level_int_ene["hl"] - cluster_level_int_ene["ll"]),
                ]

    # Compute the final SKZCAM interaction energy
    final_int_ene = np.sum(
        [skzcam_int_ene[layer_name][0] for layer_name in skzcam_int_ene]
    )
    final_int_ene_error = np.sqrt(
        np.sum([skzcam_int_ene[layer_name][1] ** 2 for layer_name in skzcam_int_ene])
    )

    skzcam_int_ene["Total"] = [final_int_ene, final_int_ene_error]

    return skzcam_int_ene


def _get_method_int_ene(
    energy_dict: EnergyInfo, method_type: Literal["mp2", "ccsd", "ccsd(t)", "scf"]
) -> float:
    """
    Get the interaction energy for a given method.

    Parameters
    ----------
    energy_dict
        The dictionary containing the energy information.
    method_type
        The method to extract the interaction energy from.

    Returns
    -------
    float
        The interaction energy for the given method.
    """

    type_to_energy_label = {
        "scf": "scf_energy",
        "mp2": "mp2_corr_energy",
        "ccsd": "ccsd_corr_energy",
        "ccsdt": "ccsdt_corr_energy",
    }

    if energy_dict[type_to_energy_label[method_type]] is None:
        raise ValueError("The energy is None.")
    return energy_dict[type_to_energy_label[method_type]]


def extrapolate_to_bulk(x_data: list[float], y_data: list[float]) -> float:
    """
    Function to perform a linear fit of 1/x_data vs y_data to extrapolate to the bulk limit.

    Parameters
    ----------
    x_data
        The x data points.
    y_data
        The y data points.

    Returns
    -------
    tuple[float,float,float]
        The bulk limit, which is the zero intercept of the linear fit.
    """

    x_transformed_data = np.array([1 / x for x in x_data])

    # Degree of the polynomial
    degree = 1  # You can change the degree based on your requirement

    # Perform the polynomial fit
    coefficients = np.polyfit(x_transformed_data, y_data, degree)

    # Print the coefficients of the polynomial
    return coefficients[1]


def get_cbs_extrapolation(
    hf_X: float,
    corr_X: float,
    hf_Y: float,
    corr_Y: float,
    X_size: Literal["DZ", "TZ", "QZ"] = "DZ",
    Y_size: Literal["TZ", "QZ", "5Z"] = "TZ",
    family: Literal["def2", "cc", "acc", "mixcc"] = "mixcc",
):
    """
    Function to perform basis set extrapolation of HF and correlation energies for both the cc-pVXZ and def2-XZVP basis sets

    Parameters
    ----------
    hf_X : float
        HF energy in X basis set
    corr_X : float
        Correlation energy in X basis set
    hf_Y : float
        HF energy in Y basis set where Y = X+1 cardinal zeta number
    corr_Y : float
        Correlation energy in Y basis set
    X_size : str
        Cardinal zeta number of X basis set
    Y_size : str
        Cardinal zeta number of Y basis set
    family : str
        Basis set family. Options are 'cc', 'def2', 'acc', and 'mixcc'. Where cc is for non-augmented correlation consistent basis sets, def2 is for def2 basis sets, acc is for augmented correlation consistent basis sets while mixcc is for mixed augmented + non-augmented correlation consistent basis sets

    Returns
    -------
    hf_cbs : float
        HF CBS energy
    corr_cbs : float
        Correlation CBS energy
    tot_cbs : float
        Total CBS energy
    """

    # Dictionary of alpha parameters followed by beta parameters in CBS extrapoation. Refer to: Neese, F.; Valeev, E. F. Revisiting the Atomic Natural Orbital Approach for Basis Sets: Robust Systematic Basis Sets for Explicitly Correlated and Conventional Correlated Ab Initio Methods. J. Chem. Theory Comput. 2011, 7 (1), 33-43. https://doi.org/10.1021/ct100396y.
    alpha_dict = {
        "def2_2_3": 10.39,
        "def2_3_4": 7.88,
        "cc_2_3": 4.42,
        "cc_3_4": 5.46,
        "cc_4_5": 5.46,
        "acc_2_3": 4.30,
        "acc_3_4": 5.79,
        "acc_4_5": 5.79,
        "mixcc_2_3": 4.36,
        "mixcc_3_4": 5.625,
        "mixcc_4_5": 5.625,
    }

    beta_dict = {
        "def2_2_3": 2.40,
        "def2_3_4": 2.97,
        "cc_2_3": 2.46,
        "cc_3_4": 3.05,
        "cc_4_5": 3.05,
        "acc_2_3": 2.51,
        "acc_3_4": 3.05,
        "acc_4_5": 3.05,
        "mixcc_2_3": 2.485,
        "mixcc_3_4": 3.05,
        "mixcc_4_5": 3.05,
    }

    size_to_num = {"DZ": 2, "TZ": 3, "QZ": 4, "5Z": 5}

    X = size_to_num[X_size]
    Y = size_to_num[Y_size]

    # Check if X and Y are consecutive cardinal zeta numbers
    if Y != X + 1:
        raise ValueError("The cardinal number of Y does not equal X+1")

    # Get the corresponding alpha and beta parameters depending on the basis set family
    alpha = alpha_dict[f"{family}_{X}_{Y}"]
    beta = beta_dict[f"{family}_{X}_{Y}"]

    # Perform CBS extrapolation for HF and correlation components
    hf_cbs = hf_X - np.exp(-alpha * np.sqrt(X)) * (hf_Y - hf_X) / (
        np.exp(-alpha * np.sqrt(Y)) - np.exp(-alpha * np.sqrt(X))
    )
    corr_cbs = (X ** (beta) * corr_X - Y ** (beta) * corr_Y) / (
        X ** (beta) - Y ** (beta)
    )

    return hf_cbs, corr_cbs, (hf_cbs + corr_cbs)


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
            raise ValueError(
                "The embedded_cluster_path or EmbeddedCluster object must be provided."
            )
        EmbeddedCluster = np.load(embedded_cluster_path, allow_pickle=True).item()
    elif EmbeddedCluster is None and embedded_cluster_path is not None:
        EmbeddedCluster = np.load(embedded_cluster_path, allow_pickle=True).item()

    # Check that EmbeddedCluster.skzcam_calcs is not None
    if EmbeddedCluster.skzcam_calcs is None:
        raise ValueError(
            "The skzcam_calcs attribute of the EmbeddedCluster object is None."
        )

    skzcam_calcs_analysis = {
        cluster_num: {calculation_label: {} for calculation_label in calculation_labels}
        for cluster_num, calculation_labels in EmbeddedCluster.skzcam_calcs.items()
    }

    for cluster_num, calculations_list in EmbeddedCluster.skzcam_calcs.items():
        cluster_size = len(
            EmbeddedCluster.quantum_cluster_indices_set[cluster_num - 1]
        ) - len(EmbeddedCluster.adsorbate)
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
                    f"{code}.out",
                )
                skzcam_calcs_analysis[cluster_num][calculation_label][structure] = (
                    parse_energy(filename=system_path, code=code)
                )
            energy_dict = {}
            for method in skzcam_calcs_analysis[cluster_num][calculation_label][
                "adsorbate"
            ]:
                if (
                    skzcam_calcs_analysis[cluster_num][calculation_label]["adsorbate"][
                        method
                    ]
                    is not None
                ):
                    energy_dict[method] = (
                        skzcam_calcs_analysis[cluster_num][calculation_label][
                            "adsorbate_slab"
                        ][method]
                        - skzcam_calcs_analysis[cluster_num][calculation_label]["slab"][
                            method
                        ]
                        - skzcam_calcs_analysis[cluster_num][calculation_label][
                            "adsorbate"
                        ][method]
                    )
                else:
                    energy_dict[method] = None
            skzcam_calcs_analysis[cluster_num][calculation_label]["int_ene"] = (
                energy_dict
            )
        skzcam_calcs_analysis[cluster_num]["cluster_size"] = cluster_size
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


def get_quasi_rrho(r_freq, i_freq, T):
    """
    Uses the quasi rigid rotor harmonic approximation to calculate the thermal change and zero-point energies from vibrational frequencies in cm-1 and a temperature in Kelvin.

    Parameters
    ----------
    r_freq : list
        List of real vibrational frequencies in cm-1.
    i_freq : list
        List of imaginary vibrational frequencies in cm-1.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    dU : float
        The total change in energy including thermal energy and zero-point energy in eV.
    eth : float
        The thermal energy in eV.
    zpe : float
        The zero-point energy in eV.
    kT : float
        The product of Boltzmann constant (kB) and temperature (kT) in eV.
    """

    k = 8.617330337217213e-05  # Boltzmann constant
    hundredcm1 = 100000 / 8065.54429
    combined_freq = r_freq + [0.0001] * len(
        i_freq
    )  # Combine real and imaginary frequencies
    kT = k * T * 1000  # Calculate kT in eV

    dU = 0  # Initialize total energy change
    zpe = 0.0  # Initialize zero-point energy correction
    eth = 0.0  # Initialize thermal energy contribution
    for i in combined_freq:
        omega = 1 / (
            1 + ((hundredcm1 / i) ** 4)
        )  # Calculate the vibrational frequenecy in meV
        dURRho = (
            i / (np.exp(i / kT) - 1.0) + 0.5 * i
        )  # Calculate the contribution to thermal energy from this frequency
        zpe += omega * 0.5 * i  # Calculate the contribution to zero-point energy
        eth += (
            omega * i / (np.exp(i / kT) - 1.0) + (1 - omega) * 0.5 * kT
        )  # Calculate the thermal energy contribution
        dU += (
            omega * dURRho + (1 - omega) * 0.5 * kT
        )  # Calculate the total energy change

    return dU, eth, zpe, kT  # Return the calculated values
