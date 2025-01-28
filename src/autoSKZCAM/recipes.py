from __future__ import annotations

import numpy as np

from autoSKZCAM.recipes_dft import dft_ensemble_analyse, dft_ensemble_flow
from autoSKZCAM.recipes_skzcam import (
    skzcam_analyse,
    skzcam_eint_flow,
    skzcam_initialise,
)


def get_final_autoSKZCAM_Hads(
    skzcam_eint_analysis: dict[str, list[float]],
    dft_ensemble_analysis: dict[str, list[float]],
) -> dict[str, list[float]]:
    """
    Gets the final Hads from the autoSKZCAM workflow after dft_ensemble and skzcam analysis.

    Parameters
    ----------
    skzcam_eint_analysis
        The dictionary of the SKZCAM Eint analysis.
    dft_ensemble_analysis
        The dictionary of the DFT ensemble analysis.

    Returns
    -------
    dict[str, list[float]]
        The final Hads dictionary including the Hads contributions from the DFT ensemble and the SKZCAM calculations.
    """

    final_Hads = skzcam_eint_analysis.copy()

    final = skzcam_eint_analysis["Overall Eint"].copy()
    for key, contribution in dft_ensemble_analysis.items():
        final_Hads[key] = contribution
        final[0] += contribution[0]
        final[1] = np.sqrt(final[1] ** 2 + contribution[1] ** 2)

    final_Hads["Final Hads"] = final
    return final_Hads


__all__ = [
    "dft_ensemble_analyse",
    "dft_ensemble_flow",
    "get_final_autoSKZCAM_Hads",
    "skzcam_analyse",
    "skzcam_eint_flow",
    "skzcam_initialise",
]
