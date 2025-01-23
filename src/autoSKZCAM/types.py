from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.atoms import Atoms
    from typing_extensions import TypedDict

    # ----------- Atoms handling type hints -----------

    ElementStr = Literal[
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]

    class SkzcamOutput(TypedDict):
        adsorbate_slab_embedded_cluster: Atoms
        quantum_cluster_indices_set: list[list[int]]
        ecp_region_indices_set: list[list[int]]

    class ElementInfo(TypedDict):
        core: int | None
        basis: str | None
        ecp: str | None
        ri_scf_basis: str | None
        ri_cwft_basis: str | None

    class OniomLayerInfo(TypedDict):
        max_cluster_num: int
        method: str
        frozen_core: Literal["valence", "semicore"]
        basis: str
        code: Literal["mrcc", "orca"]
        element_info: dict[ElementStr, ElementInfo] | None
        code_inputs: dict[str, str] | None

    class EmbeddingInfo(TypedDict):
        adsorbate_indices: list[int]
        slab_center_indices: list[int]
        atom_oxi_states: dict[ElementStr, int]
        adsorbate_slab_file: str | Path

    class BlockInfo(TypedDict):
        adsorbate_slab: str
        adsorbate: str
        slab: str

    class MrccInputDict(TypedDict):
        adsorbate_slab: dict[str, str]
        adsorbate: dict[str, str]
        slab: dict[str, str]

    class MultiplicityDict(TypedDict):
        adsorbate_slab: int
        slab: int
        adsorbate: int

    class CalculatorInfo(TypedDict):
        adsorbate_slab: Atoms
        adsorbate: Atoms
        slab: Atoms

    class EnergyInfo(TypedDict):
        energy: float | None
        scf_energy: float | None
        mp2_corr_energy: float | None
        ccsd_corr_energy: float | None
        ccsdt_corr_energy: float | None

    class SkzcamAnalysisInfo(TypedDict):
        adsorbate_slab: EnergyInfo
        adsorbate: EnergyInfo
        slab: EnergyInfo
        int_ene: EnergyInfo
