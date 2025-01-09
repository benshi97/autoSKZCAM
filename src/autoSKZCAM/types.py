from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    class SKZCAMOutput(TypedDict):
        adsorbate_slab_embedded_cluster: Atoms
        quantum_cluster_indices_set: list[list[int]]
        ecp_region_indices_set: list[list[int]]

    class ElementInfo(TypedDict):
        core: int
        basis: str
        ecp: str
        ri_scf_basis: str
        ri_cwft_basis: str

    class SKZCAMInfo(TypedDict):
        max_cluster_num: int
        element_frozencore: Literal["valence", "semicore"] | dict[ElementStr, int]
        element_basis: Literal["DZ", "TZ", "QZ", "5Z", "6Z"] | dict[ElementStr, str]
        code: Literal["mrcc", "orca"]
        nprocs: int | None
        max_memory: int | None
        element_ecp: dict[ElementStr, str] | None
        multiplicities: MultiplicityDict | None
        cation_cap_ecp: dict[ElementStr, str] | None
        orca_method_block: dict[str, str] | None
        orca_scf_block: dict[str, str] | None
        mrcc_calc_inputs: dict[str, str] | None

    class BlockInfo(TypedDict):
        adsorbate_slab: str
        adsorbate: str
        slab: str

    class MRCCInputDict(TypedDict):
        adsorbate_slab: dict[str, str]
        adsorbate: dict[str, str]
        slab: dict[str, str]

    class MultiplicityDict(TypedDict):
        adsorbate_slab: int
        slab: int
        adsorbate: int
