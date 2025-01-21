from __future__ import annotations

from ase.data import atomic_numbers

element_from_atomic_num_dict = {v: k for k, v in atomic_numbers.items()}

frozen_core_defaults = {
    "semicore": {
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
    },
    "valence": {
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
    },
}

capped_ecp_defaults = {
    "orca": """NewECP
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
    "mrcc": """
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
*""",
}

code_calculation_defaults = {
    "mrcc": {
        "LNO-CCSD(T)": {
            "calc": "LNO-CCSD(T)",
            "scftype": "rhf",
            "verbosity": 3,
            "mem": "80000MB",
            "symm": "off",
            "unit": "angs",
            "scfiguess": "small",
            "scfmaxit": 1000,
            "scfalg": "locfit1",
            "lcorthr": "tight",
            "bpedo": 0.99999,
            "ccmaxit": 400,
            "usedisk": 0,
            "ccsdalg": "dfdirect",
            "ccsdthreads": 4,
            "ccsdmkl": "thr",
            "ptthreads": 4,
        },
        "CCSD(T)": {
            "calc": "DF-CCSD(T)",
            "scftype": "rhf",
            "verbosity": 3,
            "mem": "80000MB",
            "symm": "off",
            "unit": "angs",
            "scfiguess": "small",
            "scfmaxit": 1000,
            "ccmaxit": 400,
            "ccsdalg": "dfdirect",
            "ccsdthreads": 4,
            "ccsdmkl": "thr",
            "ptthreads": 4,
        },
        "MP2": {
            "calc": "DF-MP2",
            "scftype": "rhf",
            "verbosity": 3,
            "mem": "80000MB",
            "symm": "off",
            "unit": "angs",
            "scfiguess": "small",
            "scfmaxit": 1000,
            "scfalg": "locfit1",
        },
        "LMP2": {
            "calc": "LMP2",
            "scftype": "rhf",
            "verbosity": 3,
            "mem": "80000MB",
            "symm": "off",
            "unit": "angs",
            "lcorthr": "tight",
            "bpedo": 0.99999,
            "scfiguess": "small",
            "scfmaxit": 1000,
            "scfalg": "locfit1",
        },
        "Other": {
            "verbosity": 3,
            "mem": "80000MB",
            "symm": "off",
            "unit": "angs",
            "scfiguess": "small",
        },
    },
    "orca": {
        "orcasimpleinput": {
            "MP2": "TightSCF RI-MP2 TightPNO RIJCOSX DIIS",
            "DLPNO-CCSD(T)": "TightSCF DLPNO-CCSD(T) TightPNO RIJCOSX DIIS",
            "DLPNO-MP2": "TightSCF DLPNO-MP2 TightPNO RIJCOSX DIIS",
            "CCSD(T)": "TightSCF CCSD(T) RIJCOSX DIIS",
            "Other": "TightSCF RIJCOSX DIIS",
        },
        "orcablocks": """
%pal nprocs 8 end
%maxcore 25000
%method
Method hf
RI on
RunTyp Energy
end
%scf
HFTyp rhf
SCFMode Direct
sthresh 1e-6
AutoTRAHIter 60
MaxIter 1000
end
""",
    },
}
