from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ase.calculators.vasp.create_input import count_symbols
from ase.io import read
from quacc.schemas.ase import Summarize

from autoSKZCAM.calculators import SkzcamMrccTemplate, SkzcamOrcaTemplate
from autoSKZCAM.embed import CreateEmbeddedCluster
import gzip

if TYPE_CHECKING:
    from ase import Atoms

FILE_DIR = Path(__file__).parent
MRCC_DIR = Path(FILE_DIR, "mrcc_run")
ORCA_DIR = Path(FILE_DIR, "orca_run")


def mock_mrcc_execute(self, directory, *args, **kwargs):

    with (
        gzip.open(MRCC_DIR / "mrcc.out.gz", "rb") as f,
        open(Path(directory, "mrcc.out"), "wb") as out,
    ):
        out.write(f.read())


@pytest.fixture(autouse=True)
def patch_mrcc_execute(monkeypatch):
    monkeypatch.setattr(SkzcamMrccTemplate, "execute", mock_mrcc_execute)


def mock_orca_execute(self, directory, *args, **kwargs):

    with (
        gzip.open(ORCA_DIR / "orca.out.gz", "rb") as f,
        open(Path(directory, "orca.out"), "wb") as out,
    ):
        out.write(f.read())


@pytest.fixture(autouse=True)
def patch_orca_execute(monkeypatch):
    monkeypatch.setattr(SkzcamOrcaTemplate, "execute", mock_orca_execute)


def mock_run_chemshell(*args, filepath=".", write_xyz_file=False, **kwargs):
    if write_xyz_file:
        with (
            gzip.open(
                Path(FILE_DIR, "skzcam_files", "REF_ChemShell_Cluster.xyz.gz"), "rb"
            ) as f_in,
            Path(filepath).with_suffix(".xyz").open(mode="wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        with (
            gzip.open(
                Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"), "rb"
            ) as f_in,
            Path(filepath).with_suffix(".pun").open(mode="wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
    else:
        with (
            gzip.open(
                Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"), "rb"
            ) as f_in,
            Path(filepath).with_suffix(".pun").open(mode="wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)


@pytest.fixture(autouse=True)
def patch_run_chemshell(monkeypatch):
    monkeypatch.setattr(CreateEmbeddedCluster, "run_chemshell", mock_run_chemshell)


def mock_vasp_run_and_summarize(atoms, additional_fields, *args, **kwargs):
    def sort_atoms(atoms: Atoms) -> Atoms:
        symbols, _ = count_symbols(atoms, exclude=())

        # Create sorting list
        srt = []

        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if atom.symbol == symbol:
                    srt.append(m)
        # Create the resorting list
        resrt = list(range(len(srt)))
        for n in range(len(resrt)):
            resrt[srt[n]] = n

        return atoms.copy()[srt]

    job_type = additional_fields["calc_results_dir"].parent.name
    xc_func = additional_fields["calc_results_dir"].name

    mock_results_dir = Path(
        FILE_DIR, "mocked_vasp_runs", "dft_calc_dir", job_type, xc_func
    )

    with open(Path(mock_results_dir, "OUTCAR"), encoding="ISO-8859-1") as file:
        final_unsorted_atoms = read(file)

    final_atoms = sort_atoms(final_unsorted_atoms)
    final_atoms.calc = final_unsorted_atoms.calc
    return Summarize(directory=mock_results_dir).run(final_atoms, atoms)


@pytest.fixture(autouse=True)
def patch_vasp_run(monkeypatch):
    monkeypatch.setattr(
        "autoSKZCAM.recipes_dft.run_and_summarize", mock_vasp_run_and_summarize
    )
