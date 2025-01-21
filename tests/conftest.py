from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

FILE_DIR = Path(__file__).parent
MRCC_DIR = Path(FILE_DIR, "mrcc_run")
ORCA_DIR = Path(FILE_DIR, "orca_run")


def mock_mrcc_execute(self, directory, *args, **kwargs):
    import gzip

    with (
        gzip.open(MRCC_DIR / "mrcc.out.gz", "rb") as f,
        open(Path(directory, "mrcc.out"), "wb") as out,
    ):
        out.write(f.read())


@pytest.fixture(autouse=True)
def patch_mrcc_execute(monkeypatch):
    from autoSKZCAM.calculators import SkzcamMrccTemplate

    monkeypatch.setattr(SkzcamMrccTemplate, "execute", mock_mrcc_execute)


def mock_orca_execute(self, directory, *args, **kwargs):
    import gzip

    with (
        gzip.open(ORCA_DIR / "orca.out.gz", "rb") as f,
        open(Path(directory, "orca.out"), "wb") as out,
    ):
        out.write(f.read())


@pytest.fixture(autouse=True)
def patch_orca_execute(monkeypatch):
    from autoSKZCAM.calculators import SkzcamOrcaTemplate

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
    from autoSKZCAM.embed import CreateEmbeddedCluster

    monkeypatch.setattr(CreateEmbeddedCluster, "run_chemshell", mock_run_chemshell)
