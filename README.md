<div align="center">
  <img src=https://github.com/benshi97/autoSKZCAM/blob/main/docs/images/logo.png width="700"><br>
</div>

# `autoSKZCAM` – Accurate Ionic Surface Predictions

`autoSKZCAM` is a computational framework for performing accurate yet efficient predictions of ionic surfaces.

- `autoSKZCAM` is highly flexible, currently supporting two popular quantum chemistry codes: [MRCC](https://mrcc.hu/) and [ORCA](https://orcaforum.kofo.mpg.de/), combining an arbitrary number electrostatic and mechanical embedding [ONIOM](https://pubs.acs.org/doi/10.1021/cr5004419) layers.

- `autoSKZCAM` is powered by [QuAcc](https://github.com/Quantum-Accelerators/quacc) and provides pre-made surface chemistry workflows that can be efficiently dispatched (and restarted) anywhere: locally, HPC, the cloud, or any combination thereof.

## Installation

For local development of the code:

1. Clone the repository

```
git clone https://github.com/benshi97/autoSKZCAM.git
```

2. Then install the package in editable mode

```
pip install -e .
```

where this command is run in the root directory. All dependences (i.e., QuAcc) will be automatically installed. By using the `-e` flag, the package will be installed in editable mode, meaning that changes to the code will be reflected in the installed package. Installation should only take a few minutes.

Note: You also will need to have [py-ChemShell](https://chemshell.org/) installed to run the code. It can be downloaded for free and installation instructions can be found [here](https://chemshell.org/static_files/py-chemshell/manual/build/html/install.html)

## Instructions and Demo

Instructions for running autoSKZCAM can be found in [`example/autoskzcam.ipynb`](example/autoskzcam.ipynb). It features:
- pre-calculated wave-function and DFT (ensemble) data, found in `example/calc_dir` and `example/dft_calc_dir`, respectively for the CO on MgO(001) system. This allows for the CO on MgO(001) results to be reproduced (Note: there are minor differences to the  [arXiv:2412.17204](https://arxiv.org/abs/2412.17204) paper due to differing basis set procedures).
- a Jupyter Notebook demo with instructions to run on the pre-calculated data (which should only take a couple of minutes to analyse), together with the expected final outputs.
- detailed explanation of keyword arguments entering each of the functions for the `autoSKZCAM` recipes.

As the `example.ipynb` is restarting from completed calculations, it does not perform any quantum chemistry calculations. If the user would like to perform these calculations, please move or delete the example/calc_dir and example/dft_calc_dir folders and follow the guidance within the demo to initialise these calculations. `autoSKZCAM` makes heavy use of the QuAcc computational materials workflow library and its documentation can be found [here](https://quantum-accelerators.github.io/quacc/index.html). The key requirement is that the calculators for each of the codes VASP, MRCC and ORCA must be set-up:
```
# Setup ORCA
export QUACC_ORCA_CMD="/path/to/orca/orca"

# Setup MRCC
export QUACC_MRCC_CMD="/path/to/orca/mrcc/dmrcc"

# Setup VASP
export QUACC_VASP_PARALLEL_CMD="srun -N 2 --ntasks-per-node 24"
export QUACC_VASP_PP_PATH="/path/to/POTCARs"
```
To improve the ORCA and MRCC calculation efficiencies, it is important to scale up the requested RAM (default: 2GB) and number of processes (default: 1) to the size of your computing system. This can be controlled in the `OniomInfo` (see `example/autoskzcam.ipynb`) parameter under `code_inputs`. For example, you can change the MRCC memory requirement using `{'mem': '20000MB'}` or using `{'orcablocks': 'nprocs 8 end\nmaxcore 10000'}`. Similarly the (parallel) VASP DFT calculations can be made more efficient by changing how you setup VASP (see code block above) as well as changing e.g., `NCORE` in the `job_params` parameter described in `example/autoskzcam.ipynb`.

## Requirements

`autoSKZCAM` requires only a standard computer with e.g., 8 GB of RAM to be performed. The quantum chemistry calculations in e.g., MRCC or ORCA can require more RAM depending on the type of system studied and calculations become significantly accelerated with greater availability of RAM and computing resources.

This package is supported for both macOS and Linux. The package has been tested on the following systems:
+ Apple: macOS 15 Sequoia
+ Linux: Ubuntu 20.04 (via Windows Subsystem for Linux 2) and Ubuntu 22.04.5

`autoSKZCAM` mainly depends on the Python >= 3.9 scientific stack, with the following dependencies:
- `quacc` >= 0.11.13 (its dependencies are automatically installed and can be found [here](https://github.com/Quantum-Accelerators/quacc/blob/main/pyproject.toml))
- `py-ChemShell` >= 20.0.0 (its pre-requisites can be found [here](https://chemshell.org/static_files/py-chemshell/manual/build/html/install.html#prerequisites))

To perform quantum chemistry calculations, [MRCC](https://mrcc.hu/) (>= 2023) and/or [ORCA](https://orcaforum.kofo.mpg.de/) (>= 5) must also be installed, both of which are free for academics.

Calculations with the DFT ensemble currently supports [VASP](https://vasp.at/), which requires the purchase of a licence.

## Citation and Reproducing Data

If you use `autoSKZCAM` in your work, please cite it as follows:

- An accurate and efficient framework for predictive insights into ionic surface chemistry, [arXiv:2412.17204](https://arxiv.org/abs/2412.17204)

This repository features the underlying code to power and perform the calculations found in the above work, using the CO on MgO(001) system as an example.

In the companion repository found at [benshi97/Data_autoSKZCAM](https://github.com/benshi97/Data_autoSKZCAM), we have compiled the data and outputs for **all** of the calculations in [arXiv:2412.17204](https://arxiv.org/abs/2412.17204) with detailed explanation/codes for analysing and reproducing **all** the outcomes. This can be viewed online on [Colab](https://colab.research.google.com/github/benshi97/Data_autoSKZCAM/blob/master/analyse.ipynb).


## License

`autoSKZCAM` is released under a [BSD 3-Clause license](https://github.com/quantum-accelerators/quacc/blob/main/LICENSE.md).

