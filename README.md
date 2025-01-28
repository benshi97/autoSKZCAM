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

2. Then install the package in editable mode.

```
pip install -e .
```

where this command is run in the root directory. All dependences (i.e., QuAcc) will be automatically installed. By using the `-e` flag, the package will be installed in editable mode, meaning that changes to the code will be reflected in the installed package.

Note: You also will need to have [py-ChemShell](https://chemshell.org/) installed to run the code. It can be downloaded for free and installation instructions are [here](https://chemshell.org/static_files/py-chemshell/manual/build/html/install.html)

## Documentation

Instructions for running autoSKZCAM can be found in [example/autoskzcam.ipynb](example/autoskzcam.ipynb). We will provide more detailed documentation following its official release.

## Citation

If you use `autoSKZCAM` in your work, please cite it as follows:

- An accurate and efficient framework for predictive insights into ionic surface chemistry, [arXiv:2412.17204](https://arxiv.org/abs/2412.17204)

## License

`autoSKZCAM` is released under a [BSD 3-Clause license](https://github.com/quantum-accelerators/quacc/blob/main/LICENSE.md).

