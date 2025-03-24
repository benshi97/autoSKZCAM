# Installation


`autoSKZCAM` requires python >= 3.11. We recommend starting in a new python environment using [miniconda](https://docs.anaconda.com/miniconda/):

```
conda create --name autoskzcam python=3.11
conda activate autoskzcam
```

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





