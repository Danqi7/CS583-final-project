# Toy Trajectory Net
> Toy Trajectory Net Project


## Currently TOC is broken, temp fix:
- [`ToyTrajectoryNet.ode`](https://dsm-72.github.io/ToyTrajectoryNet/ode)
- [`ToyTrajectoryNet.losses`](https://dsm-72.github.io/ToyTrajectoryNet/losses)
- [`ToyTrajectoryNet.utils`](https://dsm-72.github.io/ToyTrajectoryNet/utils)
- [`ToyTrajectoryNet.models`](https://dsm-72.github.io/ToyTrajectoryNet/models)
- [`ToyTrajectoryNet.plots`](https://dsm-72.github.io/ToyTrajectoryNet/plots)
- [`ToyTrajectoryNet.train`](https://dsm-72.github.io/ToyTrajectoryNet/train)
- [`ToyTrajectoryNet.constants`](https://dsm-72.github.io/ToyTrajectoryNet/constants)
- [`ToyTrajectoryNet.datasets`](https://dsm-72.github.io/ToyTrajectoryNet/datasets)
- [`ToyTrajectoryNet.exp`](https://dsm-72.github.io/ToyTrajectoryNet/exp)
- [`ToyTrajectoryNet.geo`](https://dsm-72.github.io/ToyTrajectoryNet/geo)
- [`ToyTrajectoryNet.eval`](https://dsm-72.github.io/ToyTrajectoryNet/eval)


## Setup

To get all the pagackes required, run the following command:

```bash
$ conda env create -f environment.yml
```

This will create a new conda environment `sklab-toy-tjnet`, which can be activated via:

```
conda activate sklab-toy-tjnet
```

### Add kernel to Jupyter Notebook

#### automatic conda kernels
For greater detail see the official docs for [`nb_conda_kernels`][nb_conda_kernels].
In short, install `nb_conda_kernels` in the environment from which you launch JupyterLab / Jupyter Notebooks from (e.g. `base`) via:

```bash
$ conda install -n <notebook_env> nb_conda_kernels
```

to add a new or exist conda environment to Jupyter simply install `ipykernel` into that conda environment e.g.

```bash
$ conda install -n <python_env> ipykernel
```


#### manual ipykernel
add to your Jupyter Notebook kernels via

```bash
$ python -m ipykernel install --user --name sklab-toy-tjnet
```

It can be removed via:

```bash
$ jupyter kernelspec uninstall sklab-toy-tjnet
```

#### list kernels found by Jupyter

kernels recognized by conda
```bash
$ python -m nb_conda_kernels list
```

check which kernels are discovered by Jupyter:
```bash
$ jupyter kernelspec list
```

[nb_conda_kernels]: https://github.com/Anaconda-Platform/nb_conda_kernels

## Install

### For developers and internal use:
```
cd path/to/this/repository
pip install -e ToyTrajectoryNet
```

### For production use:
`pip install ToyTrajectoryNet`

## How to use

This repository consists of our python library `ToyTrajectoryNet` as well as a directory of scripts for running and using it. 

### Scripts

To recreate our results with MMD loss and density regulariazation you can run the following command:

```bash
python scripts/run.py -d petals -c mmd -n petal-mmd
```

This will generate the directory `results/petals-mmd` and save everything there.

For a full list of parameters try running:

```bash
python scripts/run.py --help
```

### Python Package
One could simply import everything and use it piecemeal:

```
from ToyTrajectoryNet.ode import *
from ToyTrajectoryNet.losses import *
from ToyTrajectoryNet.utils import *
from ToyTrajectoryNet.models import *
from ToyTrajectoryNet.plots import *
from ToyTrajectoryNet.train import *
from ToyTrajectoryNet.constants import *
from ToyTrajectoryNet.datasets import *
from ToyTrajectoryNet.exp import *
from ToyTrajectoryNet.geo import *
from ToyTrajectoryNet.eval import *
```

### Tutorials
One can also consult or modify the tutorial notebooks for their uses:
- [EB Bodies tutorial][ebbodies]
- [Dyngen tutorial][dyngen]
- [Petals tutorial][petals]

[ebbodies]: https://github.com/dsm-72/ToyTrajectoryNet/blob/main/notebooks/EB-Bodies.ipynb
[dyngen]: https://github.com/dsm-72/ToyTrajectoryNet/blob/main-clean/notebooks/Dyngen_good.ipynb
[petals]: https://github.com/dsm-72/ToyTrajectoryNet/blob/main-clean/notebooks/petal_good.ipynb