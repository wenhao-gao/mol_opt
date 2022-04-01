# ChemBO

ChemBO is library for joint molecular optimization and synthesis. It is based on Dragonfly - a framework for scalable Bayesian optimization.

## Structure of the repo

* `experiments` package contains experiment scripts. In particular, `run_chemist.py` script illustrates usage of the classes.
* `chemist_opt` package isolates the Chemist class which performs joint optimization and synthesis. Contains harnesses for calling molecular functions (`MolFunctionCaller`) and handling optimization over molecular domains (`MolDomain`). Calls for `mols` and `explore`.
* `explorer` implements the exploration of molecular domain. Currently, a `RandomExplorer` is implemented, which explores reactions randoml, starting from a given pool. Calls for `synth`.
* `mols` contains the `Molecule` class, the `Reaction` class, a few examples of objective function definitions, as well as implementations of molecular versions of all components needed for BO to work: `MolCPGP` and `MolCPGPFitter` class and molecular kernels.
* `synth` is responsible for performing forward synthesis.
* `rdkit_contrib` is an extension to rdkit that provides computation of a few molecular scores (for older versions of `rdkit`).
* `baselines` contains wrappers for models we compare against.

## Current work

In the coming few weeks, we will try to clean up, refactor and further comment the code.

## Getting started

It's recommended to use python3.

**Python packages** 

First, set up environment for RDKit and Dragonfly:

```bash
conda create -c rdkit -n chemist-env rdkit python=3.6
# optionally: export PATH="/opt/miniconda3/bin:$PATH"
conda activate chemist-env  # or source activate chemist-env with older conda
```

Install basic requirements with pip:

```bash
pip install -r requirements.txt
```

**Kernel-related packages**

Certain functionality (some of the graph-based kernels) require the `graphkernels` package, which can be installed additionally. First, you need to install `eigen3`, `pkg-config`: [see instructions here](https://github.com/BorgwardtLab/GraphKernels):

```bash
sudo apt-get install libeigen3-dev; sudo apt-get install pkg-config  # on Linux
brew install eigen; brew install pkg-config  # on MacOS
pip install graphkernels
```

If the above fails on MacOS (see [stackoverflow](https://stackoverflow.com/questions/16229297/why-is-the-c-standard-library-not-working)), the simplest solution is

```bash
MACOSX_DEPLOYMENT_TARGET=10.9 pip install graphkernels
```

To use distance-based kernels, you need Cython and OT distance computers:

```bash
pip install Cython
pip install cython POT  # prepended with MACOSX_DEPLOYMENT_TARGET=10.9 if needed
```

**Synthesis Path Plotting Functionality**
For plotting the synthesis path for an optimal molecule, install `graphviz` via:

```bash
pip install graphviz
```

However, the above only works on Linux as Homebrew removed the `--with-pango` option (see [this](https://github.com/parrt/dtreeviz/issues/33))

### Environment

Set PYTHONPATH for imports:

```bash
source setup.sh 
```

### Getting data

ChEMBL data as txt can be found [in kevinid's repo](https://github.com/kevinid/molecule_generator/releases/), [official downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads). ZINC database can be downloaded from [the official site](http://zinc.docking.org/browse/subsets/). Run the following to automatically download the datasets and put them into the right directory:

```bash
bash download_data.sh
```

## Running tests

TODO

## Running experiments

See `experiments/run_chemist.py` for the Chemist usage example.

## Citation

If you found this code helpful, please consider citing [this manuscript](https://arxiv.org/abs/1908.01425):

```
@misc{korovina2019chembo,
    title={ChemBO: Bayesian Optimization of Small Organic Molecules with Synthesizable Recommendations},
    author={Ksenia Korovina and Sailun Xu and Kirthevasan Kandasamy and Willie Neiswanger and Barnabas Poczos and Jeff Schneider and Eric P. Xing},
    year={2019},
    eprint={1908.01425},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
