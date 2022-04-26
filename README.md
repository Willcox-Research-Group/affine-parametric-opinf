[![License](https://img.shields.io/github/license/Willcox-Research-Group/affine-parametric-opinf.svg)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/Willcox-Research-Group/affine-parametric-opinf.svg)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/Willcox-Research-Group/affine-parametric-opinf.svg)
[![Latest commit](https://img.shields.io/github/last-commit/Willcox-Research-Group/affine-parametric-opinf.svg)](https://github.com/Willcox-Research-Group/affine-parametric-opinf/commits/main)
[![Research article](https://img.shields.io/badge/PDF-arXiv-A42C25.svg)](https://arxiv.org/pdf/2110.07653.pdf)

# Operator Inference for Parametric PDEs

This repository is the source code for the preprint [_Non-intrusive reduced-order models for parametric partial differential equations via data-driven operator inference_](https://arxiv.org/abs/2110.07653) ([PDF](https://arxiv.org/pdf/2110.07653.pdf)) by [McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Khodabakhshi](https://scholar.google.com/citations?user=lYr_g-MAAAAJ), and [Willcox](https://kiwi.oden.utexas.edu/).<details><summary>BibTeX</summary><pre>
@article{mcquarrie2021opinf,
    author = {Shane A. McQuarrie and Parisa Khodabakhshi and Karen E. Willcox},
    title = {Non-intrusive reduced-order models for parametric partial differential equations via data-driven operator inference},
    journal = {arXiv preprint arXiv:2110.07753},
    year = {2021},
}</pre></details>

<p align="center">
    <img src="https://github.com/Willcox-Research-Group/affine-parametric-opinf/blob/images/fhn.gif">
</p>

## Abstract

This work formulates a new approach to reduced modeling of parameterized, time-dependent partial differential equations (PDEs). The method employs Operator Inference, a scientific machine learning framework combining data-driven learning and physics-based modeling. The parametric structure of the governing equations is embedded directly into the reduced-order model, and parameterized reduced-order operators are learned via a data-driven linear regression problem. The result is a reduced-order model that can be solved rapidly to map parameter values to approximate PDE solutions. Such parameterized reduced-order models may be used as physics-based surrogates for uncertainty quantification and inverse problems that require many forward solves of parametric PDEs. Numerical issues such as well-posedness and the need for appropriate regularization in the learning problem are considered, and an algorithm for hyperparameter selection is presented. The method is illustrated for a parametric heat equation and demonstrated for the FitzHugh-Nagumo neuron model (shown above).

## Repository Contents

**Heat Equation**
- [`heat.py`](./heat.py): defines classes for solving the one-dimensional parametric heat problem with piecewise constant diffusion.
    - `HeatSolver`: high-fidelity finite difference solver.
    - `HeatROM`: operator inference reduced-order model solver.

**FitzHugh-Nagumo System**
- [`fhn.py`](./fhn.py): defines classes for solving the FitzHugh-Nagumo neuron model.
    - `FHNSolver`: high-fidelity finite difference solver.
    - `FHNROMSolver`: reduced-order model solver.
    - `AffineFHNROM`: operator inference reduced-order model.
    - `AffineFHNROM_Intrusive`: reduced-order model from intrusive projection.
- [`fhn_rom_search.py`](./fhn_rom_search.py): script for operator inference hyperparameter search.

**Utilities**
- [`config.py`](./config.py): configuration (naming conventions, plot customizations, etc.).
- [`utils.py`](./utils.py): utilities (logging, timing, data management).

## Citation

If you find this repository useful, please consider citing our paper:

[McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Khodabakhshi, P](https://scholar.google.com/citations?user=lYr_g-MAAAAJ&hl=en&oi=ao) and [Willcox, K. E.](https://kiwi.oden.utexas.edu/), [**Non-intrusive reduced-order models for parametric partial differential equations via data-driven operator inference**](https://arxiv.org/abs/2110.07653). _arXiv preprint 2110.07653_, 2021.
```
@article{mcquarrie2021popinf,
    author = {Shane A. McQuarrie and Parisa Khodabakhshi and Karen E. Willcox},
    title = {Non-intrusive reduced-order models for parametric partial differential equations via data-driven operator inference},
    journal = {arXiv preprint arXiv:2110.07753},
    year = {2021},
}
```
