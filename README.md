# FLamby
- [Overview](#overview)
- [Source datasets](#source-datasets)
- [Extending FLamby](#extending-flamby)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Overview
FLamby is a benchmark for cross-silo Federated Learning with natural partitioning. 
It spans multiple data modalities and should allow easy interfacing with most 
Federated-Learning frameworks (FedML, Fed-BioMed, ...). It contains implementations of different
standard strategies.

The FLamby package contains:

- Data loaders that automatically handle data preprocessing and splitting for distributed datasets.  
- Evaluation functions to evaluate trained models on the different tracks as defined in the companion paper. 
- Benchmark code using the utilities below to obtain the performances of baselines using different strategies.

It does not contain datasets that have to be downloaded separately.
See the section below.

## Source datasets
We do not distribute datasets in this repository. **The use of any of the datasets
included in FLamby requires accepting its corresponding license on the original
website.**
We do not own copyrights on any of the datasets.


## Extending FLamby

If you would like to add another cross-silo dataset please fork the repository
and do a Pull-Request.
Similarly one can add the results of a new strategy/training algorithm.

## Installation

We recommend using anaconda and pip.

First setup a virtual environment (optional) with Python>=3.8.

```
conda create -n flamby_env python=3.8
conda activate flamby_env
```
Then install the FLamby benchmark by executing:
```
pip install flamby
```
Then proceed to read and accept the different licenses and download the data from
all the datasets you are interested in by following the instructions provided in each folder:
- [Fed-Camelyon16](./flamby/datasets/fed_camelyon16/README.md)
- [Fed-LIDIC-IDRI](./flamby/datasets/fed_lidc_idri/README.md)

## Usage

Look at our tutorials and get started sections.

## Contributing

After installing the package in dev mode (``pip install -e .``)
You should also initialize ``pre-commit``:
```
pre-commit install
```

The ``pre-commit`` tool will automatically run [black](https://github.com/psf/black) and 
[isort](https://github.com/PyCQA/isort) and check [flake8](https://flake8.pycqa.org/en/latest/) compatibility.
