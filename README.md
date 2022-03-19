# FLamby
 PLACEHOLDER LOGO

## Table of Contents
- [Overview](#overview)
- [Source datasets](#source-datasets)
- [Extending FLamby](#extending-flamby)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Overview
FLamby is a benchmark for cross-silo Federated Learning with natural partitioning. 
It spans multiple data modalities and should allow easy interfacing with most 
Federated-Learning frameworks ([FedML](https://github.com/FedML-AI/FedML), [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed), [Substra](https://github.com/Substra/substra), ...). It contains implementations of different
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
The next step is to git clone the repository and to cd inside the FLamby directory then the FLamby benchmark can be installed by executing:
```
pip install flamby
```
Then proceed to read and accept the different licenses and download the data from
all the datasets you are interested in by following the instructions provided in each folder:
- [Fed-Camelyon16](./flamby/datasets/fed_camelyon16/README.md)
- [Fed-LIDIC-IDRI](./flamby/datasets/fed_lidc_idri/README.md)

## Usage

Look at our tutorials and get started sections.
#TODO write tutorials and get started sections

## Contributing

After installing the package in dev mode (``pip install -e .``)
You should also initialize ``pre-commit`` by running:
```
pre-commit install
```

The ``pre-commit`` tool will automatically run [black](https://github.com/psf/black) and 
[isort](https://github.com/PyCQA/isort) and check [flake8](https://flake8.pycqa.org/en/latest/) compatibility.
Which will format the code automatically making the code more homogeneous and helping catching typos and errors.

Looking and or commenting the open issues is a good way to start. Once you have found a way to contribute the next steps are:
- Following the installation instructions but using the -e option when pip installing
- Installing pre-commit
- Creating a new branch following the convention name_contributor/short_explicit_name-wpi: `git checkout -b name_contributor/short_explicit_name-wpi`
- Potentially pushing the branch to origin with : `git push origin name_contributor/short_explicit_name-wpi`
- Working on the branch locally by making commits frequently: `git commit -m explicit description of the commit's content`
- Once the branch is ready or after considering you have made significant progresses opening a Pull Request using Github interface, selecting your branch as a source and the target to be the main branch and creating the PR **in draft mode**  after having made **a detailed description of the content of the PR** and potentially linking to related issues.
Rebasing the branch onto main by doing `git fetch origin` and  `git rebase origin/main`, solving potential conflicts adding the resolved files `git add myfile.py`
then continuing with `git rebase --continue` until the rebase is complete. Then pushing the branch to origin with `git push origin --force-with-lease`.
- Waiting for reviews then commiting and pushing changes to comply with the reviewer's requests
- Once the PR is approved click on the arrow on the right of the merge button to select rebase and click on it




