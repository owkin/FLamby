# FLamby
<img src="/docs/logo.png" width="600">


## Table of Contents
- [Overview](#overview)
- [Dataset suite](#dataset-suite)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [FAQ](#FAQ)
- [Team](#team)
- [Acknowledgements](#acknowledgements)


## Overview

:arrow_right:[The API doc is available here](https://owkin.github.io/FLamby):arrow_left:

FLamby is a benchmark for cross-silo Federated Learning with natural partitioning,
currently focused in healthcare applications.
It spans multiple data modalities and should allow easy interfacing with most
Federated Learning frameworks (including [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed), [FedML](https://github.com/FedML-AI/FedML), [Substra](https://docs.substra.org/en/stable/)...).
It contains implementations of different
standard federated learning strategies.

The FLamby package contains:

- Data loaders that automatically handle data preprocessing and partitions of distributed datasets.
- Evaluation functions to evaluate trained models on the different tracks as defined in the companion paper.
- Benchmark code using the utilities below to obtain the performances of baselines using different strategies.

It does not contain datasets, which have to be downloaded separately (see the section below).

FLamby was tested on Ubuntu and MacOS environment. If you are facing any problems installing or executing FLamby code
please help us improve it by filing an issue on FLamby github page ensuring to explain it in detail.

## Dataset suite
FLamby is a dataset **suite** instead of a repository.
We provide code to easily access existing datasets stored in other repositories.
In particular, we do not distribute datasets in this repository,
and we do not own copyrights on any of the datasets.

**The use of any of the datasets
included in FLamby requires accepting its corresponding license on the original
website.** We refer to each dataset's README for more details.

For any problem or question with respect to any license related matters,
please open a github issue on this repository.

## Installation

We recommend using anaconda and pip. You can install anaconda by downloading and executing appropriate installers from the [Anaconda website](https://www.anaconda.com/products/distribution), pip often comes included with python otherwise check [the following instructions](https://pip.pypa.io/en/stable/installation/). We support all Python version starting from **3.7**.

You may need `make` for simplification. The following command will install all packages used by all datasets within FLamby. If you already know you will only need a fraction of the datasets inside the suite you can do a partial installation and update it along the way using the options described below.
Create and launch the environment using:

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make install
conda activate flamby
```

To limit the number of installed packages you can use the `enable` argument to specify which dataset(s)
you want to build required dependencies for and if you need to execute the tests (tests) and build the documentation (docs):

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make enable=option_name install
conda activate flamby
```

where `option_name` can be one of the following:
cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests

if you want to use more than one option you can do it using comma
(**WARNING:** there should be no space after `,`), eg:

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make enable=cam16,kits19,tests install
conda activate flamby
```
Be careful, each command tries to create a conda environment named flamby therefore make install will fail if executed
numerous times as the flamby environment will already exist. Use `make update` as explained in the next section if you decide to
use more datasets than intended originally.

### Update environment
Use the following command if new dependencies have been added, and you want to update the environment for additional datasets:
```bash
make update
```

or you can use `enable` option:
```bash
make enable=cam16 update
```


### In case you don't have the `make` command (e.g. Windows users)
You can install the environment by running:
```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
conda env create -f environment.yml
conda activate flamby
pip install -e .[all_extra]
```

or if you wish to install the environment for only one or more datasets, tests or documentation:
```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
conda env create -f environment.yml
conda activate flamby
pip install -e .[option_name]
```

where `option_name` can be one of the following:
cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests. If you want to use more than one option you can do it
using comma (`,`) (no space after comma), eg:
```bash
pip install -e .[cam16,ixi]
```

### Accepting data licensing
Then proceed to read and accept the different licenses and download the data from
all the datasets you are interested in by following the instructions provided in each folder:
- [Fed-Camelyon16](./flamby/datasets/fed_camelyon16/README.md)
- [Fed-LIDC-IDRI](./flamby/datasets/fed_lidc_idri/README.md)
- [Fed-ISIC2019](./flamby/datasets/fed_isic2019/README.md)
- [Fed-TCGA-BRCA](./flamby/datasets/fed_tcga_brca/README.md)
- [Fed-Heart-Disease](./flamby/datasets/fed_heart_disease/README.md)
- [Fed-IXITiny](./flamby/datasets/fed_ixi/README.md)
- [Fed-KITS2019](./flamby/datasets/fed_kits19/README.md)

## Quickstart

Follow the [quickstart section](./Quickstart.md) to learn how to get started with FLamby.

## Reproduce benchmark and figures from the companion article

### Benchmarks
The results are stored in flamby/results in corresponding subfolders `results_benchmark_fed_dataset` for each dataset.
These results can be plotted using:
```
python plot_results.py
```
which produces the plot at the end of the main article.

In order to re-run each of the benchmark on your machine, first download the dataset you are interested in and then run the following command replacing `config_dataset.json` by one of the listed config files (`config_camelyon16.json`, `config_heart_disease.json`, `config_isic2019.json`, `config_ixi.json`, `config_kits19.json`, `config_lidc_idri.json`, `config_tcga_brca.json`):
```
cd flamby/benchmarks
python fed_benchmark.py --seed 42 -cfp ../config_dataset.json
python fed_benchmark.py --seed 43 -cfp ../config_dataset.json
python fed_benchmark.py --seed 44 -cfp ../config_dataset.json
python fed_benchmark.py --seed 45 -cfp ../config_dataset.json
python fed_benchmark.py --seed 46 -cfp ../config_dataset.json
```
We have observed that results vary from machine to machine and are sensitive to GPU randomness. However you should be able to reproduce the results up to some variance and results on the same machine should be perfecty reproducible. Please open an issue if it is not the case.
The script `extract_config.py` allows to go from a results file to a `config.py`.
See the [quickstart section](./Quickstart.md) to change parameters.

## Containerized execution

A good step towards float-perfect reproducibility in your future benchmarks is to use docker. We give a base docker image and examples containing dataset download and benchmarking.
For [Fed-Heart-Disease](./flamby/datasets/fed_heart_disease/README.md), cd to the flamby dockers folder, replace `myusername` and `mypassword` with your git credentials (OAuth token) in the command below and run:
```
docker build -t flamby-heart -f Dockerfile.base --build-arg DATASET_PREFIX="heart" --build-arg GIT_USER="myusername" --build-arg GIT_PWD="mypassword" .
docker build -t flamby-heart-benchmark -f Dockerfile.heart .
docker run -it flamby-heart-benchmark
```
If you are convinced you will use many datasets with docker, build the base image using `all_extra` option for flamby's install, you will be able to reuse it for all datasets with multi-stage build:
```
docker build -t flamby-all -f Dockerfile.base --build-arg DATASET_PREFIX="all_extra" --build-arg GIT_USER="myusername" --build-arg GIT_PWD="mypassword" .
# modify Dockerfile.* line 1 to FROM flamby-all by replacing * with the dataset name of the dataset you are interested in
# Then run the following command replacing * similarly
#docker build -t flamby-* -f Dockerfile.* .
#docker run -it flamby-*-benchmark
```

Checkout `Dockerfile.tcga`.
Similar dockerfiles can be theoretically easily built for the other datasets as well by
replicating instructions found in each dataset folder following the model of `Dockerfile.heart`.
Note that for bigger datasets execution can be prohibitively slow and docker can run out of time/memory.



### Using FLamby with FL-frameworks

FLamby can be easily adapted to different frameworks as the pytorch abstractions
are quite flexible.
We give an example of interfacing with [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed) [here](./integration/fedbiomed/flamby-integration-into-fedbiomed.md), another one with [FedML](https://github.com/FedML-AI/FedML), [here](./integration/fedbiomed/flamby-integration-into-fedbml.md) and last one with [Substra](https://docs.substra.org/en/stable/) [there](./integration/Substra/flamby-integration-into-substra.md).
All major FL-frameworks should be compatible with FLamby modulo some glue code.
If you have a working example of using FLamby with another FL-framework please open a PR.

### Heterogeneity plots

Most plots from the article can be reproduced using the following commands after having downloaded the corresponding datasets:
- [Fed-TCGA-BRCA](./flamby/datasets/fed_tcga_brca/README.md)
```
cd flamby/datasets/fed_tcga_brca
python plot_kms.py
```
- [Fed-LIDC-IDRI](./flamby/datasets/fed_lidc_idri/README.md)
```
cd flamby/datasets/fed_lidc_idri
python lidc_heterogeneity_plot.py
```
- [Fed-ISIC2019](./flamby/datasets/fed_isic2019/README.md) 

**In order to exactly reproduce the plot in the article**, one needs to first deactivate color constancy normalization when preprocessing the dataset (change `cc` to `False` in `resize_images.py`) when following download and preprocessing instructions [here](./flamby/datasets/fed_isic2019/README.md).  Hence one might have to download the dataset a second time, if it was already downloaded, and therefore to potentially update `dataset_location.yaml` files accordingly.
```
cd flamby/datasets/fed_isic2019
python heterogeneity_pic.py
```
- [Fed-IXITiny](./flamby/datasets/fed_ixi/README.md)
```
cd flamby/datasets/fed_ixi
python ixi_plotting.py
```
- [Fed-KITS2019](./flamby/datasets/fed_kits19/README.md)
```
cd flamby/datasets/fed_kits19/dataset_creation_scripts
python kits19_heterogenity_plot.py
```
- [Fed-Heart-Disease](./flamby/datasets/fed_heart_disease/README.md)
```
cd flamby/datasets/fed_heart_disease
python heterogeneity_plot.py
```
- [Fed-Camelyon16](./flamby/datasets/fed_camelyon16/README.md) 

First concatenate as many 224x224 image patches extracted from regions on the slides containing matter from Hospital 0 and Hospital 1 (see what is done in the [tiling script](./flamby/datasets/fed_camelyon16/dataset_creation_scripts/tiling_slides.py) to collect image patches) as can be fit in
the RAM. Then compute both histograms **per-color-channel** using 256 equally sized bins with the `np.histogram` 
function with `density=True`. 
Then save the results respectively as: histogram_0.npy, histogram_1.npy and bins_0.npy
```
cp -t flamby/datasets/fed_camelyon16 histograms_{0, 1}.npy bins_0.npy
cd flamby/datasets/fed_camelyon16
python plot_camelyon16_histogram.py 
```

## Deploy documentations

We use [sphinx](https://www.sphinx-doc.org/en/master/) to create FLamby's documentation.
In order to build the doc locally, activate the environment then:
```bash
cd docs
make clean
make html
```
This will generate html pages in the folder _builds/html that can be accessed in your browser:
```bash
open _build/html/index.html
```
## Contributing

### Extending FLamby

FLamby is a living project and contributions by the FL community are welcome.

If you would like to add another cross-silo dataset **with natural splits**,
please fork the repository
and do a Pull-Request following the guidelines described below.

Similarly, you can propose pull requests introducing novel training algorithms
or models.

### Guidelines

After installing the package in dev mode (``pip install -e .[all_extra]``)
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
- Working on the branch locally by making commits frequently: `git commit -m "explicit description of the commit's content"`
- Once the branch is ready or after considering you have made significant progresses opening a Pull Request using Github interface, selecting your branch as a source and the target to be the main branch and creating the PR **in draft mode**  after having made **a detailed description of the content of the PR** and potentially linking to related issues.
Rebasing the branch onto main by doing `git fetch origin` and  `git rebase origin/main`, solving potential conflicts adding the resolved files `git add myfile.py`
then continuing with `git rebase --continue` until the rebase is complete. Then pushing the branch to origin with `git push origin --force-with-lease`.
- Waiting for reviews then commiting and pushing changes to comply with the reviewer's requests
- Once the PR is approved click on the arrow on the right of the merge button to select rebase and click on it


# FAQ
### How can I do a clean slate?
To clean the environment you must execute (after being inside the FLamby folder `cd FLamby/`):
```bash
conda deactivate
make clean
```

### I get an error when installing Flamby
#### error: [Errno 2] No such file or directory: 'pip'
Try running:
```bash
conda deactivate
make clean
pip3 install --upgrade pip
```
and try running your make installation option again.

### I am installing Flamby on a machine equipped with macOS and an intel processor
In that case, you should use 
```bash
make install-mac
```
instead of the standard installation. If you have already installed the flamby environment, just run 
```bash
conda deactivate 
make clean
```
before running the install-mac installation again. This is to avoid the following error, which will appear when running scripts.
#### error : OMP: Error #15  

### I or someone else already downloaded a dataset using another copy of the flamby repository, my copy of flamby cannot find it and I don't want to download it again, what can I do ?

There are two options. The safest one is to cd to the flamby directory and run:
```
python create_dataset_config.py --dataset-name fed_camelyon16 OR fed_heart_disease OR ... --path /path/where/the/dataset/is/located
```
This will create the required `dataset_location.yaml` file in your copy of the repository allowing FLamby to find it.

One can also directly pass the `data_path` argument when instantiating the dataset but this is not recommended.
```python
from flamby.datasets.fed_heart_disease import FedHeartDisease
center0 = FedHeartDisease(center=0, train=True, data_path="/path/where/the/dataset/is/located")
```
### Collaborative work on FLamby: I am working with FLamby on a server with other users, how can we share the datasets efficiently ?

The basic answer is to use the answer just above to recreate the config file in every copy of the repository.

It can possibly become more seamless in the future if we introduce checks for environment variables in FLamby, which would allow to setup a general server-wise config so that all users of the server have access to all needed paths.
In the meantime one can fill/comment the following bash script after downloading the dataset and share it with all users of the server:

```bash
python create_dataset_config.py --dataset-name fed_camelyon16 --path TOFILL
python create_dataset_config.py --dataset-name fed_heart_disease --path TOFILL
python create_dataset_config.py --dataset-name fed_lidc_idri --path TOFILL
python create_dataset_config.py --dataset-name fed_kits19 --path TOFILL
python create_dataset_config.py --dataset-name fed_isic2019 --path TOFILL
python create_dataset_config.py --dataset-name fed_ixi --path TOFILL
```
Which allows users to set all necessary paths in their local copies.

### Can I run clients in different threads with FLamby? How does it run under the hood?

FLamby is a lightweight and simple solution, designed to allow researchers
to quickly use cleaned datasets with a standard API.
As a consequence, the benchmark code performing the FL simulation is minimalistic.
All clients run sequentially in the same python environment, without multithreading.
Datasets are assigned to clients as different python objects.

### Does FLamby support GPU acceleration?

FLamby supports GPU acceleration thanks to the underlying deep learning backend
(pytorch for now).

## Team

This repository was made possible thanks to numerous contributors.
We list them in the order of the companion article, following the [CREDIT framework](https://credit.niso.org/):
[Jean Ogier du Terrail](https://github.com/jeandut), [Samy-Safwan Ayed](https://github.com/AyedSamy), [Edwige Cyffers](https://github.com/totilas), [Felix Grimberg](https://github.com/Grim-bot), [Chaoyang He](https://github.com/chaoyanghe), [Régis Loeb](https://github.com/regloeb), [Paul Mangold](https://github.com/pmangold), [Tanguy Marchand](https://github.com/tanguy-marchand), [Othmane Marfoq](https://github.com/omarfoq/), [Erum Mushtaq](https://github.com/ErumMushtaq), [Boris Muzellec](https://github.com/BorisMuzellec), [Constantin Philippenko](https://github.com/philipco), [Santiago Silva](https://github.com/sssilvar), [Maria Telenczuk](https://github.com/maikia), [Shadi Albarqouni](https://albarqouni.github.io/), [Salman Avestimehr](https://www.avestimehr.com/), [Aurélien Bellet](http://researchers.lille.inria.fr/abellet/), [Aymeric Dieuleveut](http://www.cmap.polytechnique.fr/~aymeric.dieuleveut/), [Martin Jaggi](https://people.epfl.ch/martin.jaggi), [Sai Praneeth Karimireddy](https://github.com/Saipraneet), [Marco Lorenzi](https://marcolorenzi.github.io/publications.html), [Giovanni Neglia](http://www-sop.inria.fr/members/Giovanni.Neglia/publications.htm), [Marc Tommasi](http://researchers.lille.inria.fr/tommasi/#publications), [Mathieu Andreux](https://github.com/mandreux-owkin).

## Acknowledgements
FLamby was made possible thanks to the support of the following institutions:
- [Owkin](https://www.owkin.com)
- [Inria](https://www.inria.fr)
- [Ecole polytechnique](https://www.polytechnique.edu)
- [University of California - Berkeley](https://www.berkeley.edu/)
- [University of Southern California (USC)](https://www.usc.edu/)
- [EPFL](https://www.epfl.ch)
- [Universitätsklinikum Bonn](https://www.ukbonn.de/patient_innen/international/english/)

<img src="docs/img/Owkin_Logo_Black.png" alt="owkin-logo" width="200"/><img src="docs/img/inr_logo_rouge.jpg" alt="inria-logo" width="200"/><img src="docs/img/ecole_polytechnique.png" alt="ecole-polytechnique-logo" width="200"/><img src="docs/img/Berkeley_wordmark_blue.png" alt="berkeley-logo" width="200"/>
<img src="docs/img/usc-logo.png" alt="usc-logo" width="200"/><img src="docs/img/logo-epfl-1024x576.png" alt="epfl-logo" width="200"/><img src="docs/img/ukb-logo.png" alt="ukb-logo" width="200"/>

## Citing FLamby

```bibtex
@inproceedings{NEURIPS2022_232eee8e,
 author = {Ogier du Terrail, Jean and Ayed, Samy-Safwan and Cyffers, Edwige and Grimberg, Felix and He, Chaoyang and Loeb, Regis and Mangold, Paul and Marchand, Tanguy and Marfoq, Othmane and Mushtaq, Erum and Muzellec, Boris and Philippenko, Constantin and Silva, Santiago and Tele\'{n}czuk, Maria and Albarqouni, Shadi and Avestimehr, Salman and Bellet, Aur\'{e}lien and Dieuleveut, Aymeric and Jaggi, Martin and Karimireddy, Sai Praneeth and Lorenzi, Marco and Neglia, Giovanni and Tommasi, Marc and Andreux, Mathieu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {5315--5334},
 publisher = {Curran Associates, Inc.},
 title = {FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/232eee8ef411a0a316efa298d7be3c2b-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}

```
