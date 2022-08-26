# Camelyon16

Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will first fetch the slides from the public Google Drive, and will then tile
the matter using a feature extractor, producing a bag of features for each slide.

## Dataset description
Please refer to the [dataset website](https://camelyon17.grand-challenge.org/Data/)
for an [exhaustive data sheet](https://academic.oup.com/gigascience/article/7/6/giy065/5026175#117856577). The table below provides a high-level description
of the dataset.

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Dataset from Camelyon16
| Dataset size       | 900 GB (and 50 GB after features extraction).
| Centers            | 2 centers - RUMC and UMCU.
| Records per center | RUMC: 169 (Train) + 74 (Test), UMCU: 101 (Train) + 55 (Test)
| Inputs shape       | Tensor of shape (10000, 2048) (after feature extraction).
| Total nb of points | 399 slides.
| Task               | Weakly Supervised (Binary) Classification.

### License and terms of use
This dataset is licensed under an open access Creative Commons 1.0 Universal (**CC0 1.0**)
license by its authors.
*Anyone using this dataset should abide by its licence and*
*give proper attribution to the original authors.*

### Ethical approval
As indicated by the [dataset authors](https://academic.oup.com/gigascience/article/7/6/giy065/5026175#117856619),
> The collection of the data was approved by the local ethics committee
> (Commissie Mensgebonden Onderzoek regio Arnhem - Nijmegen) under 2016-2761,
> and the need for informed consent was waived.

## Download instructions

### Introduction
The dataset is hosted on a [several mirrors](https://camelyon17.grand-challenge.org/Data/) (GigaScience, Google Drive, Baidu Pan).
We provide below some scripts to automatically download the dataset
based on the Google Drive API, which requires a Google Account.
If you do not have a Google account, you can alternatively download
manually the dataset through one of the mirrors.
You will find below detailed instructions for each method.
In both cases, make sure you have enough space to store the raw dataset (~900GB).

### Method A: Automatic download with the Google Drive API

In order to use the Google Drive API you need to have a google account and to access the [google developpers console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) in order to get a json containing an OAuth2.0 secret.

All steps necessary to obtain the JSON are described in numerous places in the internet such as in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html), or in this [very nice tutorial's first 5 minutes](https://www.youtube.com/watch?v=1y0-IfRW114) on Youtube.
It should not take more than 5 minutes. The important steps are listed below.

#### Step 1: Setting up Google App and associated secret

1. Create a project in [Google console](https://console.cloud.google.com/apis/credentials/consent?authuser=1). For instance, you can call it `flamby`.
2. Go to Oauth2 consent screen (on the left of the webpage), choose a name for your app and publish it for external use.
3. Go to Credentials, create an id, then client oauth id
4. Choose Web app, go through the steps and **allow URI redirect** towards http://localhost:6006 and http://localhost:6006/ (notice the last backslash)
5. Retrieve the secrets in JSON by clicking on Download icon at the end of the process.
6. Enable Google Drive API for this project, by clicking on "API and services" on the left panel

Then copy-paste your secrets to the directory you want:

```bash
cp ~/Downloads/code_secret_client_bignumber.apps.googleusercontent.com.json client_secrets.json
```

#### Step 2: Downloading the dataset

- **Remark 1: If you are downloading on a remote server**, make sure you do ssh forwarding of the port 6006 onto the port 6006 of your laptop.
- Remark 2 : Make sure you have enough space to hold the dataset (900GB).
-
First cd into the `dataset_creation_scripts` folder:
```bash
cd flamby/datasets/fed_camelyon16/dataset_creation_scripts
```

Then run:

```bash
python download.py --output-folder ./camelyon16_dataset --path-to-secret /path/towards/client_secrets.json --port 6006
```

The first time this script is launched, the user will be asked to explicitly allow the app to operate by logging into his/her Google account (hence the need for the port 6006 forwarding in the case of a remote machine without browser).

This script will download all of Camelyon's slides in the output folder. As there are multiple
slides that are quite big, this script can take a few hours to complete. It can be stopped and
resumed anytime however if you are ssh into a server better use detached mode (screenrc/tmux/etc.).

**IMPORTANT :** If you choose to relocate the dataset after downloading it, it is
imperative that you run the following script otherwise all subsequent scripts will not find it:
```
python update_config.py --new-path /new/path/towards/dataset #adding --debug if you are in debug mode
```

### Method B: Manual download from the official mirrors
We are interested in the Camelyon16 portion of the [Camelyon dataset](https://camelyon17.grand-challenge.org/Data/).
In the following, we will detail the steps to manually download the dataset
from the Google Drive repository.
You can easily adapt the steps to the other mirrors.

Camelyon16 is stored on a public [Google Drive](https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M?resourcekey=0-FREBAxB4QK4bt9Zch_g5Mg).
The dataset is pre-split into training and testing slides. The training slides
are further divided into 2 folders: normal and tumor.
Download all the `.tif` files in the [normal](https://drive.google.com/drive/folders/0BzsdkU4jWx9BNUFqRE81QS04eDg?resourcekey=0-p6LFOzRfCTfyi_JpshhoTQ),
[tumor](https://drive.google.com/drive/folders/0BzsdkU4jWx9BUzVXeUg0dUNOR1U?resourcekey=0-dODmENBQPCw06DITRJfnfg) and [testing images](https://drive.google.com/drive/folders/0BzsdkU4jWx9BWk11WEtZZUNFY0U?resourcekey=0-U0E7SyHPJeQd77VAi3z15Q) folders.
Put all the resulting files into a single folder.
You should end up with 399 `.tif` files in a given folder `PATH-TO-FOLDER`.

The last step consists in creating a metadata file that will be used by the
preprocessing step. Create a file name `dataset_location.yaml` under
`flamby/datasets/fed_camelyon16/dataset_creation_scripts/` with the following content:

```yaml
dataset_path: PATH-TO-FOLDER
download_complete: true
```

The download is now complete.
## Dataset preprocessing (tile extraction)

The next step is to tile the matter on each slide with a feature extractor pretrained on IMAGENET.

We will use the [histolab package](https://github.com/histolab/histolab) to segment the matter on each slide and torchvision to download a pretrained ResNet50 that will be applied on each tile to convert each slide to a bag of numpy features.
This package requires the installation of [Openslide](https://openslide.org/download/). The associated webpage contains instructions to install it on every major distributions. On Linux simply run:

```python
sudo apt-get install openslide-tools
```

One can choose to remove or not the original slides that take up quite some space to keep only the features (therefore using only approximatively 50GB instead of 800GB).

As extracting the matter on all the slides is a lengthy process this script might take a few hours (and a few days if the tiling is done from scratch).
It can also be stopped and resumed anytime and should be preferably run in detached mode.
This process should be run on an environment with GPU, otherwise it might be prohibitively slow.

```bash
python tiling_slides.py --batch-size 64
```

or

```bash
python tiling_slides.py --batch-size 64 --remove-big-tiff
```

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by running in a python shell:

```python
from flamby.datasets.fed_camelyon16 import FedCamelyon16, Camelyon16Raw

# To load the first center as a pytorch dataset
center0 = FedCamelyon16(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedCamelyon16(center=1, train=True)

# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl
# For this specific dataset samples do not have the same size and therefore batching requires padding implemented in collate_fn
from flamby.datasets.fed_camelyon16 import collate_fn

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)).next()

```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)

## Benchmarking the baseline in a pooled setting

In order to benchmark the baseline on the pooled dataset one needs to download and preprocess the dataset and launch the following script:

```bash
python benchmark.py --log --num-workers-torch 10
```

This will launch 5 single-centric runs and store log results for training in ./runs/seed42-47 and testing in ./runs/tests-seed42-47.
The command:

```bash
tensorboard --logdir=./runs
````

can then be used to visualize results (use [port forwarding if necessary](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)).
