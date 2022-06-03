## Camelyon16

Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will use the [Google-Drive-API-v3](https://developers.google.com/drive/api/v3/quickstart/python) in order to fetch the slides from the public Google Drive and will then tile the matter using a feature extractor producing a bag of features for each slide.

## Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Dataset from Camelyon16
| Dataset size       | 900 GB (and 50 GB after features extraction).
| Centers            | 2 centers - RUMC and UMCU.
| Records per center | RUMC: 170 (Train) + 89 (Test), UMCU: 100 (Train) + 50 (Test)
| Inputs shape       | Tensor of shape (10000, 2048) (after feature extraction).
| Total nb of points | 399 slides.
| Task               | Weakly Supervised (Binary) Classification.


## Download and preprocessing instructions

### Introduction
In order to use the Google Drive API you need to have a google account and to access the [google developpers console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) in order to get a json containing an OAuth2.0 secret.  

All steps necessary to obtain the JSON are described in numerous places in the internet such as in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html), or in this [very nice tutorial's first 5 minutes](https://www.youtube.com/watch?v=1y0-IfRW114) on Youtube.
It should not take more than 5 minutes. The important steps are listed below.

### Step 1: Setting up Google App and associated secret

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

### Step 2: Downloading the dataset

- **Remark 1: If you are downloading on a remote server**, make sure you do ssh forwarding of the port 6006 onto the port 6006 of your laptop.
- Remark 2 : Make sure you have enough space to hold the dataset (900GB).

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

### Step 3: Dataset preprocessing (tile extraction)

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

# To load the first center
center1 = FedCamelyon16(center=0, train=True)
# To load the second center
center1 = FedCamelyon16(center=1, train=True)
```

## Benchmarking the baseline in a pooled setting

In order to benchmark the baseline on the pooled dataset one needs to download and preprocess the dataset and launch the following script:

```bash
python benchmark.py --log --num-workers-torch 10 
```

This will launch 5 runs and store log results for training in ./runs/seed42-47 and testing in ./runs/tests-seed42-47.
The command:

```bash
tensorboard --logdir=./runs
````

can then be used to visualize results (use [port forwarding if necessary](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)).
