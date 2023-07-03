# KiTS19

KiTS19 dataset is an open access Kidney Tumor Segmentation dataset that was made public in 2019 for a segmentation Challenge (https://kits19.grand-challenge.org/data/).
We use the official KiTS19 repository (https://github.com/neheller/kits19) to download the dataset.

## License and dataset terms of use
The dataset is provided under the [CC-BY-NC-SA-4.0 license](https://data.donders.ru.nl/doc/dua/CC-BY-NC-SA-4.0.html?0).
Please ensure you comply with this license and the [dataset terms](https://kits19.grand-challenge.org/data/)
before using this dataset.

## Ethics
As stated in the [official manuscript](https://arxiv.org/pdf/1904.00445.pdf),
the dataset collection was

>reviewed and approved by the Institutional Review Board at the University of Minnesoty as Study 1611M00821

## Acknowledgements
See below the full citations:
```bash
@article{heller2020state,
  title={The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge},
  author={Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others},
  journal={Medical Image Analysis},
  pages={101821},
  year={2020},
  publisher={Elsevier}
}

@article{heller2019kits19,
  title={The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations, and surgical outcomes},
  author={Heller, Nicholas and Sathianathen, Niranjan and Kalapara, Arveen and Walczak, Edward and Moore, Keenan and Kaluzniak, Heather and Rosenberg, Joel and Blake, Paul and Rengel, Zachary and Oestreich, Makinna and others},
  journal={arXiv preprint arXiv:1904.00445},
  year={2019}
}
```

## Dataset Description
The table below provides summary information.
Please refer to the [data manuscript](https://arxiv.org/pdf/1904.00445.pdf) for a more in-depth data sheet.

|                   | Dataset description
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from KiTS19 Challenge.
| Dataset           | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmentation masks, we cannot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset.
| Centers           | Data comes from 87 different centers. The sites information can be found in fed_kits19/dataset_creation_scripts/anony_sites.csv file. We include only those silos (total of 6) that have greater than 10 data samples (images), which leaves us with 96 patients data samples.
| Task              | Supervised Segmentation



## Data Download instructions
The commands for data download
(as given on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Cd to a different directory with sufficient space to hold kits data (~30GB) and clone the kits19 git repository:
```bash
git clone https://github.com/neheller/kits19
```

2. Proceed to read and accept the license and data terms

3. Run the following commands to download the dataset. Make sure you have ~30GB space available.
```bash
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
These commands will populate the data folder (given in the kits19 repository) with the imaging data.

4. To configure the KiTS19 data path so that it can be accessed by the Flamby library, run the following command in the directory `flamby/datasets/fed_kits19/dataset_creation_scripts/`,
```bash
python3 create_config.py --output_folder /path/where/you/cloned_kits19/kits19
```
You can add an option '--debug', if you want to run the whole pipeline on only a part of the dataset. Note that "/path/where/you/cloned_kits19/kits19" should contain the path to the kits19 git repository, for example, '~/Desktop/kits19' can be an example of the "/path/where/you/cloned_kits19/kits19" given you cloned the kits19 git repository in the Desktop folder and the data folder containing KiTS19 dataset resides in this kits19 git repository.

## Data Preprocessing
For preprocessing, we use [nnunet](https://github.com/MIC-DKFZ/nnUNet) library and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) packages. We exploit nnunet preprocessing pipeline
to apply intensity normalization, voxel and foreground resampling. In addition, we apply extensive transformations such as random crop, rotation, scaling, mirror etc from the batchgenerators package.

1. To run preprocessing, first step is dataset conversion. For this step, go to the following directory from the fed_kits19 directory
```bash
cd dataset_creation_scripts
```
and run the following command to prepare the data for preprocessing.
```bash
python3 parsing_and_adding_metadata.py
```
You should add the option '--debug', if you already did so during the step 3 of the data download.
2. After data conversion, next step is to run the preprocessing which involves, data intensity normalization and voxel resampling. To run preprocessing, run the following command to go to the right directory from fed_kits19 directory
```bash
python3 run_nnUnet_plan_and_preprocess.py --num_threads 1
```
Similarly, you should add the option '--debug' if you used it on the previous steps.

**Warning:** If you use more threads than your machine has available CPUs it, the preprocessing can halt indefinitely.
With this preprocessing, running the experiments can be very time efficient as it saves the preprocessing time for every experiment run.

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_kits19 import FedKits19

# To load the first center as a pytorch dataset
center0 = FedKits19(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedKits19(center=1, train=True)
# To load the 3rd center ...

# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()
```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)

## Pooled Experiment
To run a pooled strategy with GPUs, run the following command in the 'flamby/datasets/fed_kits19' directory,
```bash
python3 benchmark.py --GPU $GPU_ID
```
$GPU_ID should contain the GPU number that will be used to perform training, for example, 3 can be an example of $GPU_ID if you want to run the pooled strategy on 'cuda:3'.
If you don't have a GPU, then --GPU argument can be skipped and the following command can be used,
```bash
python3 benchmark.py
```
Note that estimated memory requirement for this training is around 14.5 GB.

# Citation:
```bash
@article{isensee2018nnu,
  title={nnu-net: Self-adapting framework for u-net-based medical image segmentation},
  author={Isensee, Fabian and Petersen, Jens and Klein, Andre and Zimmerer, David and Jaeger, Paul F and Kohl, Simon and Wasserthal, Jakob and Koehler, Gregor and Norajitra, Tobias and Wirkert, Sebastian and others},
  journal={arXiv preprint arXiv:1809.10486},
  year={2018}
}

@misc{isensee2020batchgenerators,
  title={batchgenerators—a python framework for data augmentation. 2020},
  author={Isensee, F and J{\"a}ger, P and Wasserthal, J and Zimmerer, D and Petersen, J and Kohl, S and others},
  year={2020}
}
```
