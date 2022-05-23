## KiTS19

KiTS19 dataset is an open access Kidney Tumor Segmentation dataset that was made public in 2019 for a segmentation Challenge (https://kits19.grand-challenge.org/data/).
We use the official KiTS19 repository (https://github.com/neheller/kits19) to download the dataset. 

#License and Citations:
Find attached the link to [the full license](https://data.donders.ru.nl/doc/dua/CC-BY-NC-SA-4.0.html?0) and [dataset terms](https://kits19.grand-challenge.org/data/).

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

|                   | Dataset description 
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from KiTS19 Challenge.
| Dataset           | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmentation masks, we cannot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset. 
| Centers           | Data comes from 87 different centers. The sites information can be found in fed_kits19/dataset_creation_scripts/anony_sites.csv file. Since most the sites have small amount of data, we set a threshold of 10 on the amount of data a silo should have, and include only those silos (total 6) that meet this threshold for the Training. This leaves us with 96 patients data.
| Task              | Supervised Segmentation



## Data Download instructions
The commands for data download
(as given on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Clone the kits19 git repository
```bash
git clone https://github.com/neheller/kits19
```

2. Run the following commands to download the dataset. Make sure you have ~30GB space available.
```bash
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
These commands will populate the data folder (given in the kits19 repository) with the imaging data. 

3. Move the downloaded dataset to the data directory you want to keep it in.
4. To store the data path (path to the data folder given by KiTS19), run the following command in the directory 'fed_kits19/dataset_creation_scripts/nnunet_library/dataset_conversion',
```bash
python3 create_config.py --output_folder "data_folder_path" 
```
Note that it should not include the name of the data folder such as 'Desktop/kits19' can be an example of the "data_folder_path" given data folder resides in the kits19 directory.
## Data Preprocessing   
For preprocessing, we use [nnunet](https://github.com/MIC-DKFZ/nnUNet) library and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) packages. We exploit nnunet preprocessing pipeline
to apply intensity normalization, voxel and foreground resampling. In addition, we apply extensive transformations such as random crop, rotation, scaling, mirror etc from the batchgenerators package. 

1. To run preprocessing, first step is dataset conversion. For this step, go to the following directory from the fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/dataset_conversion
```
and run the following command to prepare the data for preprocessing.
```bash
python3 Task064_KiTS_labelsFixed.py 
```
2. After data conversion, next step is to run the preprocessing which involves, data intensity normalization and voxel resampling. To run preprocessing, run the following command to go to the right directory from fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/experiment_planning
```
and run the following command to preprocess the data,
```bash
python3 nnUNet_plan_and_preprocess.py -t 064
```
For the preprocessing, it can take around ~30-45 minutes. 
With this preprocessing, running the experiments can be very time efficient as it saves the preprocessing time for every experiment run.

## Pooled Experiment
To run a pooled strategy, run the following command in the fed_kits19 directory,
```bash
python3 benchmarks.py --GPU $GPU_ID
```
Estimated memory requirement for this training is around 14.5 GB.

#Citation:
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

