## KiTS19

KiTS19 dataset is an open access Kidney Tumor Segmentation dataset that was made public in 2019 for a segmentation Challenge.
We use the official KiTS19 repository (https://github.com/neheller/kits19) to download the dataset. 

#Citations:
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
| Dataset           | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmetation masks, we canot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset. 
| Centers           | Data comes from 87 different centers. The sites information can be found in fed_kits19/dataset_creation_scripts/anony_sites.csv file.
| Task              | Supervised Segmentation



## Data Download instructions
The commands for data download
(as also mentioned on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Clone the kits19 git repository
```bash
git clone https://github.com/neheller/kits19
```

2. Follow the following commands to download the dataset. Make sure you have ~30GB space available.
```bash
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
These commands will download the KiTS19 dataset in your download directory. 

3. Move the downloaded dataset to the data directory you want to keep it in, and save
4. To store the data path (path to the datafolder given by KiTS19), run the following command in the directory 'fed_kits19/dataset_creation_scripts/nnunet_library/dataset_conversion',
```bash
python3 create_config.py --output_folder "data_folder_path" --debug True
```
OR
```bash
python3 create_config.py --output_folder "data_folder_path" --debug ''
```
## Data Preprocessing   
For preprocessing, we use nnunet library and batchgenerators packages.
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

1. To run preprocessing, first step is dataset conversion, from fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/dataset_conversion
```
and run the following command to prepare a small chunk of data for preprocessing, 
```bash
python3 Task064_KiTS_labelsFixed.py --debug True 
```
if you want complete data to be preprocessed, set debug argument to be False as follows,
```bash
python3 Task064_KiTS_labelsFixed.py --debug False
```

2. After data conversion, run the preprocessing, from fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/experiment_planning
```
and run the following command to preprocess the data,
```bash
python3 nnUNet_plan_and_preprocess.py -t 064
```
For debug mode, it can take ~2-3 hours to preprocess the data. For the complete dataset, it can be take even longer however, once data is preprocessed,
running the experiments can be very time efficient as it saves the preprocessing time for every experiment run.



