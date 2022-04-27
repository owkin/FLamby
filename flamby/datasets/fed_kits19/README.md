## KiTS19

KiTS19 dataset is an open access Kidney Tumor Segmentation dataset that was made public in 2019 for a segmentation Challenge.
We use the official KiTS19 repository (https://github.com/neheller/kits19) to download the dataset. 

## Dataset Description

|                   | Dataset description 
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from KiTS19 Challenge.
| Dataset           | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmetation masks, we canot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset. 
| Centers           | Data comes from 87 different centers.
| Task              | Supervised Segmentation



## Data Download instructions
The commands for data download
(as also mentioned on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Clone the kits19 git repository
```bash
git clone https://github.com/neheller/kits19
```

2. Follow the following commands to download the dataset,
```bash
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
These commands will download the KiTS19 dataset in your download directory. 

3. Move the downloaded dataset to the data directory you want to keep it in and save
the path in the data_directory.yaml file present in the flamby/datasets/fed_kits19/dataset_creation_scripts folder.
   
4. For preprocessing, we rely on nnunet library as it is considered a benchmark for this dataset. {Side fact: The team won the KiTS19 challenge.} Any changes made to this library are explicity mentioned as changed in comments. 
   To run preprocessing, {To be added} 



