This repo is based on the training dataset of the ISIC 2019 challenge (see https://challenge.isic-archive.com/landing/2019/).

The "ISIC 2019: Training" is the aggregate of the following datasets:

BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006 ; https://arxiv.org/abs/1902.03368

In order to extract origins of the images in the HAM10000 Dataset, we do make use of its metadata file that can be found here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T


Please run the following steps:

1)
To import the ISIC 2019 training data and back out the datacenter attribution of images, run:
python download_ISIC_2019_raw_data.py

2)
To perform the train/test split (both stratified by center and pooled)), run:
python folds.py
This step displays basic statistics about the dataset.

3)
To preprocess and resize images, create a directory for the preprocessed images:
mkdir ../ISIC_2019_Training_Input_preprocessed
and run:
python resize_images.py --input_folder ../ISIC_2019_Training_Input --output_folder ../ISIC_2019_Training_Input_preprocessed --sz 224 --cc --pad_resize True

This will resize all images squares of size 224px by 224px. Color constancy is added in the preprocessing.

4)
To train and evaluate a classification model for the pooled dataset, run:
python train2.py
