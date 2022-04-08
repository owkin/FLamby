The dataset used in this repo comes from the [ISIC2019 challenge](https://challenge.isic-archive.com/landing/2019/) and the [HAM1000 database](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
We do not own the copyright of the data, everyone using those datasets should abide by their licences (see below) and give proper attribution to the original authors.

## Dataset description

|                   | Dataset description
| ----------------- | -----------------------------------------------------------------------------------------------
| Description       | Dataset from the ISIC 2019 challenge, we keep images for which the datacenter can be extracted.
| Dataset           | 23,247 images of skin lesions
| Centers           | 6 centers (BCN, HAM_vidir_molemax, HAM_vidir_modern, HAM_rosendahl, MSK, HAM_vienna_dias)
| Task              | Multiclass image classification


The "ISIC 2019: Training" is the aggregate of the following datasets:

BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161)

MSK Dataset: (c) Anonymous; [challenge 2017](https://arxiv.org/abs/1710.05006); [challenge 2018](https://arxiv.org/abs/1902.03368)

See below the full citations:
[1] Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)
[2] Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.
[3] Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: “BCN20000: Dermoscopic Lesions in the Wild”, 2019; arXiv:1908.02288.

Find attached the link to the [full licence](https://creativecommons.org/licenses/by-nc/4.0/)


In order to extract the origins of the images in the HAM10000 Dataset (cited above), we store in this repository a copy of [the original HAM10000 metadata file](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
Find attached the link to the [full licence and dataset terms](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T&version=3.0&selectTab=termsTab)


Thank you to [Aman Arora](https://github.com/amaarora) for his [implementation](https://github.com/amaarora/melonama) and [blog](https://amaarora.github.io/2020/08/23/siimisic.html) that we used as a base for our own code.


Please first accept the licences on the HAM10000 and ISIC2019 dataset pages and run the following steps.

## Data
To download the ISIC 2019 training data and extract the original datacenter information for each image, run:
```
python download_isic.py --output-folder /path/to/user/folder
```
The file train_test_folds contains the train/test split of the images (stratified by center).

## Image preprocessing
To preprocess and resize images, run:
```
python resize_images.py
```
This script will resize all images so that the shorter edge of the resized image is 224px and the aspect ratio of the inpit image is maintained.
[Color constancy](https://en.wikipedia.org/wiki/Color_constancy) is added in the preprocessing.

## Baseline training and evaluation in a pooled setting
To train and evaluate a classification model for the pooled dataset, run:
```
python benchmark.py --GPU 0 --batch 64 --workers 4
```
