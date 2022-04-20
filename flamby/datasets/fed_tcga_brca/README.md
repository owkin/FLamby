# TCGA BRCA
The dataset used in this repo comes from [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). We selected one single cancer type: Breast Invasive Carcinoma (BRCA) and only use clinical tabular data. We use data preprocessed by the authors of this [article](https://arxiv.org/pdf/2006.08997.pdf) i.e. a subset of the features in the raw TCGA-BRCA dataset, one-hot encoding the categorical variables.


## Dataset description

|                   | Dataset description
| ----------------- | -----------------------------------------------------------------------------------------------
| Description       | Clinical data from the TCGA-BRCA study.
| Dataset           | 1,088 patients, each patient being represented by a vector of dimension 39.
| Centers           | 6 regions (Northeast, South, West, Midwest, Europe, Canada)
| Task              | Survival analysis


Raw TCGA-BRCA data can be viewed and downloaded [here](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).

## Data
Preprocessed data is stored in this repo in the file brca.csv. The medical centers (with their geographic regions) providing the data are stored in the file centers.csv. From this file and the patients' TCGA barcodes, we can extract the region of origin of each patient's tissue sort site (TSS). The numbers of sites being too large (64) we regroup them in 6 different regions (Northeast, South, West, Midwest, Europe, Canada). The patients' stratified split by region is static and available in the train_test_split.csv file.

## Baseline training and evaluation in a pooled setting
To train and evaluate a model for the pooled dataset, run:
```
python benchmark.py --GPU 0 --workers 4
```
