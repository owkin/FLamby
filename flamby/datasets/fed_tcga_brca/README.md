# TCGA-BRCA
The dataset used in this repo comes from [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) from [the GDC data portal](https://portal.gdc.cancer.gov/).

We selected one single cancer type: Breast Invasive Carcinoma (BRCA) and only use clinical tabular data. We replicate the preprocessing used by [Andreux et al.](https://arxiv.org/pdf/2006.08997.pdf) from data originally computed from TCGA by [Liu et al.](https://pubmed.ncbi.nlm.nih.gov/29625055/):

Liu J, Lichtenberg T, Hoadley KA, Poisson LM, Lazar AJ, Cherniack AD, Kovatich AJ, Benz CC, Levine DA, Lee AV, Omberg L, Wolf DM, Shriver CD, Thorsson V; Cancer Genome Atlas Research Network, Hu H. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell. 2018 Apr 5;173(2):400-416.e11. doi: 10.1016/j.cell.2018.02.052. PMID: 29625055; PMCID: PMC6066282.

Andreux, M., Manoel, A., Menuet, R., Saillard, C., and Simpson, C., “Federated Survival Analysis with Discrete-Time Cox Models”, <i>arXiv e-prints</i>, 2020.

i.e. a subset of the features in the raw TCGA-BRCA dataset (categorical variables are one-hot encoded).

## Terms of use
The data terms can be found [here](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).
Note that we only use unrestricted data.
We do not guarantee that the use of this data can be done freely by the user.
As such it is mandatory that one should check the applicability of the licence associated with this data before using it.

In particular, as per the [GDC data access policy](https://gdc.cancer.gov/about-gdc/gdc-policies),
users should
> not attempt to identify individual human research participants from whom the data were obtained.

## Ethics
As per the [TCGA policies](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/history/policies),
special care was devoted to ensure privacy protection of research subjects,
including but not limited to HIPAA compliance.
Note that we do not use the genetic part of TCGA whose access is restricted due to its sensitivity.

In particular, as per the [GDC data access policy](https://gdc.cancer.gov/about-gdc/gdc-policies),
the terms bind users as to "not attempt to identify individual human research participants from whom the data were obtained."

## Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Clinical data from the TCGA-BRCA study with 1,088 patients.
| Dataset size       | 117,5 KB (stored in this repository).
| Centers            | 6 regions - Northeast, South, West, Midwest, Europe, Canada.
| Records per center | Train/Test: 248/63, 156/40, 164/42, 129/33, 129/33, 40/11.
| Inputs shape       | 39 features (tabular data).
| Targets shape      | (E,T). E: relative risk, continuous variable. T: event observed (1) or censorship (0)
| Total nb of points | 1088.
| Task               | Survival analysis.

For a more thorough presentation of data, raw TCGA-BRCA data can be viewed, investigated, and downloaded [here](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).

## Data
Preprocessed data is stored in this repo in the file ```brca.csv```, so the dataset does not need to be downloaded. The medical centers (with their geographic regions) are stored in the file ```centers.csv```. From this file and the patients' TCGA barcodes, we can extract the region of origin of each patient's tissue sort site (TSS). The numbers of sites being too large (64) we regroup them in 6 different regions (Northeast, South, West, Midwest, Europe, Canada). The patients' stratified split by region is static and stored in the train_test_split.csv file.

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_tcga_brca import FedTcgaBrca

# To load the first center as a pytorch dataset
center0 = FedTcgaBrca(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedTcgaBrca(center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()
```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)


## Baseline training and evaluation in a pooled setting
To train and evaluate a model for the pooled dataset, run:
```
python benchmark.py --GPU 0 --workers 4
```
