# Heart Disease

The Heart Disease dataset [1] was collected in 1988 in four centers:
Cleveland, Hungary, Switzerland and Long Beach V. We do not own the
copyright of the data: everyone using this dataset should abide by its
licence and give proper attribution to the original authors. It is
available for download
[here](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease).

## Dataset description
Please refer to the [dataset website](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease)
for an exhaustive data sheet. The table below provides a high-level description
of the dataset.

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Heart Disease dataset.
| Dataset size       | 39,6 KB.
| Centers            | 4 centers - Cleveland, Hungary, Switzerland and Long Beach V.
| Records per center | Train/Test: 199/104, 172/89, 30/16, 85/45.
| Inputs shape       | 16 features (tabular data).
| Total nb of points | 740.
| Task               | Binary classification

### License and data usage terms
This dataset is licensed under a Creative Commons Attribution
4.0 International (**CC-BY 4.0**) license by its authors.
*Anyone using this dataset should abide by its*
*licence and give proper attribution to the original authors.*

### Ethics
As per the [dataset website](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease),
sensitive entries of the dataset were removed by the original authors:

> The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

## Download and preprocessing instructions

To download the data,
First cd into the `dataset_creation_scripts` folder:
```bash
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
```
then simply run the following command:
```
python download.py --output-folder ./heart_disease_dataset
```
This will download 38.6ko of data.

**IMPORTANT :** If you choose to relocate the dataset after downloading it, it is
imperative that you run the following script otherwise all subsequent scripts will not find it:
```
python update_config.py --new-path /new/path/towards/dataset
```

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_heart_disease import FedHeartDisease, HeartDiseaseRaw

# To load the first center as a pytorch dataset
center0 = FedHeartDisease(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedHeartDisease(center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()
```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)

## Benchmarking the baseline on a pooled setting

In order to benchmark the baseline on the pooled dataset one need to download and preprocess the dataset and launch the following script:
```
python benchmark.py
```
This will train a logistic regression classifier (which is the strongest baseline according to [UCI ML Repository](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease).


## References

[1] Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, Detrano,
Robert & M.D., M.D.. (1988). Heart Disease. UCI Machine Learning
Repository.
