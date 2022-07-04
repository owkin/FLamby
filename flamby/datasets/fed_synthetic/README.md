## Synthetic Data

The synthetic dataset can be either a regression or a classification
task. Number of centers, samples, features, their repartition and
heterogeneity can be controlled in depth. See the documentation of
`generate_synthetic_dataset` or the full paper for details on the
generation process.

## Dataset description

|                   | Dataset description
| ----------------- | -----------------------------------------------
| Description       | Synthetic Dataset
| Dataset           | Arbitrary number of centers, samples and features
| Centers           | N/A
| Task              | Classification of tabular data


## Using the dataset

To generate the dataset, use the following command. See helper of the script to
control each parameter of the dataset (number of centers, samples, features,
data heterogeneity...)
```
python download.py --output-folder ./synthetic
```

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_heart_disease import FedSynthetic, SyntheticRaw

# To load the first center as a pytorch dataset
center0 = FedSynthetic(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedSynthetic(center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl


X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()

```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)


## Benchmarking the baseline on a pooled setting

The benchmark is designed to benchmark a logistic regression on a
classification dataset with two classes. To generate the relevant
dataset, run
```
python download.py --classification --clusters 2 --output-folder synthetic
```

To run the benchmark on the pooled dataset one need to generate the
dataset and launch the following script:
```
python benchmark.py
```
This will train a logistic regression classifier.
