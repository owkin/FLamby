## Synthetic Data

The synthetic dataset is

## Dataset description

|                   | Dataset description
| ----------------- | -----------------------------------------------
| Description       | Synthetic Dataset
| Dataset           | Arbitrary number of centers, samples and features
| Centers           | N/A
| Task              | Classification of tabular data


## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_heart_disease import FedHeartDisease, HeartDiseaseRaw

# To load the first center
center0 = FedSynthetic(center=0, train=True)
# To load the second center
center1 = FedSynthetic(center=1, train=True)
```

## Benchmarking the baseline on a pooled setting

In order to benchmark the baseline on the pooled dataset one need to download and preprocess the dataset and launch the following script:
```
python benchmark.py
```
This will train a logistic regression classifier.


## References
