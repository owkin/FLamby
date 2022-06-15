Quickstart
##########

The Fed-TCGA-BRCA dataset requires no downloading or preprocessing.

To train and evaluate a model on the Fed-TCGA-BRCA dataset in a federated way using the FedAvg strategy, run:
```
cd ./flamby
python fed_benchmark.py --strategy FedAvg --config-file-path ./tcga_brca_config.json --results-file-path ./quickstart_results.csv --workers 0
```

You can visualize the results of your experiments in the file `./flamby/quickstart_results.csv`.

