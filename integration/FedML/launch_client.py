import fedml
from fedml import FedMLRunner
from fedml_utils import HeartDiseaseAggregator, HeartDiseaseTrainer
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_heart_disease import BATCH_SIZE, Baseline, FedHeartDisease

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = "cpu"
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    # The number of classes
    nc = 2
    # No need for data for the server
    if args.role == "server":
        # load data
        client_idx = int(args.process_id) - 1

        # We create the traditional FLamby datasets
        train_dataset = FedHeartDisease(center=client_idx, train=True)
        train_dataloader = dl(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10
        )
        test_dataset = FedHeartDisease(center=client_idx, train=False)
        test_dataloader = dl(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            drop_last=True,
        )

        data_local_num_dict[client_idx] = len(train_dataset)
        train_data_local_dict[client_idx] = train_dataloader
        test_data_local_dict[client_idx] = test_dataloader
        dataset = (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            nc,
        )

    # create model and trainer
    model = Baseline()
    trainer = HeartDiseaseTrainer(model=model, args=args)
    aggregator = HeartDiseaseAggregator(model=model, args=args)
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
