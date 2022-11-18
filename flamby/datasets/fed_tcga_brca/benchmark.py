import argparse
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.utils import evaluate_model_on_tests


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def train_model(
    model,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    lossfunc,
    num_epochs,
    seed,
    log,
    log_period,
):
    """Training function
    Parameters
    ----------
    model : torch model to be trained
    optimizer : torch optimizer used for training
    scheduler : torch scheduler used for training
    dataloaders : dictionary {"train": train_dataloader, "test": test_dataloader}
    dataset_sizes : dictionary {"train": len(train_dataset), "test": len(test_dataset)}
    device : device where model parameters are stored
    lossfunc : function, loss function
    num_epochs : int, number of epochs for training
    seed: int, the sint for the training
    log_period: int, the number of batches between two dumps if log is activated.
    Returns
    -------
    tuple(torch.nn.Module, float) : torch model output by training loop and
    cindex on test.
    """

    since = time.time()

    if log:
        writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

    num_local_steps_per_epoch = len(dataloaders["train"].dataset) // BATCH_SIZE
    num_local_steps_per_epoch += int(
        (len(dataloaders["train"].dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0
    )
    model = model.train()
    for epoch in range(0, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        running_loss = 0.0
        y_true = []
        y_pred = []

        # Iterate over data.
        for idx, (X, y) in enumerate(dataloaders["train"]):
            X = X.to(device)
            y = y.to(device)
            y_true.append(y)

            optimizer.zero_grad()
            outputs = model(X)
            y_pred.append(outputs)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()

            current_step = idx + num_local_steps_per_epoch * epoch

            if log and (idx % log_period) == 0:
                writer.add_scalar("Loss/train/client", loss.item(), current_step)

            running_loss += loss.item() * X.size(0)

            scheduler.step()

        epoch_loss = running_loss / dataset_sizes["train"]
        y = torch.cat(y_true)
        y_hat = torch.cat(y_pred)
        epoch_c_index = metric(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        if log:
            writer.add_scalar("Loss/average-per-epoch/client", epoch_loss, epoch)
            writer.add_scalar("C-index/full-training/client", epoch_c_index, epoch)

        print(
            "{} Loss: {:.4f} c-index: {:.4f}".format("train", epoch_loss, epoch_c_index)
        )

    # Iterate over data.
    dict_cindex = evaluate_model_on_tests(model, [dataloaders["test"]], metric)

    if log:
        writer.add_scalar("Test/C-index", dict_cindex["client_test_0"], 0)

    print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print()

    return model, dict_cindex["client_test_0"]


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    torch.use_deterministic_algorithms(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    lossfunc = BaselineLoss()
    num_epochs = NUM_EPOCHS_POOLED
    log = args.log
    log_period = args.log_period

    results0 = []
    results1 = []
    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = FedTcgaBrca(train=True, pooled=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers
        )
        test_dataset = FedTcgaBrca(train=False, pooled=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.workers,
            # drop_last=True,
        )

        dataloaders = {"train": train_dataloader, "test": test_dataloader}
        dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

        model = Baseline()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
        )

        results0.append(evaluate_model_on_tests(model, [test_dataloader], metric))

        model, test_cindex = train_model(
            model,
            optimizer,
            scheduler,
            dataloaders,
            dataset_sizes,
            device,
            lossfunc,
            num_epochs,
            seed,
            log,
            log_period,
        )
        results1.append(test_cindex)

    print("Before training")
    print("Test C-index ", results0)
    print("Average test C-index ", dict_mean(results0))
    print("After training")
    print("Test C-index ", results1)
    print("Average test C-index ", sum(results1) / len(results1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU", type=int, default=0, help="GPU to run the training on (if available)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Numbers of workers for the dataloader"
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether or not to dump tensorboard events."
    )
    parser.add_argument(
        "--log-period",
        type=int,
        help="The period in batches for the logging of metric and loss",
        default=10,
    )
    args = parser.parse_args()

    main(args)
