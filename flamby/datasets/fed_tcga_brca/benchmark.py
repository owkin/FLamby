import argparse
import os
import time

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
    num_epochs : int, numuber of epochs for training
    Returns
    -------
    model : torch model output by training loop
    """

    since = time.time()

    if log:
        writer = SummaryWriter(log_dir=f"./runs/seed{seed}")
        batch = 0

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            y_true = []
            y_pred = []

            # Iterate over data.
            for sample in dataloaders[phase]:
                X = sample[0].to(device)
                y = sample[1].to(device)
                y_true.append(sample[1])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(X)
                    loss = lossfunc(outputs, y)
                    y_pred.append(outputs)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                if log & (phase == "train"):
                    print(batch, loss.item())
                    writer.add_scalar("Loss", loss.item(), batch)
                    batch += 1
                running_loss += loss.item() * X.size(0)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            y = torch.cat(y_true)
            y_hat = torch.cat(y_pred)
            epoch_c_index = metric(
                y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()
            )
            print(
                "{} Loss: {:.4f} c-index: {:.4f}".format(
                    phase, epoch_loss, epoch_c_index
                )
            )
            if (phase == "test") & (epoch == num_epochs):
                test_loss = epoch_loss
                test_cindex = epoch_c_index

    print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print()

    return model, test_loss, test_cindex


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    torch.use_deterministic_algorithms(False)

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
        drop_last=True,
    )

    dataloaders = {"train": train_dataloader, "test": test_dataloader}
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    lossfunc = BaselineLoss()
    num_epochs = NUM_EPOCHS_POOLED
    log = True

    results1 = []
    results2 = []
    for seed in range(5):
        torch.manual_seed(seed)
        model = Baseline()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
        )
        model, test_loss, test_cindex = train_model(
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
        )
        results1.append(test_loss)
        results2.append(test_cindex)

    print("Test loss ", sum(results1) / len(results1))
    print("Test c-index ", sum(results2) / len(results2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU",
        type=int,
        default=0,
        help="GPU to run the training on (if available)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Numbers of workers for the dataloader",
    )
    args = parser.parse_args()

    main(args)
