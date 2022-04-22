import argparse
import os
import time
from pathlib import Path

import torch

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


def train_model(
    model, optimizer, scheduler, dataloaders, dataset_sizes, device, lossfunc, num_epochs
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

    # checking loss and metric before training starts
    for phase in ["train", "test"]:
        model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        y_true = []
        y_pred = []
        for sample in dataloaders[phase]:
            inputs = sample[0].to(device)
            labels = sample[1].to(device)
            y_true.append(sample[1])
            with torch.no_grad():
                outputs = model(inputs)
                loss = lossfunc(outputs, labels)
                y_pred.append(outputs)
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_sizes[phase]
        y = torch.cat(y_true)
        y_hat = torch.cat(y_pred)
        epoch_c_index = metric(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        print(
            "Initial {} Loss: {:.4f} c-index: {:.4f}".format(
                phase, epoch_loss, epoch_c_index
            )
        )

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
                inputs = sample[0].to(device)
                labels = sample[1].to(device)
                y_true.append(sample[1])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = lossfunc(outputs, labels)
                    y_pred.append(outputs)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

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

    print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print()

    return model


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

    model = Baseline()
    model = model.to(device)

    lossfunc = BaselineLoss

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
    )

    num_epochs = NUM_EPOCHS_POOLED

    model = train_model(
        model,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        device,
        lossfunc,
        num_epochs,
    )

    ppath = Path(os.path.realpath(__file__)).parent.resolve()
    dic = {"model_dest": os.path.join(ppath, "saved_model_state_dict")}
    dest_file = dic["model_dest"]
    torch.save(model.state_dict(), dest_file)


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

    print("Loading the saved model and running flamby.utils.evaluate_model_on_tests")

    test_dataset = FedTcgaBrca(train=False, pooled=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    model = Baseline()
    ppath = Path(os.path.realpath(__file__)).parent.resolve()
    dic = {"model_dest": os.path.join(ppath, "saved_model_state_dict")}
    model.load_state_dict(torch.load(dic["model_dest"]))
    model.eval()

    torch.use_deterministic_algorithms(False)
    print(evaluate_model_on_tests(model, [test_dataloader], metric, use_gpu=True))
