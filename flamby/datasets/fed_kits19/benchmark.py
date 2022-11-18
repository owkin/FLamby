import argparse
import copy
import os
import time

import numpy as np
import torch
from torch.optim import lr_scheduler

from flamby.datasets.fed_kits19 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedKits19,
    evaluate_dice_on_tests,
    metric,
    softmax_helper,
)
from flamby.utils import check_dataset_from_config


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
    model : torch model that scored the best test accuracy
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)
    # To draw loss and accuracy plots
    training_loss_list = []
    training_dice_list = []
    print(" Train Data Size " + str(dataset_sizes["train"]))
    print(" Test Data Size " + str(dataset_sizes["test"]))
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        dice_list = []
        running_loss = 0.0
        dice_score = 0.0
        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = lossfunc(outputs, labels)

                    # backward + optimize only if in training phase, record training loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)

                    # if test: record dice
                    if phase == "test":
                        preds_softmax = softmax_helper(outputs)
                        preds = preds_softmax.argmax(1)
                        dice_score = metric(preds.cpu(), labels.cpu())
                        dice_list.append(dice_score)

            # if phase == "train":
            #     scheduler.step(epoch)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = np.mean(dice_list)  # average dice

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            "Training Loss: {:.4f} Validation Acc: {:.4f} ".format(epoch_loss, epoch_acc)
        )
        training_loss_list.append(epoch_loss)
        training_dice_list.append(epoch_acc)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best test Balanced acc: {:4f}".format(best_acc))
    print("----- Training Loss ---------")
    print(training_loss_list)
    print("------Validation Accuracy ------")
    print(training_dice_list)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    torch.use_deterministic_algorithms(False)

    check_dataset_from_config(dataset_name="fed_kits19", debug=False)

    train_dataset = FedKits19(train=True, pooled=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers
    )
    test_dataset = FedKits19(train=False, pooled=True)
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
    # data = torch.full([2, 1, 64, 192, 192], 0, dtype=torch.float32)

    model = model.to(device)
    # data = data.to(device)
    # model(data)
    # print('forward pass worked')
    # exit()

    lossfunc = BaselineLoss()

    optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=3e-5, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=30,
        verbose=True,
        threshold=1e-3,
        threshold_mode="abs",
    )

    # TODO: Add 5 seeds
    torch.manual_seed(args.seed)
    # TODO: Test train model on lambda 5 (Preprocessing running)
    model = train_model(
        model,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        device,
        lossfunc,
        args.epochs,  # for easier debug
    )
    print("----- Test Accuracy ----------")
    print(evaluate_dice_on_tests(model, [test_dataloader], metric, use_gpu=True))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU", type=int, default=0, help="GPU to run the training on (if available)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Numbers of workers for the dataloader"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS_POOLED, help="Numbers of Epochs"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    args = parser.parse_args()

    main(args)
