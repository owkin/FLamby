import argparse
import copy
import os
import random
import time

import albumentations
import dataset
import torch
from sklearn import metrics
from tqdm import tqdm
from flamby.datasets.fed_kits19.dataset import FedKiTS19
from torch import nn
import numpy as np

from flamby.utils import check_dataset_from_config, evaluate_model_on_tests
from flamby.datasets.fed_kits19.model import Baseline
from flamby.datasets.fed_kits19.loss import BaselineLoss
from flamby.datasets.fed_kits19.metric import metric, softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from torch.optim import lr_scheduler



def evaluate_model_on_tests(model, test_dataloaders, metric, use_gpu=True):
    """This function takes a pytorch model and evaluate it on a list of\
    dataloaders using the provided metric function.
    Parameters
    ----------
    model: torch.nn.Module,
        A trained model that can forward the test_dataloaders outputs
    test_dataloaders: List[torch.utils.data.DataLoader]
        A list of torch dataloaders
    metric: callable,
        A function with the following signature:\
            (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
    use_gpu: bool, optional,
        Whether or not to perform computations on GPU if available. \
        Defaults to True.
    Returns
    -------
    dict
        A dictionnary with keys client_test_{0} to \
        client_test_{len(test_dataloaders) - 1} and associated scalar metrics \
        as leaves.
    """
    results_dict = {}
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            composite_dice_list = []
            tumor_dice_list = []
            kidney_dice_list = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X).detach().cpu()
                y = y.detach().cpu()
                composite_dice, kidney_dice, tumor_dice = metric(y_pred, y)
                composite_dice_list.append(composite_dice)
                tumor_dice_list.append(tumor_dice)
                kidney_dice_list.append(kidney_dice)

            results_dict[f"client_test_{i}"] = {'Composite dice': np.mean(composite_dice_list), 'Kidney Dice ': np.mean(kidney_dice_list), 'Tumor Dice ': np.mean(tumor_dice_list)}
    return results_dict


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

    composite_dice_epoch_list = []
    tumor_dice_epoch_list = []
    kidney_dice_epoch_list = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        composite_dice_list = []
        tumor_dice_list = []
        kidney_dice_list = []

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            # y_true = []
            # y_pred = []

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)
                # y_true.append(sample[1])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = lossfunc(outputs, labels)
                    print('loss '+str(loss))
                    preds_softmax = softmax_helper(outputs)
                    preds = preds_softmax.argmax(1)
                    # y_pred.append(preds)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()


                composite_dice, kidney_dice, tumor_dice = metric(preds.cpu(), labels.cpu())
                composite_dice_list.append(composite_dice)
                tumor_dice_list.append(tumor_dice)
                kidney_dice_list.append(kidney_dice)


                # TODO: double check these statistics definitions (esp epoch acc and epoch balanced acc)
                running_loss += loss.item() * inputs.size(0)
                epoch_acc = (np.mean(composite_dice_list) + np.mean(tumor_dice_list))/2
                break
            composite_dice_epoch_list.append(np.mean(composite_dice_list))
            tumor_dice_epoch_list.append(np.mean(tumor_dice_list))
            kidney_dice_epoch_list.append(np.mean(kidney_dice_list))
                # break
            if phase == "train":
                scheduler.step(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_balanced_acc = (np.mean(kidney_dice_epoch_list) + np.mean(tumor_dice_epoch_list))/2

            print(
                "{} Loss: {:.4f} Acc: {:.4f} Balanced acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc, epoch_balanced_acc
                )
            )

            # deep copy the model
            if phase == "test" and epoch_balanced_acc > best_acc:
                best_acc = epoch_balanced_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best test Balanced acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    torch.use_deterministic_algorithms(False)


    dict = check_dataset_from_config(dataset_name="fed_kits19", debug=False)
    input_path = dict["dataset_path"]
    dic = {"model_dest": os.path.join(input_path, "saved_model_state_dict")}

    train_dataset = FedKiTS19(train=True, pooled=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.workers
    )
    test_dataset = FedKiTS19(train=False, pooled=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
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
    lossfunc = BaselineLoss()

    #add args for the following params,
    lr_scheduler_eps = 1e-3
    lr_scheduler_patience = 30
    initial_lr = 3e-4
    weight_decay = 3e-5
    num_epochs = 2
    optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=weight_decay,
                                      amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                       patience=lr_scheduler_patience,
                                                       verbose=True, threshold=lr_scheduler_eps,
                                                       threshold_mode="abs")

    #TODO: Test train model on lambda machines
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

    print(evaluate_model_on_tests(model, [test_dataloader], metric, use_gpu=True))



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
        default=1,
        help="Numbers of workers for the dataloader",
    )
    #TODO: add other args as well
    args = parser.parse_args()

    main(args)


