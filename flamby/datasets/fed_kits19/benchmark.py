import argparse
import copy
import os
import random
import time

import albumentations
import dataset
import torch
from sklearn import metrics
from flamby.datasets.fed_kits19.dataset import FedKiTS19
from torch import nn

from flamby.utils import check_dataset_from_config, evaluate_model_on_tests
from flamby.datasets.fed_kits19.model import Generic_UNet
from flamby.datasets.fed_kits19.loss import DC_and_CE_loss
from nnunet.network_architecture.initialization import InitWeights_He
from torch.optim import lr_scheduler


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

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
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
                    _, preds = torch.max(outputs, 1)
                    y_pred.append(preds)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            y = torch.cat(y_true)
            y_hat = torch.cat(y_pred)

            epoch_balanced_acc = metrics.balanced_accuracy_score(y.cpu(), y_hat.cpu())

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
        train_dataset, batch_size=2, shuffle=True, num_workers=args.workers
    )
    test_dataset = FedKiTS19(train=False, pooled=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    dataloaders = {"train": train_dataloader, "test": test_dataloader}
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)


    net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    model = Generic_UNet(1, 32, 3, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
        {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False, False, lambda x: x, InitWeights_He(1e-2),
             net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    model = model.to(device)
    lossfunc = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    #add args for the following params,
    lr_scheduler_eps = 1e-3
    lr_scheduler_patience = 30
    initial_lr = 3e-4
    weight_decay = 3e-5
    num_epochs = 1000
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

    script_directory = os.path.abspath(os.path.dirname(__file__))
    dest_file = os.path.join(script_directory, dic["model_dest"])
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
    #TODO: add other arguments as well
    args = parser.parse_args()

    main(args)

    # loading the saved model and running evaluate_model_on_tests
    #
    # sz = 200
    # test_aug = albumentations.Compose(
    #     [
    #         albumentations.CenterCrop(sz, sz),
    #         albumentations.Normalize(always_apply=True),
    #     ]
    # )
    # test_dataset = dataset.FedIsic2019(0, True, "test", augmentations=test_aug)
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     drop_last=True,
    # )
    #
    # model = Baseline()
    # dict = check_dataset_from_config(dataset_name="fed_isic2019", debug=False)
    # input_path = dict["dataset_path"]
    # dic = {"model_dest": os.path.join(input_path, "saved_model_state_dict")}
    # model.load_state_dict(torch.load(dic["model_dest"]))
    # model.eval()
    #
    # torch.use_deterministic_algorithms(False)
    # print(evaluate_model_on_tests(model, [test_dataloader], metric, use_gpu=True))
