import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader as dl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flamby.datasets.fed_ixi import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    SEEDS,
    Baseline,
    BaselineLoss,
    FedIXITiny,
    Optimizer,
    metric,
)
from flamby.utils import evaluate_model_on_tests


def main(num_workers_torch, use_gpu=True, gpu_id=0, log=False):
    """
    Train a UNet for brain mask segmentation on IXI-Tiny by maximizing the DICE
    coefficient.
    Parameters
    ----------
    num_workers_torch: int
        Number of workers to use for batching
    use_gpu: bool, default = True
        Whether to use a GPU (highly advised)
    gpu_id: int, default = 0
        PCI bus id of the GPU to use
    log: bool, default = False
        Whether to log the train loss for tensorboard
    """

    # Set environment variables for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    training_dl = dl(
        FedIXITiny(train=True, pooled=True),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_dl = dl(
        FedIXITiny(train=False, pooled=True),
        num_workers=num_workers_torch,
        batch_size=1,  # Do not change this as it would mess up DICE evaluation
        shuffle=False,
    )

    print(f"The pooled training set contains {len(training_dl.dataset)} T1 images.")
    print(f"The pooled test set contains {len(test_dl.dataset)} T1 images.")

    # Compute the number of batches per epoch
    num_local_steps_per_epoch = len(training_dl.dataset) // BATCH_SIZE
    num_local_steps_per_epoch += int(
        (len(training_dl.dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0
    )

    results = []

    for seed in SEEDS:
        # At each new seed we re-initialize the model
        # and training_dl is shuffled as well
        torch.manual_seed(seed)

        m = Baseline(normalization=None)
        # Transfer to GPU if possible
        if torch.cuda.is_available() and use_gpu:
            m = m.cuda()

        loss = BaselineLoss()
        optimizer = Optimizer(m.parameters(), lr=LR)

        if log:
            # Create one SummaryWriter for each seed in order to overlay the plots
            writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

        for e in tqdm(range(NUM_EPOCHS_POOLED)):
            m.train()
            tot_loss = 0
            for X, y in training_dl:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                y_pred = m(X)
                lm = loss(y_pred, y)
                lm.backward()
                optimizer.step()

                tot_loss += lm.item()

            print(f"epoch {e} avg loss: {tot_loss / num_local_steps_per_epoch:.2e}")

            if log:
                writer.add_scalar(
                    "Loss/train/client", tot_loss / num_local_steps_per_epoch, e
                )

        # Finally, evaluate DICE
        current_results_dict = evaluate_model_on_tests(
            m, [test_dl], metric, use_gpu=use_gpu
        )
        print(current_results_dict)

        results.append(current_results_dict["client_test_0"])

    results = np.array(results)

    if log:
        for i in range(results.shape[0]):
            writer = SummaryWriter(log_dir=f"./runs/tests_seed{SEEDS[i]}")
            writer.add_scalar("DICE coefficient", results[i], 0)

    print("Benchmark Results on pooled IXI:")
    print(f"mDICE on {len(SEEDS)} runs: {results.mean():.2%} \\pm {results.std():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers-torch",
        type=int,
        help="How many workers to use for the batching.",
        default=1,
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="PCI Bus id of the GPU to use."
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Deactivate the GPU to perform all computations on CPU only.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Whether to activate tensorboard logging or not. Default is no logging",
    )

    args = parser.parse_args()

    main(
        args.num_workers_torch,
        log=args.log,
        use_gpu=not args.cpu_only,
        gpu_id=args.gpu_id,
    )
