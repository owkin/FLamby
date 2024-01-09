import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as dl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flamby.datasets.fed_camelyon16 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedCamelyon16,
    Optimizer,
    collate_fn,
    metric,
)
from flamby.utils import evaluate_model_on_tests


def main(num_workers_torch, log=False, log_period=10, debug=False, cpu_only=False):
    """Function to execute the benchmark on Camelyon16.

    Parameters
    ----------
    num_workers_torch : int
        The number of parallel workers for torch
    log : bool
        Whether to activate tensorboard logging. Default to False.
    log_period : int
        The period between two logging of parameters in batches. Defaults to 10.
    debug : bool
        Whether or not to use the dataset obtained in debug mode. Default to False.
    cpu_only : bool
        Whether to disable the use of GPU. Defaults to False.
    """
    metrics_dict = {"AUC": metric}
    use_gpu = torch.cuda.is_available() and not (cpu_only)
    training_set = FedCamelyon16(train=True, pooled=True, debug=debug)
    # extract feature dimension used
    features_dimension = training_set[0][0].size(1)
    training_dl = dl(
        training_set,
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
    )
    test_dl = dl(
        FedCamelyon16(train=False, pooled=True, debug=debug),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )
    print(f"The training set pooled contains {len(training_dl.dataset)} slides")
    print(f"The test set pooled contains {len(test_dl.dataset)} slides")

    if log:
        # We compute the number of batches per epoch
        num_local_steps_per_epoch = len(training_dl.dataset) // BATCH_SIZE
        num_local_steps_per_epoch += int(
            (len(training_dl.dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0
        )

    results = []
    seeds = np.arange(42, 47).tolist()
    for seed in seeds:
        # At each new seed we re-initialize the model
        # and training_dl is shuffled as well
        torch.manual_seed(seed)
        m = Baseline(features_dimension)
        # We put the model on GPU whenever it is possible
        if use_gpu:
            m = m.cuda()
        loss = BaselineLoss()
        optimizer = Optimizer(m.parameters(), lr=LR)
        if log:
            # We create one summarywriter for each seed in order to overlay the plots
            writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

        for e in tqdm(range(NUM_EPOCHS_POOLED)):
            if log:
                # At each epoch we look at the histograms of all the network's parameters
                for name, p in m.named_parameters():
                    writer.add_histogram(f"client_0/{name}", p, e)
            for s, (X, y) in enumerate(training_dl):
                # traditional training loop with optional GPU transfer
                if use_gpu:
                    X = X.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                y_pred = m(X)
                lm = loss(y_pred, y)
                lm.backward()
                optimizer.step()
                if log:
                    current_step = s + num_local_steps_per_epoch * e
                    if (current_step % log_period) == 0:
                        writer.add_scalar(
                            "Loss/train/client",
                            lm.item(),
                            s + num_local_steps_per_epoch * e,
                        )
                        for k, v in metrics_dict.items():
                            train_batch_metric = v(
                                y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
                            )
                            writer.add_scalar(
                                f"{k}/train/client",
                                train_batch_metric,
                                s + num_local_steps_per_epoch * e,
                            )

        current_results_dict = evaluate_model_on_tests(
            m, [test_dl], metric, use_gpu=use_gpu
        )
        print(current_results_dict)
        results.append(current_results_dict["client_test_0"])

    results = np.array(results)

    if log:
        for i in range(results.shape[0]):
            writer = SummaryWriter(log_dir=f"./runs/tests_seed{seeds[i]}")
            writer.add_scalar("AUC-test", results[i], 0)

    print("Benchmark Results on Camelyon16 pooled:")
    print(f"mAUC on 5 runs: {results.mean(): .2%} \\pm {results.std(): .2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers-torch",
        type=int,
        help="How many workers to use for the batching.",
        default=20,
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Whether to activate tensorboard logging or not default to no logging",
    )
    parser.add_argument(
        "--log-period",
        type=int,
        help="The period in batches for the logging of metric and loss",
        default=10,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to use the dataset obtained in debug mode.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Deactivate the GPU to perform all computations on CPU only.",
    )

    args = parser.parse_args()
    main(args.num_workers_torch, args.log, args.log_period, args.debug, args.cpu_only)
