import os

import torch


def use_gpu_idx(idx, cpu_only=False):
    """Small util that put computations on the chosen GPU.

    Parameters
    ----------
    idx : int
        The GPU to use.
    cpu_only : bool, optional
        Whether to force comptations to be on CPU even in the presence of GPUS,
        by default False

    Returns
    -------
    bool
        Whether we will be using GPU or not.
    """
    gpu_detected = torch.cuda.is_available()
    if not (gpu_detected) or cpu_only:
        return False
    else:
        n_gpus = torch.cuda.device_count()
        assert idx < n_gpus, f"You chose GPU {idx} that does not exist."
        # We use environment variables to manage GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        return True
