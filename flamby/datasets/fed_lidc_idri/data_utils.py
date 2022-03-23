import numpy as np
import torch
import torch.nn.functional as F


def resize_by_crop_or_pad(X, output_shape=(384, 384, 384)):
    """
    Resizes by padding or cropping centered on the image.
    Works with whatever number of dimensions.
    Parameters
    ----------
    X : torch.Tensor
        Tensor to reshape.
    output_shape : Tuplet or None
        Desired shape, n_channels included. If None, returns tensor as is.
    Returns
    -------
    torch.Tensor
        The resized tensor
    Raises
    ------
    ValueError
        If dimension mismatch
    """

    if output_shape is None:
        return X

    output_shape = torch.Size(output_shape)
    input_shape = X.shape

    # Pad missing dimensions
    missing = np.maximum(np.subtract(output_shape, input_shape), 0)
    pad_begin = missing // 2
    pad_last = np.maximum(output_shape, input_shape) - pad_begin - input_shape
    padding = np.stack([pad_begin, pad_last], axis=-1).flatten()
    X = F.pad(X, tuple(padding[::-1]))

    # Crop extra
    extra = np.maximum(np.subtract(X.shape, output_shape), 0) // 2
    # Is there a way to vectorize this?
    for d, start in enumerate(extra):
        X = X.narrow(d, start, output_shape[d])

    return X
