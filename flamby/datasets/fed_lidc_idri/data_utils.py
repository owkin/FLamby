import numpy as np
import torch
import torch.nn.functional as F


class Sampler(object):
    """
    Attributes
    ----------
    patch_shape : int Tuple
        Desired patch shape
    n_samples : int
        Number of patches to sample per input
    algo : str
        Sampling algorithm. Default = 'random'.
    """

    def __init__(self, patch_shape=(48, 48, 48), n_samples=2, algo="random"):
        """
        Parameters
        ----------
        patch_shape : int Tuple
            Desired patch shape
        n_samples : int
            Number of patches to sample per input
        algo : str
            Sampling algorithm. Default = 'random'.
        """
        self.patch_shape = patch_shape
        self.n_samples = n_samples
        assert algo in ["random"], "Unsupported sampling algorithm."
        self.algo = algo

    def __call__(self, X, y):
        """
        Sample patches from image X and label (mask) y.
        Parameters
        ----------
        X : torch.Tensor
        y : torch.Tensor
        Returns
        -------
        image_patches : torch.Tensor
        label_patches : torch.Tensor
        """
        if self.algo == "random":
            return random_sampler(X, y, self.patch_shape, self.n_samples)


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


def random_sampler(image, label, patch_shape=(48, 48, 48), n_samples=2):
    """
    Sample random patches from input of any dimension
    Parameters
    ----------
    image : torch.Tensor
        input image tensor
    label : torch.Tensor
        label map
    patch_shape : int Tuple
        desired shape of patches
    n_samples : int, optional
        number of output patches
    Returns
    -------
    image_patches : torch.Tensor
        random image patches
    label_patches : torch.Tensor
        random label patches (at same positions as images)
    """
    # Assumes one channel for the image
    maxvals = np.maximum(np.subtract(image.shape, patch_shape), 1)

    begins = (torch.rand((n_samples, image.dim())) * np.max(maxvals)).long() % maxvals[
        None, ...
    ]

    begins = begins[:, None, None, None, :]

    ranges = [torch.arange(s) for s in patch_shape]

    # Build the tensor of indices for further slicing
    indices = (
        begins + torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)[None, ...]
    )

    # Dirty hack to mimic tf.gather_nd
    indices_list = [indices[..., i] for i in range(indices.shape[-1])]
    image_patches = image[indices_list]
    label_patches = label[indices_list]

    return image_patches, label_patches
