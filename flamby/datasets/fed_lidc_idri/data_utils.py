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
        assert algo in ["random", "all"], "Unsupported sampling algorithm."
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
        if self.algo == "all":
            return all_sampler(X, y, patch_shape=(48, 48, 48), overlap=0.0)


def resize_by_crop_or_pad(X, output_shape=(384, 384, 384)):
    """
    Resizes by padding or cropping centered on the image.
    Works with whatever number of dimensions.
    Parameters
    ----------
    X : torch.Tensor
        Tensor to reshape.
    output_shape : Tuplet or None
        Desired shape. If None, returns tensor as is.
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

    output_shape = torch.tensor(output_shape)
    input_shape = torch.tensor(X.shape)

    # Pad missing dimensions
    missing = torch.clamp(output_shape - input_shape, min=0)
    pad_begin = missing.div(2, rounding_mode="floor")
    pad_last = torch.maximum(output_shape, input_shape) - pad_begin - input_shape
    padding = torch.stack([pad_begin, pad_last], dim=-1).flatten()
    X = F.pad(X, tuple(padding)[::-1])

    # Crop extra
    extra = torch.clamp(torch.tensor(X.shape) - output_shape, min=0).div(
        2, rounding_mode="floor"
    )

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

    maxvals = torch.clamp(torch.tensor(image.shape) - torch.tensor(patch_shape), min=1)

    begins = (
        torch.rand((n_samples, image.dim())) * torch.max(maxvals)
    ).long() % maxvals[None, ...]

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


def all_sampler(
    X,
    y,
    patch_shape=(48, 48, 48),
    overlap=0.0,
):
    """
    Returns all patches of desired shape from image and mask. To be used for inference.
    Parameters
    ----------
    X : torch.Tensor
        Input voxel image
    y  : torch.Tensor
        Nodule mask
    patch_shape : tuple, optional
        Desired shape for extracted patches, channels excluded
    overlap : float, optional
        Fraction of overlap between two patches.
        If 0., each pixel is sampled exactly once.
    Returns
    -------
    Two torch.Tensor
        of shape (n_patches, *desired_shape)
    """
    native_shape = patch_shape
    patch_shape = torch.tensor(patch_shape).long()

    assert type(overlap) == float

    image_shape = X.shape
    centroids = get_all_centroids(image_shape, native_shape, overlap)

    centroids = centroids.reshape(centroids.shape[0], 1, 1, 1, centroids.shape[-1])

    ranges = [
        torch.arange(-patch_shape[i].item() // 2, patch_shape[i].item() // 2)
        for i in range(len(native_shape))
    ]

    paddings = patch_shape
    paddings = paddings.long()

    X = F.pad(
        X,
        tuple(torch.stack([paddings, paddings], dim=-1).flatten().tolist()),
        mode="constant",
    )

    y = F.pad(
        y,
        tuple(torch.stack([paddings, paddings], dim=-1).flatten().tolist()),
        mode="constant",
    )

    indices = (
        centroids
        + torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)[None, ...]
    )

    indices += patch_shape
    # Dirty hack to mimic tf.gather_nd
    indices_list = [indices[..., i] for i in range(indices.shape[-1])]

    image_patches = X[indices_list]
    label_patches = X[indices_list]
    return image_patches, label_patches


def get_all_centroids(image_shape, patch_shape, overlap):
    """
    Compute coordinates of patch centers.
    Parameters
    ----------
    image_shape : Tuple of ints
        Shape of original image
    patch_shape : Tuple of ints
        Desired patch shape
    overlap : float
        Fraction of overlap between two patches.
    Returns
    -------
    torch.Tensor
        of patch centers
    """

    offset = (np.floor(overlap / 2) * torch.tensor(patch_shape)).long()

    centroids = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(
                    patch_shape[i] // 2 - offset[i],
                    image_shape[i] + patch_shape[i] // 2,
                    int((1 - overlap) * patch_shape[i]),
                    dtype=torch.long,
                )
                for i in range(len(patch_shape))
            ],
            indexing="ij"
        ),
        dim=-1,
    )
    centroids = torch.reshape(centroids, (-1, len(patch_shape)))

    return centroids
