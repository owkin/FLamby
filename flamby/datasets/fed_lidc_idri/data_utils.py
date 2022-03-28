import numpy as np
import torch
import torch.nn.functional as F


class Sampler(object):
    """
    Attributes
    ----------
    patch_shape : int Tuple
        Desired patch shape
    n_patches : int
        Number of patches to sample per input
    ratio : float, optional
        Ratio of patches with lesions versus patches without.
    center : boolean, optional
        If False, add some noise to the coordinates of the centroids.
    algo : str
        Sampling algorithm. Default = 'fast'.
    """

    def __init__(
        self,
        patch_shape=(48, 48, 48),
        n_patches=4,
        ratio=0.8,
        center=False,
        algo="fast",
    ):
        """
        Parameters
        ----------
        patch_shape : int Tuple
            Desired patch shape
        n_patches : int
            Number of patches to sample per input
        ratio : float, optional
            Ratio of patches with lesions versus patches without.
            Useful to balance the dataset. Only for fast sampler.
        center : boolean, optional
            If False, add some noise to the coordinates of the centroids.
            Only for fast sampler.
        algo : str
            Sampling algorithm. Default = 'random'.
        """
        self.patch_shape = patch_shape
        self.n_patches = n_patches
        self.ratio = ratio
        self.center = center
        assert algo in [
            "fast",
            "random",
            "all",
            "none",
        ], "Unsupported sampling algorithm."
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
        Two torch.Tensor
            of shape (n_patches, *desired_shape, n_channels)
        """
        if self.algo == "fast":
            return fast_sampler(
                X, y, self.patch_shape, self.n_patches, self.ratio, self.center
            )
        elif self.algo == "random":
            return random_sampler(X, y, self.patch_shape, self.n_patches)
        elif self.algo == "all":
            return all_sampler(X, y, patch_shape=(48, 48, 48), overlap=0.0)
        elif self.algo == "none":
            return X, y


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

    indices_list = build_indices_list(indices)
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
    indices_list = build_indices_list()

    image_patches = X[indices_list]
    label_patches = X[indices_list]
    return image_patches, label_patches


def fast_sampler(
    X,
    y,
    patch_shape=(48, 48, 48),
    n_patches=4,
    ratio=0.8,
    center=False,
):
    """
    Parameters
    ----------
    image : tf.tensor
        Input voxel image
    label : tf.tensor
        Label mask
    patch_shape : tuple, optional
        Desired shape for extracted patches
    n_patches : int, optional
        Number of patches to extract
    ratio : float, optional
        Ratio of patches with lesions versus patches without.
        Useful to balance the dataset.
    center : boolean, optional
        If False, add some noise to the coordinates of the centroids.
    Returns
    -------
    Two torch.Tensor
        of shape (n_patches, *desired_shape, n_channels)
    """

    native_shape = patch_shape
    patch_shape = torch.tensor(patch_shape, dtype=torch.int)

    centroids_1 = torch.stack(torch.where(y), dim=-1)

    # If the input contains no positive voxels, return random patches:
    if len(centroids_1) == 0:
        return random_sampler(X, y, patch_shape, n_patches)

    n_patches_with = np.floor(n_patches * ratio).astype(int)

    # Add noise to centroids so that the nodules are not always centered in
    # the patch:
    if not center:
        noise = (torch.rand(centroids_1.shape[0], 3) * torch.max(patch_shape)).long() % (
            patch_shape.div(2, rounding_mode="floor")[None, ...]
        )
        centroids_1 += noise - patch_shape.div(4, rounding_mode="floor")

    means = torch.tensor(X.shape).div(2, rounding_mode="floor").float()
    sigmas = torch.tensor(X.shape).div(2, rounding_mode="floor").float() / (3 * 2)

    # Sample centroids at random
    centroids_0 = torch.randn((n_patches - n_patches_with, X.ndim)) * sigmas + means

    # Could this be vectorized?
    for i in range(1, centroids_0.ndim):
        centroids_0[:, i] = torch.clamp(centroids_0[:, i], 0, X.shape[i])

    centroids_0 = centroids_0.long()

    # Subsample centroids with positive labels
    selection_1 = (torch.rand(n_patches_with) * centroids_1.shape[0]).long()

    centroids_1 = centroids_1[selection_1]

    centroids = torch.cat([centroids_0, centroids_1])
    centroids = torch.maximum(centroids, patch_shape[None, ...])
    centroids = torch.minimum(
        centroids,
        (torch.tensor(X.shape) - patch_shape.div(2, rounding_mode="floor"))[None, ...],
    )

    centroids = centroids.reshape(centroids.shape[0], 1, 1, 1, centroids.shape[-1])

    ranges = [
        torch.arange(
            -patch_shape[i].div(2, rounding_mode="floor"),
            patch_shape[i].div(2, rounding_mode="floor"),
        )
        for i in torch.arange(len(native_shape))
    ]

    indices = (
        centroids
        + torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1)[None, ...]
    )

    paddings = patch_shape.long()

    X = F.pad(X, tuple(torch.stack([paddings, paddings], dim=-1).flatten()))
    y = F.pad(y, tuple(torch.stack([paddings, paddings], dim=-1).flatten()))
    indices += patch_shape

    indices_list = build_indices_list(indices)

    image_patches = X[indices_list]
    label_patches = y[indices_list]

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


def build_indices_list(indices):
    return [indices[..., i] for i in range(indices.shape[-1])]
