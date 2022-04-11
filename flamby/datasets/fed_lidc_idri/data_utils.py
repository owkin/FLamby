import numpy as np
import torch
import torch.nn.functional as F


class Sampler(object):
    """
    3D image patch sampler.
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
        patch_shape=(128, 128, 64),
        n_patches=2,
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
            Sampling algorithm. Default = 'fast'.
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
            of shape (n_patches, n_channels, *desired_shape)
        """
        if self.algo == "fast":
            return fast_sampler(
                X, y, self.patch_shape, self.n_patches, self.ratio, center=self.center
            )
        elif self.algo == "random":
            return random_sampler(X, y, self.patch_shape, self.n_patches)
        elif self.algo == "all":
            return all_sampler(X, y, patch_shape=self.patch_shape)
        elif self.algo == "none":
            return X, y


class ClipNorm(object):
    """
    Clip then normalize transformation.
    Clip to [minval, maxval] then map linearly to [0, 1].
    Attributes
    ----------
    minval : float
        lower bound
    maxval : float
        upper bound
    """

    def __init__(self, minval=-1024, maxval=600):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval

    def __call__(self, image):
        x = torch.clamp(image, self.minval, self.maxval)
        x -= self.minval
        x /= self.maxval - self.minval
        return x


def resize_by_crop_or_pad(X, output_shape=(384, 384, 384)):
    """
    Resizes by padding or cropping centered on the image.
    Works with whatever number of dimensions.
    Parameters
    ----------
    X : torch.Tensor
        Tensor to reshape.
    output_shape : Tuple or None
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
    X = F.pad(X, tuple(padding)[::-1], mode="constant", value=X.min())

    # Crop extra
    extra = torch.clamp(torch.tensor(X.shape) - output_shape, min=0).div(
        2, rounding_mode="floor"
    )

    for d, start in enumerate(extra):
        X = X.narrow(d, start, output_shape[d])

    return X


def random_sampler(image, label, patch_shape=(128, 128, 64), n_samples=2):
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

    patch_shape = torch.tensor(patch_shape).long()
    centroids = sample_centroids(image, n_samples)

    paddings = patch_shape.long()

    image = F.pad(
        image[None, None, :, :, :],
        tuple(torch.stack([paddings, paddings], dim=-1).flatten())[::-1],
        mode="reflect",
    ).squeeze()
    label = F.pad(
        label,
        tuple(torch.stack([paddings, paddings], dim=-1).flatten())[::-1],
        mode="constant",
    )

    image_patches = extract_patches(image, centroids + patch_shape, patch_shape)
    label_patches = extract_patches(label, centroids + patch_shape, patch_shape)

    return image_patches, label_patches


def all_sampler(X, y, patch_shape=(128, 128, 64)):
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
    Returns
    -------
    Two torch.Tensor
        of shape (n_patches, *desired_shape)
    """

    image_patches = (
        X.unfold(0, patch_shape[0], patch_shape[0])
        .unfold(1, patch_shape[1], patch_shape[1])
        .unfold(2, patch_shape[2], patch_shape[2])
        .reshape(-1, *patch_shape)
    )

    label_patches = (
        y.unfold(0, patch_shape[0], patch_shape[0])
        .unfold(1, patch_shape[1], patch_shape[1])
        .unfold(2, patch_shape[2], patch_shape[2])
        .reshape(-1, *patch_shape)
    )

    return image_patches, label_patches


def fast_sampler(
    X,
    y,
    patch_shape=(128, 128, 64),
    n_patches=2,
    ratio=1.0,
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
        of shape (n_patches, n_channels, *desired_shape)
    """

    centroids_1 = torch.stack(torch.where(y), dim=-1)

    # If the input contains no positive voxels, return random patches:
    if len(centroids_1) == 0:
        return random_sampler(X, y, patch_shape, n_patches)

    patch_shape = torch.tensor(patch_shape, dtype=torch.int)

    n_patches_with = np.floor(n_patches * ratio).astype(int)

    # Add noise to centroids so that the nodules are not always centered in
    # the patch:
    if not center:
        noise = (torch.rand(centroids_1.shape[0], 3) * torch.max(patch_shape)).long() % (
            patch_shape.div(2, rounding_mode="floor")[None, ...]
        )
        centroids_1 += noise - patch_shape.div(4, rounding_mode="floor")

    # Sample random centroids
    centroids_0 = sample_centroids(X, n_patches - n_patches_with)

    # Subsample centroids with positive labels
    selection_1 = (torch.rand(n_patches_with) * centroids_1.shape[0]).long()

    centroids_1 = centroids_1[selection_1]

    centroids = torch.cat([centroids_0, centroids_1])
    centroids = torch.maximum(centroids, patch_shape[None, ...])
    centroids = torch.minimum(
        centroids,
        (torch.tensor(X.shape) - patch_shape.div(2, rounding_mode="floor"))[None, ...],
    )

    paddings = patch_shape.long()

    X = F.pad(
        X[None, None, :, :, :],
        tuple(torch.stack([paddings, paddings], dim=-1).flatten())[::-1],
        mode="reflect",
    ).squeeze()
    y = F.pad(
        y,
        tuple(torch.stack([paddings, paddings], dim=-1).flatten())[::-1],
        mode="constant",
    )

    image_patches = extract_patches(X, centroids + patch_shape, patch_shape)
    label_patches = extract_patches(y, centroids + patch_shape, patch_shape)

    return image_patches, label_patches


def extract_patches(image, centroids, patch_shape):
    """
    Extract patches from nD image at given locations.
    Parameters
    ----------
    image: torch.Tensor
        nD Image from which to extract patches.
    centroids: torch.Tensor
        Coordinates of patch centers
    patch_shape: torch.Tensor
        Shape of the desired patches
    Returns
    -------
        torch.Tensor: patches at centroids with desired shape
    """
    slices = [
        [
            slice(
                (centroids[ii] - patch_shape.div(2, rounding_mode="floor"))[i],
                (centroids[ii] + patch_shape.div(2, rounding_mode="floor"))[i],
            )
            for i in range(len(patch_shape))
        ]
        for ii in range(len(centroids))
    ]
    return torch.stack([image[s] for s in slices])


def sample_centroids(X, n_samples):
    """
    Sample eligible centroids for patches of X
    """
    means = torch.tensor(X.shape).div(2, rounding_mode="floor").float()
    sigmas = torch.tensor(X.shape).div(2, rounding_mode="floor").float() / (3 * 2)

    # Sample centroids at random
    centroids = torch.randn((n_samples, X.ndim)) * sigmas + means

    # Could this be vectorized?
    for i in range(1, centroids.ndim):
        centroids[:, i] = torch.clamp(centroids[:, i], 0, X.shape[i])

    return centroids.long()
