from os import PathLike
from pathlib import Path
from typing import Union, Iterable

from torch.utils.data import Dataset

import nibabel as nib
import pandas as pd


class IXIDataset(Dataset):
    """
    Generic interface for IXI Dataset

    Parameters
        ----------
        root: str
            Folder where data is or will be downloaded.
        transform:
            PyTorch Transform to process the data or augment it.
    """

    demographics = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls']
    image_urls = []

    def __init__(
            self,
            root: str,
            transform=None
    ):
        self.root = Path(root).expanduser()
        self.transform = transform

    def download(self) -> None:
        pass

    def __getitem__(self, item) -> Iterable:
        pass

    def __len__(self) -> int:
        pass


class T1ImagesIXIDataset(IXIDataset):
    pass


class T2ImagesIXIDataset(IXIDataset):
    pass


class PDImagesIXIDataset(IXIDataset):
    pass


class MRAImagesIXIDataset(IXIDataset):
    pass


class DTIImagesIXIDataset(IXIDataset):
    pass


class MultiModalIXIDataset(IXIDataset):
    pass


__all__ = [
    'IXIDataset',
    'T1ImagesIXIDataset',
    'T2ImagesIXIDataset',
    'PDImagesIXIDataset',
    'MRAImagesIXIDataset',
    'DTIImagesIXIDataset',
    'MultiModalIXIDataset',
]
