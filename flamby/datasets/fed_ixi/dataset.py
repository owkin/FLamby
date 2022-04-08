from pathlib import Path
from tarfile import TarFile
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.ndimage
from monai.transforms import Resize, Compose, ToTensor, AddChannel
from torch import Tensor
from torch.utils.data import Dataset

from .utils import _get_id_from_filename, _load_nifti_image_by_id


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

    MIRRORS = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/']
    DEMOGRAPHICS_FILENAME = 'IXI.xls'
    image_urls = []

    ALLOWED_MODALITIES = ['T1', 'T2', 'PD', 'MRA', 'DTI']
    CENTER_LABELS = {'HH': 1, 'Guys': 2, 'IOP': 3}

    def __init__(
            self,
            root: str,
            transform=None
    ):
        self.root_folder = Path(root).expanduser().joinpath('IXI-Dataset')
        self.transform = transform

        self.demographics = self._load_demographics()
        self.modality = 'T1'  # T1 modality by default
        self.common_shape = (-1, -1, 150)  # Common shape (some images need resizing on the z-axis

        # Validation routines for dataset robustness
        self._validate_modality()

    @property
    def subject_ids(self) -> tuple:
        filenames = self.tar_file.getnames()
        subject_ids = tuple(map(_get_id_from_filename, filenames))
        return subject_ids

    @property
    def tar_file(self) -> TarFile:
        tf = self.root_folder.joinpath(f'IXI-{self.modality.upper()}.tar')
        return TarFile(tf)

    def download(self, x) -> None:
        # 1. Create folder if it does not exist
        # 2. Download
        url_xls = self.MIRRORS[0] + self.DEMOGRAPHICS_FILENAME  # URL EXCEL

        for img_url in self.image_urls:
            # download
            pass
        pass

    def _load_demographics(self) -> pd.DataFrame:
        demographics_file = self.root_folder.joinpath(self.DEMOGRAPHICS_FILENAME)
        if not demographics_file.is_file():
            raise FileNotFoundError(f'file {demographics_file} has not been found. '
                                    f'Please make sure you downloaded the dataset '
                                    f'by marking the argument download=True')

        return pd.read_excel(demographics_file, sheet_name='Table')

    def _validate_subject_id(self, subject_id: Union[str, int]):
        assert int(subject_id) in self.subject_ids, f'It seems that subject id {subject_id:03d} is not present.' \
                                                    f'Please verify that the subject actually exists.'

    def _validate_modality(self) -> None:
        """
        Asserts permitted image modality keys.

        Allowed values are:
            - T1
            - T2
            - PD
            - MRA
            - DTI

        Raises
        -------
            AssertionError
                If `modality` argument is not contained amongst possible modalities.
        """
        assert self.modality.upper() in self.ALLOWED_MODALITIES, f'Modality {self.modality} ' \
                                                                 f'is not compatible with this dataset. ' \
                                                                 f'Existing modalities are {self.ALLOWED_MODALITIES} '

    def simple_visualization(self):
        try:
            import matplotlib.pyplot as plt
            img = self[0][0][0]  # [observation index][image tensor][color channel]
            fig, axs = plt.subplots(ncols=3)
            sx, sy, sz = scipy.ndimage.center_of_mass(np.where(img, 1, 0))
            sx, sy, sz = int(sx), int(sy), int(sz)

            axs[0].imshow(img[sx, ...], cmap='gray')
            axs[1].imshow(img[:, sy, :], cmap='gray')
            axs[2].imshow(img[..., sz], cmap='gray')

            plt.suptitle('Modality: ' + self.modality)
            plt.show()
        except ImportError:
            import warnings
            warnings.warn('To visualize it is necessary to have matplotlib. '
                          'Try using `pip install matplotlib and then '
                          'launching the command again`')

    def __getitem__(self, item) -> Tuple[Tensor, Dict]:
        patient_id = self.subject_ids[item]
        headers, img, center_name = _load_nifti_image_by_id(tar_file=self.tar_file,
                                                            patient_id=patient_id,
                                                            modality=self.modality)

        # A default transform is required due to inhomogeneities in shape
        default_transform = Compose([
            ToTensor(),
            AddChannel(),
            Resize(self.common_shape),
        ])
        img = default_transform(img)

        # Build a dictionary with the required labels: metadata
        # Keys:
        #  - 'center':
        #    - 1: Hammersmith Hospital using a Philips 3T system
        #    - 2: Guy’s Hospital using a Philips 1.5T system
        #    - 3: Institute of Psychiatry using a GE 1.5T system
        metadata = {'IXI_ID': patient_id, 'center': center_name, 'center_label': self.CENTER_LABELS[center_name]}

        # Make values compatible with PyTorch
        if self.transform:
            img = self.transform(img)
        return img, metadata

    def __len__(self) -> int:
        return len(self.tar_file.getnames())


class T1ImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(T1ImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'T1'


class T2ImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(T2ImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'T2'


class PDImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(PDImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'PD'
        self.common_shape = (-1, -1, 125)


class MRAImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(MRAImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'MRA'
        self.common_shape = (512, 512, 100)


class DTIImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(DTIImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'DTI'
        # self.common_shape = (512, 512, 100)


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