import os
from os import PathLike
from pathlib import Path
from tarfile import TarFile
from typing import Union, Iterable, List, Tuple, Dict

import numpy
from torch import Tensor
from torch.utils.data import Dataset

from monai.transforms import Resize, Compose, ToTensor

import re
import nibabel as nib
import pandas as pd
import tempfile
import requests


def _get_id_from_filename(x, verify_single_matches=True) -> Union[List[int], int]:
    """
    Extract ID from NIFTI filename for cross-reference

    Parameters
    ----------
    x : str
        Basename of NIFTI file (e.g. `IXI652-Guys-1116-MRA.nii.gz`)

    verify_single_matches: bool
        Asserts the existance of a single match.
        Used to verify that there exist only one image per subject and modality.

    Returns
    -------
    list
        A list containing the matched IDs found (each id has an integer type).
    """
    matches = re.findall(r'IXI\d{3}', x)
    matches = [int(match[-3:]) for match in matches]

    if verify_single_matches:
        assert len(matches) == 1, 'One or more subjects are repeated into the ' \
                                  'dataset. This is not expected behaviour.' \
                                  'Please verify that there is only one image ' \
                                  'per subject and modality.'
        return matches[0]
    return matches


def _assembly_nifti_filename_regex(patient_id: int, modality: str) -> Union[str, PathLike, Path]:
    """
    Assembles NIFTI filename regular expression using the standard in the IXI dataset based on id and modality.

    Parameters
    ----------
    patient_id: int
        Patient's identifier.

    modality: str
        Image modality (e.g. `'T1'`).

    Returns
    -------

    """
    nii_filename = f'IXI{patient_id:03d}-[A-Za-z]+-[0-9]+-{modality.upper()}.nii.gz'
    return nii_filename


def _find_file_in_tar(tar_file: TarFile, patient_id: int, modality) -> str:
    """
    Searches the file in a TAR file that corresponds to a particular regular expression.

    Parameters
    ----------
    patient_id: int
        Patient's ID

    Returns
    -------
    str
        Filename corresponding to the particular subject's ID

    Raises
    -------
    FileNotFoundError
        If file was not found in TAR.
    """
    regex = _assembly_nifti_filename_regex(patient_id, modality)
    filenames = tar_file.getnames()

    for filename in filenames:
        try:
            return re.match(regex, filename).group()
        except AttributeError:
            continue

    raise FileNotFoundError(f'File following the pattern {regex} could not be found.')


def _extract_center_name_from_filename(filename: str):
    """
    Extracts center name from file dataset.

    Unfortunately, IXI has the center encoded in the namefile rather than in the demographics information.

    Parameters
    ----------
    filename: str
        Basename of NIFTI file (e.g. `IXI652-Guys-1116-MRA.nii.gz`)

    Returns
    -------
    str
        Name of the center where the data comes from (e.g. Guys for the previous example)

    """
    # We decided to wrap a function for this for clarity and easier modularity for future expansion
    return filename.split('-')[1]


def _load_nifti_image_by_id(tar_file: TarFile, patient_id: int, modality) -> Tuple[
    nib.Nifti1Header, numpy.ndarray, str]:
    filename = _find_file_in_tar(tar_file, patient_id, modality)
    with tempfile.TemporaryDirectory() as td:
        full_path = os.path.join(td, filename)
        tar_file.extract(filename, td)
        nii_img = nib.load(full_path)
        img = nii_img.get_fdata()
        header = nii_img.get_header()
    return header, img, _extract_center_name_from_filename(filename)


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
    image_urls = [
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar",
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar",
        "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar"
    ]

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
        self.common_shape = (-1, 150)  # Common shape (some images need resizing on the z-axis

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

    def download(self, debug) -> None:
        """
        Downloads demographics information and image archives and stores them in a folder.

        Parameters
            ----------
            debug: bool
                Enables a light version download. hosting synthetic data ? TBD
        """
        # 1. Create folder if it does not exist
        # 2. Download

        parent_dir = "IXI-Dataset"
        if os.path.isdir(parent_dir) == False:
            os.makedirs(parent_dir)

        url_xls = self.MIRRORS[0] + self.DEMOGRAPHICS_FILENAME  # URL EXCEL
        demographics_path = f'./{parent_dir}/{self.DEMOGRAPHICS_FILENAME}'
        
        if not os.path.isfile(demographics_path):
            response = requests.get(url_xls, stream=True)
            if response.status_code == 200:
                with open(demographics_path, 'wb') as f:
                    f.write(response.raw.read())
        
        for img_url in self.image_urls:
            img_archive_name = img_url.split('/')[-1]
            img_archive_path = f'./{parent_dir}/{img_archive_name}'
            if os.path.isfile(img_archive_path):
                continue
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                with open(img_archive_path, 'wb') as f:
                    f.write(response.raw.read())

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
            img = self[0][0]
            middle_slice = img.shape[2] // 2
            plt.imshow(img[..., middle_slice], cmap='gray')
            plt.title('Modality: ' + self.modality)
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


class MRAImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None):
        super(MRAImagesIXIDataset, self).__init__(root, transform=transform)
        self.modality = 'MRA'


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
