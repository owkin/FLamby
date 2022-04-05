import os
from os import PathLike
from pathlib import Path
from tarfile import TarFile
from typing import Union, Iterable, List

from torch import Tensor
from torch.utils.data import Dataset

import re
import nibabel as nib
import pandas as pd
import tempfile


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


def _load_nifti_image_by_id(tar_file: TarFile, patient_id: int, modality) -> Tensor:
    filename = _find_file_in_tar(tar_file, patient_id, modality)
    with tempfile.TemporaryDirectory() as td:
        full_path = os.path.join(td, filename)
        tar_file.extract(filename, td)
        nii_img = nib.load(full_path)
    return nii_img


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

    def __init__(
            self,
            root: str,
            transform=None
    ):
        self.root_folder = Path(root).expanduser().joinpath('IXI-Dataset')
        self.transform = transform

        self.demographics = self._load_demographics()
        self.modality = 'T1'

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

    def download(self) -> None:
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

    def __getitem__(self, item) -> Iterable:
        patient_id = self.subject_ids[item]
        nii_img = _load_nifti_image_by_id(tar_file=self.tar_file, patient_id=patient_id, modality=self.modality)
        return nii_img

    def __len__(self) -> int:
        return len(self.tar_file.getnames())


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
