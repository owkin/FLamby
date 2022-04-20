"""Federated IXI Dataset utils

A set of function that allow data management suited for the `IXI dataset <https://brain-development.org/ixi-dataset/>`_.

"""

import os
from os import PathLike
import numpy
from tarfile import TarFile
from pathlib import Path

import re
import tempfile
from typing import Union, List, Tuple

import nibabel as nib
import nibabel.processing as processing
import numpy as np
from nibabel import Nifti1Header
from numpy import ndarray


def _get_id_from_filename(x, verify_single_matches=True) -> Union[List[int], int]:
    """Extract ID from NIFTI filename for cross-reference

    Parameters
    ----------
    x : str
        Basename of NIFTI file (e.g. `IXI652-Guys-1116-MRA.nii.gz`)

    verify_single_matches : bool
        Asserts the existence of a single match.
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
    """Assembles NIFTI filename regular expression using the standard in the IXI dataset based on id and modality.

    Parameters
    ----------
    patient_id : int
        Patient's identifier.

    modality: str
        Image modality (e.g. `'T1'`).

    Returns
    -------

    """
    nii_filename = f'IXI{patient_id:03d}-[A-Za-z]+-[0-9]+-{modality.upper()}.nii.gz'
    return nii_filename


def _find_file_in_tar(tar_file: TarFile, patient_id: int, modality) -> str:
    """Searches the file in a TAR file that corresponds to a particular regular expression.

    Parameters
    ----------
    patient_id : int
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
    """Extracts center name from file dataset.

    Unfortunately, IXI has the center encoded in the namefile rather than in the demographics information.

    Parameters
    ----------
    filename : str
        Basename of NIFTI file (e.g. `IXI652-Guys-1116-MRA.nii.gz`)

    Returns
    -------
    str
        Name of the center where the data comes from (e.g. Guys for the previous example)

    """
    # We decided to wrap a function for this for clarity and easier modularity for future expansion
    return filename.split('-')[1]


def _load_nifti_image_by_id(
        tar_file: TarFile,
        patient_id: int, modality) -> Tuple[nib.Nifti1Header, numpy.ndarray, str]:
    """Loads NIFTI file from TAR file using a specific ID.

    Parameters
    ----------
    tar_file : TarFile
        `TarFile <https://docs.python.org/3/library/tarfile.html#tarfile-objects>`_ object
    patient_id : int
        Patient's ID whose image is to be extracted.
    modality : str
        MRI modality (e.g. `'T1'`).


    Returns
    -------
    header : Nifti1Header
        NIFTI headers proceeding from the image.
    img : ndarray
        NumPy array containing the intensities of the voxels.
    center_name : str
        Name of the center the file comes from. In IXI this is encoded only in the filename.
    """
    filename = _find_file_in_tar(tar_file, patient_id, modality)
    with tempfile.TemporaryDirectory() as td:
        full_path = os.path.join(td, filename)
        tar_file.extract(filename, td)
        nii_img = nib.load(full_path)
        # nii_img = nib.as_closest_canonical(nii_img)
        # nii_img = processing.conform(nii_img) #, out_shape=nii_img.shape)
        img = nii_img.get_fdata()
        header = nii_img.get_header()

    return header, img, _extract_center_name_from_filename(filename)

def _get_center_name_from_center_id(center_labels, center_id):
    center_name = list(center_labels.keys())[list(center_labels.values()).index(center_id)]
    return center_name