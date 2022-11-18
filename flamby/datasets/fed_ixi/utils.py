"""Federated IXI Dataset utils
A set of function that allow data management suited for the `IXI dataset
<https://brain-development.org/ixi-dataset/>`_.

"""

import os
import re
import tempfile
from os import PathLike
from pathlib import Path
from tarfile import TarFile
from typing import List, Tuple, Union
from zipfile import ZipFile

import nibabel as nib
import numpy
import numpy as np


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
    matches = re.findall(r"IXI\d{3}", x)
    matches = [int(match[-3:]) for match in matches]

    if verify_single_matches:
        assert len(matches) == 1, (
            "One or more subjects are repeated into the "
            "dataset. This is not expected behaviour."
            "Please verify that there is only one image "
            "per subject and modality."
        )
        return matches[0]
    return matches


def _assembly_nifti_filename_regex(
    patient_id: int, modality: str
) -> Union[str, PathLike, Path]:
    """Assembles NIFTI filename regular expression using the standard in the
    IXI dataset based on id and modality.

    Parameters
    ----------
    patient_id : int
        Patient's identifier.

    modality: str
        Image modality (e.g. `'T1'`).

    Returns
    -------
    str
        A nifti image file name.

    """
    nii_filename = f"IXI{patient_id:03d}-[A-Za-z]+-[0-9]+-{modality.upper()}.nii.gz"
    return nii_filename


def _assembly_nifti_img_and_label_regex(
    patient_id: int, modality: str
) -> Tuple[Union[str, PathLike, Path], Union[str, PathLike, Path]]:
    """Assembles NIFTI filename regular expression for image and label using
    the standard in the IXI tiny dataset based on id and modality.

    Parameters
    ----------
    patient_id : int
        Patient's identifier.

    modality: str
        Image modality (e.g. `'T1'`).

    Returns
    -------
    str
        A nifti image file name.
    str
        A nifti label file name.

    """
    nii_filename_img = (
        f".+{modality.upper()}/IXI{patient_id:03d}-[A-Za-z]+-[0-9]+.+.nii.gz"
    )
    nii_filename_label = f".+label/IXI{patient_id:03d}-[A-Za-z]+-[0-9]+.+.nii.gz"
    return nii_filename_img, nii_filename_label


def _find_file_in_tar(tar_file: TarFile, patient_id: int, modality) -> str:
    """Searches the file in a TAR file that corresponds to a particular
    regular expression.

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

    raise FileNotFoundError(f"File following the pattern {regex} could not be found.")


def _find_files_in_zip(zip_file: ZipFile, patient_id: int, modality) -> Tuple[str]:
    """Searches the files in a ZIP file that corresponds to particular regular
    expressions.

    Parameters
    ----------
    patient_id : int
        Patient's ID

    Returns
    -------
    str
        Image filename corresponding to the particular subject's ID
    str
        Label filename corresponding to the particular subject's ID

    Raises
    -------
    FileNotFoundError
        If files were not found in ZIP.
    """
    regex_img, regex_label = _assembly_nifti_img_and_label_regex(patient_id, modality)
    filenames = zip_file.namelist()
    result = [None, None]

    for filename in filenames:
        try:
            result[0] = re.match(regex_img, filename).group()
        except AttributeError:
            try:
                result[1] = re.match(regex_label, filename).group()
            except AttributeError:
                continue
        if result[0] is not None and result[1] is not None:
            return tuple(result)

    raise FileNotFoundError(
        f"Files following the pattern {regex_img} and {regex_label} could not be found."
    )


def _extract_center_name_from_filename(filename: str):
    """Extracts center name from file dataset.

    Unfortunately, IXI has the center encoded in the namefile rather than in
    the demographics information.

    Parameters
    ----------
    filename : str
        Basename of NIFTI file (e.g. `IXI652-Guys-1116-MRA.nii.gz`)

    Returns
    -------
    str
        Name of the center where the data comes from (e.g. Guys for the previous example)

    """
    # We decided to wrap a function for this for clarity and easier modularity
    # for future expansion
    return filename.split("-")[1]


def _load_nifti_image_by_id(
    tar_file: TarFile, patient_id: int, modality
) -> Tuple[nib.Nifti1Header, numpy.ndarray, str]:
    """Loads NIFTI file from TAR file using a specific ID.

    Parameters
    ----------
    tar_file : TarFile
        `TarFile
        <https://docs.python.org/3/library/tarfile.html#tarfile-objects>`_ object
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
        Name of the center the file comes from. In IXI this is encoded only
        in the filename.
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


def _load_nifti_image_and_label_by_id(
    zip_file: ZipFile, patient_id: int, modality
) -> Tuple[nib.Nifti1Header, numpy.ndarray, numpy.ndarray, str]:
    """Loads NIFTI file from ZIP file using a specific ID.

    Parameters
    ----------
    zip_file : ZipFile
        `ZipFile
        <https://docs.python.org/3/library/zipfile.html#zipfile-objects>`_ object
    patient_id : int
        Patient's ID whose image is to be extracted.
    modality : str
        MRI modality (e.g. `'T1'`).

    Returns
    -------
    header_img : Nifti1Header
        NIFTI headers proceeding from the image.
    img : ndarray
        NumPy array containing the intensities of the voxels.
    label : ndarray
        NumPy array containing the intensities of the voxels.
    center_name : str
        Name of the center the file comes from. In IXI this is encoded only in
        the filename.
    """
    img_filename, label_filename = _find_files_in_zip(zip_file, patient_id, modality)
    with tempfile.TemporaryDirectory() as td:
        img_full_path = os.path.join(td, img_filename)
        label_full_path = os.path.join(td, label_filename)
        zip_file.extract(img_filename, td)
        zip_file.extract(label_filename, td)
        nii_img = nib.load(img_full_path)
        nii_label = nib.load(label_full_path)
        # nii_img = nib.as_closest_canonical(nii_img)
        # nii_img = processing.conform(nii_img) #, out_shape=nii_img.shape)
        img = nii_img.get_fdata()
        label = nii_label.get_fdata()
        header_img = nii_img.header
        # header_label = label.header

    return (
        header_img,
        img,
        label,
        _extract_center_name_from_filename(Path(img_filename).name),
    )


def _get_center_name_from_center_id(center_labels: dict, center_id: int) -> str:
    """Extract ID from NIFTI filename for cross-reference

    Parameters
    ----------
    center_labels : dict
        Dictionary containing hospital names as keys and numerical ids as values.

    center_id : int
        Hospital id for which we want to retrieve the name.

    Returns
    -------
    str
        Name of the center where the data comes from
    """
    center_name = list(center_labels.keys())[
        list(center_labels.values()).index(center_id)
    ]
    return center_name


def _create_train_test_split(
    images_centers: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Creates a train test split for each center using random seed to reproduce results.

    Parameters
    ----------
    images_centers: list
        List of hospital names for each subject.

    Returns
    -------
    train_test_hh : list
            A list containing randomly generated dichotomous values.
            The size is the number of images from HH hospital.
            Dichotomous values (train and test) follow a train test split
            threshold (e. g. 70%).
    train_test_guys : list
            A list containing randomly generated dichotomous values.
            The size is the number of images from Guys hospital.
            Dichotomous values (train and test) follow a train test split
            threshold (e. g. 70%).
    train_test_iop : list
            A list containing randomly generated dichotomous values.
            The size is the number of images from IOP hospital.
            Dichotomous values (train and test) follow a train test split
            threshold (e. g. 70%).

    """
    split_ratio = 0.8

    n_hh = images_centers.count("HH")
    n_guys = images_centers.count("Guys")
    n_iop = images_centers.count("IOP")

    n_train_hh = round(n_hh * split_ratio)
    n_train_guys = round(n_guys * split_ratio)
    n_train_iop = round(n_iop * split_ratio)

    n_test_hh = n_hh - n_train_hh
    n_test_guys = n_guys - n_train_guys
    n_test_iop = n_iop - n_train_iop

    np.random.seed(10)
    train_test_hh = ["train"] * n_train_hh + ["test"] * n_test_hh
    np.random.shuffle(train_test_hh)
    train_test_guys = ["train"] * n_train_guys + ["test"] * n_test_guys
    np.random.shuffle(train_test_guys)
    train_test_iop = ["train"] * n_train_iop + ["test"] * n_test_iop
    np.random.shuffle(train_test_iop)
    return train_test_hh, train_test_guys, train_test_iop
