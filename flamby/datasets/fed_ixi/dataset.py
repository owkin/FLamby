from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile
from typing import Union, Tuple, Dict, List
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.ndimage
import requests
import shutil
import os

from monai.transforms import Resize, Compose, ToTensor, AddChannel
from torch import Tensor
from torch.utils.data import Dataset

from utils import _get_id_from_filename, _load_nifti_image_by_id, _load_nifti_image_and_label_by_id, _extract_center_name_from_filename, _get_center_name_from_center_id, _create_train_test_split


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
            transform=None,
            download=False
    ):
        self.root_folder = Path(root).expanduser().joinpath('IXI-Dataset')
        self.transform = transform
        if download:
            self.download(debug=False)

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

    def download(self, debug=False) -> None:
        """
        Downloads demographics information and image archives and stores them in a folder.

        Parameters
            ----------
            debug: bool
                Enables a light version download. hosting synthetic data ? TBD
            tiny: bool
                Downloads the IXI tiny dataset which contains T1 images & labels for segmentation task instead of the standard IXI dataset.
        """
        # 1. Create folder if it does not exist
        # 2. Download

        # Make folder if it does not exist
        self.root_folder.mkdir(exist_ok=True)
        
        demographics_url = [self.MIRRORS[0] + self.DEMOGRAPHICS_FILENAME]  # URL EXCEL

        for file_url in self.image_urls + demographics_url:
            img_tarball_archive_name = file_url.split('/')[-1]
            img_archive_path = self.root_folder.joinpath(img_tarball_archive_name)
            if img_archive_path.is_file():
                continue
            with requests.get(file_url, stream=True) as response:
                # Raise error if not 200
                response.raise_for_status()
                file_size = int(response.headers.get('Content-Length', 0))
                desc = '(Unknown total file size)' if file_size == 0 else ''
                print(f'Downloading to {img_archive_path}')
                with tqdm.wrapattr(response.raw, 'read', total=file_size, desc=desc) as r_raw:
                    with open(img_archive_path, 'wb') as f:
                        shutil.copyfileobj(r_raw, f)

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

    def _validate_center(self) -> None:
        """
        Asserts permitted image center keys.

        Allowed values are:
            - 1
            - 2
            - 3
            - Guys
            - HH
            - IOP

        Raises
        -------
            AssertionError
                If `center` argument is not contained amongst possible centers.
        """
        centers =  list(self.CENTER_LABELS.keys()) + list(self.CENTER_LABELS.values())
        assert self.centers[0] in centers, f'Center {self.centers[0]} ' \
                                                                    f'is not compatible with this dataset. ' \
                                                                    f'Existing centers can be named as follow: {centers} '

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
    """
    Generic interface for T1 images in IXI Dataset. Stores all paths of images and respective center names.
    """
    def __init__(self, root, transform=None, download=False):
        super(T1ImagesIXIDataset, self).__init__(root, transform=transform, download=download)
        self.modality = 'T1'
        self.image_urls = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar']
        if download:
            self.download()

        # Download of the T1 images must be completed and extracted to run this part
        self.parent_dir_name = Path(self.tar_file.name).resolve().stem # 'IXI-T1'
        self.subjects_dir = os.path.join(root,"IXI-Dataset",self.parent_dir_name)

        self.images_paths = [] # contains paths of archives which contain a nifti image for each subject
        self.images_centers = [] # contains center of each subject: HH, Guys or IOP
        self.images_sets = [] # is the subject used for train or test

        self.subjects = os.listdir(self.subjects_dir)
        self.images_centers = [_extract_center_name_from_filename(subject) for subject in self.subjects]
        self.train_test_hh, self.train_test_guys, self.train_test_iop = _create_train_test_split(images_centers=self.images_centers)
        idx_hh, idx_guys, idx_iop = 0, 0, 0

        for subject in self.subjects:
            center_name = _extract_center_name_from_filename(subject)
            if center_name == 'HH':
                self.images_sets.append(self.train_test_hh[idx_hh])
                idx_hh += 1
            elif center_name == 'Guys':
                self.images_sets.append(self.train_test_guys[idx_guys])
                idx_guys += 1
            else:
                self.images_sets.append(self.train_test_iop[idx_iop])
                idx_iop += 1
            subject_archive = os.path.join(self.subjects_dir,subject)
            image_path = Path(subject_archive)
            self.images_paths.append(image_path)

class T2ImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None, download=False):
        super(T2ImagesIXIDataset, self).__init__(root, transform=transform, download=download)
        self.modality = 'T2'
        self.image_urls = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar']
        if download:
            self.download()


class PDImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None, download=False):
        super(PDImagesIXIDataset, self).__init__(root, transform=transform, download=download)
        self.modality = 'PD'
        self.common_shape = (-1, -1, 125)
        self.image_urls = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar']
        if download:
            self.download()


class MRAImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None, download=False):
        super(MRAImagesIXIDataset, self).__init__(root, transform=transform, download=download)
        self.modality = 'MRA'
        self.common_shape = (512, 512, 100)
        self.image_urls = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar']
        if download:
            self.download()


class DTIImagesIXIDataset(IXIDataset):
    def __init__(self, root, transform=None, download=False):
        super(DTIImagesIXIDataset, self).__init__(root, transform=transform, download=download)
        self.modality = 'DTI'
        # self.common_shape = (512, 512, 100)
        self.image_urls = ['https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar']
        if download:
            self.download()


class MultiModalIXIDataset(IXIDataset):
    pass


class FedT1ImagesIXIDataset(T1ImagesIXIDataset):
    """
    Federated class for T1 images in IXI Dataset

    Parameters
        ----------
        root: str
            Folder where data is.
        center: int, str
            Id of the center (hospital) from which to gather data. Defaults to 1.
        train : bool, optional
            Whether to take the train or test split. Defaults to True (train).
        pooled : bool, optional
            Whether to take all data from the 3 centers into one dataset.
            If True, supersedes center argument. Defaults to False.
    """
    def __init__(self, root, center=1, train=True, pooled=False):
        super(FedT1ImagesIXIDataset, self).__init__(root)

        self.modality = 'T1'
        self.centers = [center]

        # Validation routine for dataset robustness
        self._validate_center()

        if isinstance(center, int):  
            self.centers = [_get_center_name_from_center_id(self.CENTER_LABELS, center)]

        if pooled:
            self.centers = ['HH', 'Guys', 'IOP']

        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.images_centers[idx] in self.centers)
            and (self.images_sets[idx] in self.sets)
            for idx, _ in enumerate(self.images_centers)
        ]

        self.center_images_paths = [self.images_paths[i] for i, s in enumerate(to_select) if s]
        self.images_centers = [self.images_centers[i] for i, s in enumerate(to_select) if s]
        self.images_sets = [self.images_sets[i] for i, s in enumerate(to_select) if s]

    @property
    def subject_ids_fed(self) -> tuple:
        filenames = [Path(filename).name for filename in self.center_images_paths]
        subject_ids = tuple(set(map(_get_id_from_filename, filenames)))
        return subject_ids

    def __len__(self) -> int:
        return len(self.center_images_paths)

    def __getitem__(self, item) -> Tuple[Tensor, Dict]:
        patient_id = self.subject_ids_fed[item]
        headers, img, center_name = _load_nifti_image_by_id(tar_file=self.tar_file,
                                                            patient_id=patient_id,
                                                            modality=self.modality)

        default_transform = Compose([
            ToTensor(),
            AddChannel(),
            Resize(self.common_shape),
        ])
        img = default_transform(img)

        metadata = {'IXI_ID': patient_id, 'center': center_name, 'center_label': self.CENTER_LABELS[center_name]}

        if self.transform:
            img = self.transform(img)
        return img, metadata


class IXITinyDataset(Dataset):
    CENTER_LABELS = {'HH': 1, 'Guys': 2, 'IOP': 3}

    def __init__(self, root, transform=None, download=False):
        self.root_folder = Path(root).expanduser().joinpath('IXI-Dataset')
        self.image_url = 'https://data.mendeley.com/api/datasets-v2/datasets/7kd5wj7v7p/zip/download?version=1'
        self.common_shape = (48, 60, 48)
        self.transform = transform
        self.modality = 'T1'
        if download:
            self.download(debug=False)
        
        # Download of the ixi tiny must be completed and extracted to run this part
        self.parent_dir_name = os.path.join('IXI Sample Dataset','7kd5wj7v7p-1','IXI_sample')
        self.subjects_dir = os.path.join(root,'IXI-Dataset',self.parent_dir_name)

        self.images_paths = [] # contains paths of archives which contain a nifti image for each subject
        self.labels_paths = [] # contains paths of archives which contain a label (binary brain mask) for each subject
        self.images_centers = [] # contains center of each subject: HH, Guys or IOP
        self.images_sets = [] # train and test

        self.subjects = [subject for subject in os.listdir(self.subjects_dir) if os.path.isdir(os.path.join(self.subjects_dir, subject))]
        self.images_centers = [_extract_center_name_from_filename(subject) for subject in self.subjects]

        self.train_test_hh, self.train_test_guys, self.train_test_iop = _create_train_test_split(images_centers=self.images_centers)

        self.demographics = Path(os.path.join(self.subjects_dir,'IXI.xls'))

        idx_hh, idx_guys, idx_iop = 0, 0, 0

        for subject in self.subjects:
            center_name = _extract_center_name_from_filename(subject)
            if center_name == 'HH':
                self.images_sets.append(self.train_test_hh[idx_hh])
                idx_hh += 1
            elif center_name == 'Guys':
                self.images_sets.append(self.train_test_guys[idx_guys])
                idx_guys += 1
            else:
                self.images_sets.append(self.train_test_iop[idx_iop])
                idx_iop += 1
            subject_dir = os.path.join(self.subjects_dir,subject)
            image_path = Path(os.path.join(subject_dir,'T1'))
            label_path = Path(os.path.join(subject_dir,'label'))
            self.images_paths.extend(image_path.glob('*.nii.gz'))
            self.labels_paths.extend(label_path.glob('*.nii.gz'))
    
    @property
    def zip_file(self) -> ZipFile:
        zf = self.root_folder.joinpath('IXI Sample Dataset.zip')
        return ZipFile(zf)

    @property
    def subject_ids(self) -> tuple:
        filenames = [Path(filename).name for filename in self.zip_file.namelist() if Path(Path(filename).name).suffix == '.gz']
        subject_ids = tuple(set(map(_get_id_from_filename, filenames)))
        return subject_ids

    def download(self, debug=False) -> None:
        self.root_folder.mkdir(exist_ok=True)

        img_zip_archive_name = 'IXI Sample Dataset.zip'
        img_archive_path = self.root_folder.joinpath(img_zip_archive_name)
        if img_archive_path.is_file():
            return
        with requests.get(self.image_url, stream=True) as response:
            # Raise error if not 200
            response.raise_for_status()
            file_size = int(response.headers.get('Content-Length', 0))
            desc = '(Unknown total file size)' if file_size == 0 else ''
            print(f'Downloading to {img_archive_path}')
            with tqdm.wrapattr(response.raw, 'read', total=file_size, desc=desc) as r_raw:
                with open(img_archive_path, 'wb') as f:
                    shutil.copyfileobj(r_raw, f)
    
    def _validate_center(self) -> None:
        centers =  list(self.CENTER_LABELS.keys()) + list(self.CENTER_LABELS.values())
        assert self.centers[0] in centers, f'Center {self.centers[0]} ' \
                                                                    f'is not compatible with this dataset. ' \
                                                                    f'Existing centers can be named as follow: {centers} '

    def __getitem__(self, item) -> Tuple[Tensor, Dict]:
        patient_id = self.subject_ids[item]
        header_img, img, label, center_name = _load_nifti_image_and_label_by_id(zip_file=self.zip_file,
                                                            patient_id=patient_id,
                                                            modality=self.modality)

        default_transform = Compose([
            ToTensor(),
            AddChannel(),
            Resize(self.common_shape),
        ])
        img = default_transform(img)
        label = default_transform(label)

        metadata = {'IXI_ID': patient_id, 'center': center_name, 'center_label': self.CENTER_LABELS[center_name]}

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.images_paths)


class FedIXITinyDataset(IXITinyDataset):
    def __init__(self, root, center=1, train=True, pooled=False):
        super(FedIXITinyDataset, self).__init__(root)

        self.modality = 'T1'
        self.centers = [center]
        self._validate_center()

        if isinstance(center, int):  
            self.centers = [_get_center_name_from_center_id(self.CENTER_LABELS, center)]

        if pooled:
            self.centers = ['HH', 'Guys', 'IOP']
        
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.images_centers[idx] in self.centers)
            and (self.images_sets[idx] in self.sets)
            for idx, _ in enumerate(self.images_centers)
        ]

        self.center_images_paths = [self.images_paths[i] for i, s in enumerate(to_select) if s]
        self.center_labels_paths = [self.labels_paths[i] for i, s in enumerate(to_select) if s]
        self.images_centers = [self.images_centers[i] for i, s in enumerate(to_select) if s]
        self.images_sets = [self.images_sets[i] for i, s in enumerate(to_select) if s]

    @property
    def subject_ids_fed(self) -> tuple:
        filenames = [Path(filename).name for filename in self.center_images_paths]
        subject_ids = tuple(set(map(_get_id_from_filename, filenames)))
        return subject_ids

    def __len__(self) -> int:
        return len(self.center_images_paths)

    def __getitem__(self, item) -> Tuple[Tensor, Dict]:
        patient_id = self.subject_ids_fed[item]
        header_img, img, label, center_name = _load_nifti_image_and_label_by_id(zip_file=self.zip_file,
                                                            patient_id=patient_id,
                                                            modality=self.modality)

        default_transform = Compose([
            ToTensor(),
            AddChannel(),
            Resize(self.common_shape),
        ])
        img = default_transform(img)
        label = default_transform(label)

        metadata = {'IXI_ID': patient_id, 'center': center_name, 'center_label': self.CENTER_LABELS[center_name]}

        if self.transform:
            img = self.transform(img)
        return img, label

    
a = FedIXITinyDataset(".")
print(f'Data gathered in this federated dataset is from:', *a.centers, "and", *a.sets, "set")
print('Federated dataset size', len(a))
print('First entry:', a[0])

a = FedT1ImagesIXIDataset(".")
print(f'Data gathered in this federated dataset is from:', *a.centers, "and", *a.sets, "set")
print('Federated dataset size', len(a))
print('First entry:', a[0])


__all__ = [
    'IXIDataset',
    'T1ImagesIXIDataset',
    'T2ImagesIXIDataset',
    'PDImagesIXIDataset',
    'MRAImagesIXIDataset',
    'DTIImagesIXIDataset',
    'MultiModalIXIDataset',
    'FedT1ImagesIXIDataset',
    'FedT2ImagesIXIDataset',
    'FedPDImagesIXIDataset',
    'FedMRAImagesIXIDataset',
    'FedDTIImagesIXIDataset',
    'IXITinyDataset',
    'FedIXITinyDataset',
]