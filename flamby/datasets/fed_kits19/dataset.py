import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nnunet
import os
import sys

import flamby.datasets.fed_kits19
from collections import OrderedDict
from flamby.utils import check_dataset_from_config
from flamby.datasets.fed_kits19.dataset_creation_scripts.nnunet_library.paths import *
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


class KiTS19Raw(Dataset):
    """Pytorch dataset containing all the images, and segmentations
     for KiTS19
    Attributes
    ----------
    plan_dir : str, Where all preprocessing plans and augmentation details of the dataset are saved, used for preprocessing
    dataset_directory: where preprocessed dataset is saved
    debug: bool, Whether or not we use the dataset with only part of the features
    """
    def __init__(self, train = True, X_dtype=torch.float32, y_dtype=torch.float32, augmentations = None, debug=False):
        """See description above
        Parameters
        ----------
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.int64`.
        debug : bool, optional,
            Whether or not to use only the part of the dataset downloaded in
            debug mode. Defaults to False.
        """
        dict = check_dataset_from_config("fed_kits19", debug)
        #TODO: Either Define Augmentations/Transformations in this file or the benchmark file. [here would be better]

        # plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
        # plans = load_pickle(plans_file)
        # stage_plans = plans['plans_per_stage'][0]
        # patch_size = np.array(stage_plans['patch_size']).astype(int)
        # data_aug_params = default_3D_augmentation_params
        # data_aug_params['patch_size_for_spatialtransform'] = patch_size
        # basic_generator_patch_size = get_patch_size(patch_size, data_aug_params['rotation_x'],
        #                                             data_aug_params['rotation_y'],
        #                                             data_aug_params['rotation_z'],
        #                                             data_aug_params['scale_range'])
        # pad_all_sides = None




        self.dataset_directory = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'
        self.augmentations = augmentations

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.train_test = "train" if train else "test"

        df = pd.read_csv('metadata/thresholded_sites.csv')
        df2 = df.query("train_test_split == '" + self.train_test + "' ").reset_index(drop=True)
        self.images = df2.case_ids.tolist()

        # Load image paths and properties files
        c = 0 # Case
        self.images_path = OrderedDict()
        for i in self.images:
            self.images_path[c] = OrderedDict()
            self.images_path[c]['data_file'] = join(self.dataset_directory, "%s.npz" % i)

            # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
            self.images_path[c]['properties_file'] = join(self.dataset_directory, "%s.pkl" % i)

            if self.images_path[c].get('seg_from_prev_stage_file') is not None:
                self.images_path[c]['seg_from_prev_stage_file'] = join(self.dataset_directory, "%s_segs.npz" % i)
            self.images_path[c]['properties'] = load_pickle(self.images_path[c]['properties_file'])
            c += 1

        self.centers = df2.site_ids



    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        if isfile(self.images_path[idx]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self.images_path[idx]['data_file'][:-4] + ".npy", memmap_mode = "r")
        else:
            case_all_data = np.load(self.images_path[idx]['data_file'])['data']

        X = case_all_data[:-1]    # Data
        y = case_all_data[-1:]    # Segmentation

        #TODO:Oversample the data

        #TODO:Apply augmentations

        #Extract X and y.
        # X = np.load(self.features_paths[idx])[:, start:]
        # X = torch.from_numpy(X).to(self.X_dtype)
        # y = torch.from_numpy(np.asarray(self.features_labels[idx])).to(self.y_dtype)
        return X, y


class FedKiTS19(KiTS19Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Camelyon16 federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 2 centers it was created from or all data pooled.
    The train/test split corresponds to the one from the Challenge.
    """

    def __init__(
        self,
        center=0,
        train=True,
        pooled=False,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
        augmentations = None,
    ):
        """Instantiate the dataset
        Parameters
        pooled : bool, optional
            Whether to take all data from the 2 centers into one dataset, by
            default False
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.float32`.
        debug : bool, optional,
            Whether or not to use only the part of the dataset downloaded in
            debug mode. Defaults to False.
        augmentations: Augmentations to be applied on X
        center: Silo ID, must be from the set [0, 1, 2, 3, 4, 5]
        """
        super().__init__(X_dtype=X_dtype, y_dtype=y_dtype,  augmentations = None, debug=debug)

        key =  self.train_test + "_" + str(center)
        print(key)
        if not pooled:
            assert center in range(6)
            df = pd.read_csv('metadata/thresholded_sites.csv')
            df2 = df.query("train_test_split_silo == '" + key + "' ").reset_index(drop=True)
            self.images = df2.case_ids.tolist()
            c = 0
            for i in self.images:
                self.images_path[c] = OrderedDict()
                self.images_path[c]['data_file'] = join(self.dataset_directory, "%s.npz" % i)

                # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
                self.images_path[c]['properties_file'] = join(self.dataset_directory, "%s.pkl" % i)
                if self.images_path[c].get('seg_from_prev_stage_file') is not None:
                    self.images_path[c]['seg_from_prev_stage_file'] = join(self.dataset_directory, "%s_segs.npz" % i)
                self.images_path[c]['properties'] = load_pickle(self.images_path[c]['properties_file'])
                c += 1

            self.centers = df2.site_ids
            print(self.centers)


if __name__ == "__main__":
    train_dataset = FedKiTS19(5, train=False, pooled=False, augmentations=None)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    for sample in train_dataloader:
        print(sample[0].shape)
        print(sample[1].shape)



