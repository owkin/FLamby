import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from histolab.masks import TissueMask
from histolab.slide import Slide
from histolab.tiler import GridTiler
from openslide import open_slide
from torch.nn import Identity
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from flamby.utils import read_config, write_value_in_config


class SlideDataset(IterableDataset):
    def __init__(self, grid_tiles_extractor, slide, transform=None):
        self.transform = transform
        # tissue mask is needed to segment all regions
        self.it = grid_tiles_extractor._tiles_generator(
            slide, extraction_mask=TissueMask()
        )

    def __iter__(self):
        for tile in self.it:
            im = tile[0].image.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
            coords = tile[1]._asdict()
            coords = torch.Tensor([coords["x_ul"], coords["y_ul"]]).long()
            yield im, coords


class DatasetFromCoords(Dataset):
    def __init__(self, coords, slide_path, tile_size=224, level=1, transform=None):
        self.transform = transform
        self.coords = coords
        self.slide = open_slide(slide_path)
        self.level = level
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        pil_image = self.slide.read_region(
            self.coords[idx].astype("int_"), self.level, (self.tile_size, self.tile_size)
        ).convert("RGB")
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, self.coords[idx]


def dict_to_df(dict_arg):
    return pd.DataFrame().from_dict(dict_arg)


def save_dict_to_csv(dict_arg, file_name):
    df = dict_to_df(dict_arg)
    df.to_csv(file_name, index=False)


def main(batch_size, num_workers_torch, tile_from_scratch, remove_big_tiff):
    """Function tiling slides that have been downloaded using download.py.

    Parameters
    ----------
    batch_size : int
        The number of images to use for batched inference in pytorch.

    num_workers_torch: int
        The number of parallel torch worker if precomputed coords allowed.

    tile_from_scratch: bool
        If this option is activated we disregard the csv files with precomputed
        coordinates.

    remove_big_tiff : bool
        Whether or not to get rid of all original slides after tiling.

    Raises
    ------
    ValueError
        If the dataset was not downloaded no tiling is possible.
    ValueError
        If the dataset is partially downloaded we don't allow tiling. This
        constraint might be alleviated in the future.
    """
    path_to_config_file = str(Path(os.path.realpath(__file__)).parent.resolve())

    config_file = os.path.join(path_to_config_file, "dataset_location.yaml")
    write_value_in_config(config_file, "preprocessing_complete", False)
    if os.path.exists(config_file):
        debug = False
    else:
        config_file = os.path.join(path_to_config_file, "dataset_location_debug.yaml")
        if os.path.exists(config_file):
            debug = True
        else:
            raise ValueError(
                "The dataset was not downloaded in normal or debug mode, \
                    please run the download script beforehand"
            )

    if debug:
        print("WARNING ONLY DEBUG VERSION OF DATASET FOUND !")

    dict = read_config(config_file)

    if not (dict["download_complete"]):
        raise ValueError(
            "Not all slides seem to be downloaded please relaunch the download script"
        )

    if dict["preprocessing_complete"]:
        print("You have already run the preprocessing, aborting.")
        sys.exit()

    slides_dir = dict["dataset_path"]
    slides_paths = glob(os.path.join(slides_dir, "*.tif"))
    grid_tiles_extractor = GridTiler(
        tile_size=(224, 224),
        level=1,
        pixel_overlap=0,
        prefix="grid/",
        check_tissue=True,
        suffix=".png",
    )
    net = models.resnet50(pretrained=True)
    net.fc = Identity()
    # Probably unnecessary still it feels safer and might speed up computations
    for param in net.parameters():
        param.requires_grad = False

    # Putting batch-normalization layers in eval mode
    net = net.eval()

    if torch.cuda.is_available():
        net = net.cuda()
    else:
        print(
            "Consider using an environment with GPU otherwise the process will \
                take weeks."
        )
    # IMAGENET preprocessing of images scaled between 0. and 1. with ToTensor
    transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    path_to_coords_file = os.path.join(
        Path(os.path.realpath(__file__)).parent.resolve(),
        "tiling_coordinates_camelyon16.csv",
    )
    if not (os.path.exists(path_to_coords_file)):
        df = dict_to_df({"slide_name": [], "coords_x": [], "coords_y": []})
    else:
        df = pd.read_csv(path_to_coords_file)
    # For easier appending
    df_predecessor = df.to_dict("records")

    for sp in tqdm(slides_paths):
        slide_name = os.path.basename(sp)
        path_to_features = os.path.join(slides_dir, slide_name + ".npy")
        if os.path.exists(path_to_features):
            continue
        print(f"Tiling slide {sp}")
        # We use the matter detector from histolab to tile the dataset or the
        # cached coords
        coords_from_slide = df.loc[df["slide_name"] == slide_name.lower()]

        has_coords = len(coords_from_slide.index) > 0
        print(f"HAS COORDS {has_coords}")
        if has_coords and not (tile_from_scratch):
            coords = coords_from_slide[["coords_x", "coords_y"]].to_numpy()
            dataset_from_coords = DatasetFromCoords(coords, sp, transform=transform)
            dataloader = DataLoader(
                dataset_from_coords,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers_torch,
                drop_last=False,
            )
        else:
            slide = Slide(sp, "./tmp", use_largeimage=True)
            dataset_from_slide = SlideDataset(
                grid_tiles_extractor, slide, transform=transform
            )
            # We cannot use multiprocessing per slide
            dataloader = DataLoader(
                dataset_from_slide,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
                num_workers=0,
                drop_last=False,
            )

        print(
            "Extracting tiles and forwarding them with ResNet50"
            f" by batch of {batch_size}"
        )
        import time

        t1 = time.time()
        with torch.inference_mode():
            features = np.zeros((0, 2048)).astype("float32")
            coords = np.zeros((0, 2)).astype("int64")
            for i, (batch_images, batch_coords) in enumerate(iter(dataloader)):
                if torch.cuda.is_available():
                    batch_images = batch_images.cuda()
                features = np.concatenate(
                    (features, net(batch_images).detach().cpu().numpy()), axis=0
                )
                coords = np.concatenate((coords, batch_coords.numpy()))
        t2 = time.time()
        num_workers_displayed = (
            num_workers_torch if (has_coords and not (tile_from_scratch)) else 0
        )
        print(
            f"Slide {slide_name} extraction lasted"
            f" {t2-t1}s with has_coords={has_coords}"
            f" (num_workers {num_workers_displayed})"
            f" found {coords.shape[0]} tiles"
        )

        if not (has_coords):
            # We store the coordinates information to speed up
            # the extraction process via multiprocessing
            num_tiles_on_slide = coords.shape[0]
            for i in range(num_tiles_on_slide):
                df_predecessor.append(
                    {
                        "slide_name": slide_name.lower(),
                        "coords_x": float(coords[i, 0]),
                        "coords_y": float(coords[i, 1]),
                    }
                )
            save_dict_to_csv(df_predecessor, path_to_coords_file)

        # Saving features on dict
        np.save(path_to_features, features)

    write_value_in_config(config_file, "preprocessing_complete", True)

    if args.remove_big_tiff:
        print("Removing all slides")
        for slide in slides_paths:
            os.remove(slide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        help="What batch-size to use for inference.",
        default=64,
    )
    parser.add_argument(
        "--num-workers-torch",
        type=int,
        help="The number of torch workers for dataset with precomputed coords.",
        default=10,
    )
    parser.add_argument(
        "--tile-from-scratch",
        action="store_true",
        help="With this option we re-extract the matter from the slides.",
    )
    parser.add_argument(
        "--remove-big-tiff",
        action="store_true",
        help="Whether or not to remove the original slides images that take \
            up to 800G, after computing the features using them.",
    )

    args = parser.parse_args()
    main(
        args.batch_size,
        args.num_workers_torch,
        args.tile_from_scratch,
        args.remove_big_tiff,
    )
