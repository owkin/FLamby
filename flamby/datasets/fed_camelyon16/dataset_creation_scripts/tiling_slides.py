import argparse
import os
import sys
import tempfile
from glob import glob
from pathlib import Path
import itertools
import numpy as np
import torch
import torchvision.models as models
from histolab.slide import Slide
from histolab.tiler import GridTiler
from skimage import io
from torch.nn import Identity
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from flamby.utils import read_config, write_value_in_config


class SlideDataset(IterableDataset):
    def __init__(self, grid_tiles_extractor, slide, transform=None):
        self.transform = transform
        self.it = grid_tiles_extractor._tiles_generator(slide)
    def __iter__(self):
        for tile in self.it:
            im = self.transform(tile[0].image.convert("RGB"))
            coords = tile[1]._asdict()
            coords_list = []
            for k, v in coords.items():
                coords_list.append(v)
            coords = torch.Tensor(coords_list)
            yield im, coords


def main(batch_size, remove_big_tiff):
    """Function tiling slides that have been downloaded using download.py.

    Parameters
    ----------
    batch_size : int
        The number of images to use for batched inference in pytorch.

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
    # Tiling all downloaded slides with provided model
    for sp in tqdm(slides_paths):
        slide_name = os.path.basename(sp)

        path_to_features = os.path.join(slides_dir, slide_name + ".npy")
        if not(os.path.exists(path_to_features)):
            continue
        print(f"Tiling slide {sp}")
        slide = Slide(sp, "./tmp", use_largeimage=True)
        # We use the matter detection from histolab to tile the dataset
        dataset_from_slide = SlideDataset(grid_tiles_extractor, slide, transform=transform)
        dataloader = DataLoader(
            dataset_from_slide,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            drop_last=False,
        )
        print(f"Extracting tiles and forwarding them with ResNet50 by batch of {batch_size}")
        with torch.inference_mode():
            features = np.zeros((0, 2048)).astype("float32")
            for i, (batch_images, batch_coords) in enumerate(iter(dataloader)):
                if torch.cuda.is_available():
                    batch_images = batch_images.cuda()
                features = np.concatenate((features, net(batch_images).detach().cpu().numpy()), axis=0)

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
        "--remove-big-tiff",
        action="store_true",
        help="Whether or not to remove the original slides images that take \
            up to 800G, after computing the features using them.",
    )

    args = parser.parse_args()
    main(args.batch_size, args.remove_big_tiff)
