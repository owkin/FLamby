import argparse
import os
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import yaml
from histolab.slide import Slide
from histolab.tiler import GridTiler
from skimage import io
from torch.nn import Identity
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, images_paths, transform=None):
        self.images_paths = images_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images_paths[idx]
        # histolab pngs have a transparency channel
        sample = io.imread(img_name)[:, :, :3]
        if self.transform:
            sample = self.transform(sample)
        return sample


def main(batch_size, num_workers_torch, remove_big_tiff):
    """Function tiling slides that have been downloaded using download.py.

    Parameters
    ----------
    batch_size : int
        The number of images to use for batched inference in pytorch.
    num_workers_torch : int
        The number of workers to use for parallel data loading to prefetch the
        next batches.
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

    with open(config_file, "r") as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)

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
        tile_size=(256, 256),
        level=0,
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
        if os.path.exists(path_to_features):
            continue
        print(f"Tiling slide {sp}")
        with tempfile.TemporaryDirectory() as tmpdirname:
            print(f"Extracting matter tiles images in temporary directory {tmpdirname}")
            slide = Slide(sp, tmpdirname, use_largeimage=True)
            grid_tiles_extractor.extract(slide)
            print("Images extracted")
            images_list = glob(os.path.join(tmpdirname, "**", "*.png"))
            dataset_from_slide = ImageDataset(images_list, transform=transform)
            dataloader = DataLoader(
                dataset_from_slide,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers_torch,
            )
            print(f"Forwarding extracted tiles with ResNet50 by batch of {batch_size}")
            with torch.inference_mode():
                features = np.zeros((len(images_list), 2048)).astype("float32")
                for i, batch in enumerate(iter(dataloader)):
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    features[start_idx:end_idx] = net(batch).detach().cpu().numpy()
        np.save(path_to_features, features)

    dict["preprocessing_complete"] = True
    with open(config_file, "w") as file:
        yaml.dump(dict, file)
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
        help="How many workers to use for efficient data handling.",
        default=10,
    )
    parser.add_argument(
        "--remove-big-tiff",
        action="store_true",
        help="Whether or not to remove the original slides images that take \
            up to 800G, after computing the features using them.",
    )

    args = parser.parse_args()
    main(args.batch_size, args.num_workers_torch, args.remove_big_tiff)
