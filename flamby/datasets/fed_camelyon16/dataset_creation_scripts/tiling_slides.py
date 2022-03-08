import argparse
import os
import tempfile
from glob import glob

import numpy as np
import torch
import torchvision.models as models
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


def main(slides_dir, features_dir, batch_size):
    """This function tile the matter on the slides located into slides_dir using
    a resnet50 pretrained on IMAGENET with a batch size of batch_size for inference.
    The features produced are in numpy format and located in features_dir.

    Parameters
    ----------
    slides_dir : str
        The folder where the slides are. We expect .tif slides as in Camelyon16.
    features_dir : str
        The folder where to store the produced features.
    batch_size : int
        The batch size for the resnet50's inference, depends on the capacity of
        the distant machine.
    """
    os.makedirs(features_dir, exist_ok=True)
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
    # Probably unnecessary still it feels safer
    for param in net.parameters():
        param.requires_grad = False
    net = net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    net.fc = Identity()
    # IMAGENET preprocessing of images scaled between 0. and 1. with ToTensor
    transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    # Tiling all downloaded slides with provided model
    for sp in tqdm(slides_paths):
        slide_name = os.path.basename(sp)
        path_to_features = os.path.join(features_dir, slide_name + ".npy")
        if os.path.exists(path_to_features):
            continue
        with tempfile.TemporaryDirectory() as tmpdirname:
            slide = Slide(sp, tmpdirname, use_largeimage=True)
            grid_tiles_extractor.extract(slide)
            images_list = glob(os.path.join(tmpdirname, "**", "*.png"))
            dataset_from_slide = ImageDataset(images_list, transform=transform)
            dataloader = DataLoader(
                dataset_from_slide,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )
            with torch.inference_mode():
                features = np.zeros((len(images_list), 2048)).astype("float32")
                for i, batch in enumerate(iter(dataloader)):
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    features[start_idx:end_idx] = net(batch).detach().cpu().numpy()
        np.save(path_to_features, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slides-dir",
        type=str,
        help="From where the slides have been extracted.",
        required=True,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        help="Where to store the features extracted from tiling the slides.",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="What batch-size to use for inference.",
        default=64,
    )

    args = parser.parse_args()
    main(args.slides_dir, args.output_folder, args.batch_size)
