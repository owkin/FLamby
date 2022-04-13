import glob
import os
import sys
from pathlib import Path

import numpy as np
from color_constancy import color_constancy
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from flamby.utils import read_config, write_value_in_config

dir = str(Path(os.path.realpath(__file__)).parent.resolve())
config_file = os.path.join(dir, "dataset_location.yaml")
dict = read_config(config_file)
if not (dict["download_complete"]):
    raise ValueError("Download incomplete. Please relaunch the download script")
if dict["preprocessing_complete"]:
    print("You have already ran the preprocessing, aborting.")
    sys.exit()
input_path = dict["dataset_path"]


dic = {
    "inputs": "ISIC_2019_Training_Input",
    "inputs_preprocessed": "ISIC_2019_Training_Input_preprocessed",
}
input_folder = os.path.join(input_path, dic["inputs"])
output_folder = os.path.join(input_path, dic["inputs_preprocessed"])
os.makedirs(output_folder, exist_ok=True)


def resize_and_maintain(path, output_path, sz: tuple, cc):
    """Preprocessing of images
    Mantains aspect ratio fo input image. Possibility to add color constancy.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    path : path to input image
    output_path : path to output image
    sz : tuple, shorter edge of resized image is sz[0]
    cc : color constancy is added if True
    """
    fn = os.path.basename(path)
    img = Image.open(path)
    size = sz[0]
    old_size = img.size
    ratio = float(size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.BILINEAR)
    if cc:
        img = color_constancy(np.array(img))
        img = Image.fromarray(img)
    img.save(os.path.join(output_path, fn))


if __name__ == "__main__":

    sz = 224

    images = glob.glob(os.path.join(input_folder, "*.jpg"))

    print(
        "Resizing images to mantain aspect ratio in a way that the shorter side"
        " is {}px but images are rectangular.".format(sz)
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cc = True
    sz = 224

    Parallel(n_jobs=32)(
        delayed(resize_and_maintain)(i, output_folder, (sz, sz), cc)
        for i in tqdm(images)
    )

    write_value_in_config(config_file, "preprocessing_complete", True)
