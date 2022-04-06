# Thank you to [Aman Arora](https://github.com/amaarora) for his
# [implementation](https://github.com/amaarora/melonama)
# We reused his whole preprocessing pipeline.

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

path_to_config_file = str(Path(os.path.realpath(__file__)).parent.resolve())
config_file = os.path.join(path_to_config_file, "dataset_location.yaml")
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
    # mantain aspect ratio and shorter edge of resized image is 600px
    # from research paper `https://isic-challenge-stade.s3.amazonaws.com/'
    # '9e2e7c9c-480c-48dc-a452-c1dd577cc2b2/ISIC2019-paper-0816.pdf'
    # '?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=nQCPd%2F88z0rftMkXdxYG97'
    # 'Nau4Y%3D&Expires=1592222403`
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
