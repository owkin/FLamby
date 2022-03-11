import os
import glob
from PIL import Image
from joblib import Parallel, delayed
import argparse
from fastai.vision.all import *
from color_constancy import color_constancy
import albumentations
from tqdm import tqdm


def pad_and_resize(path, output_path, sz: tuple):
    fn = os.path.basename(path)  
    im = PILImage(PILImage.create(path))
    resize_method = ResizeMethod.Pad
    resize = Resize(sz[0], method=resize_method)
    im = resize(im, split_idx=0)
    im.save(os.path.join(output_path, fn))


def resize_and_save(path, output_path, sz:tuple = (256,256), cc=False):
    fn = os.path.basename(path)  
    im = np.array(Image.open(path))
    aug = albumentations.Resize(*sz)
    im = aug(image=im)['image']
    if cc: 
        im = color_constancy(np.array(im))
        im = Image.fromarray(im)
    im.save(os.path.join(output_path, fn))


def resize_and_mantain(path, output_path, sz: tuple, args):
    # mantain aspect ratio and shorter edge of resized image is 600px
    # from research paper `https://isic-challenge-stade.s3.amazonaws.com/9e2e7c9c-480c-48dc-a452-c1dd577cc2b2/ISIC2019-paper-0816.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=nQCPd%2F88z0rftMkXdxYG97Nau4Y%3D&Expires=1592222403`
    fn = os.path.basename(path)  
    img = Image.open(path)
    size = sz[0]  
    old_size = img.size
    ratio = float(size)/min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.BILINEAR)
    if args.cc: 
        img = color_constancy(np.array(img))
        img = Image.fromarray(img)
    img.save(os.path.join(output_path, fn))


def resize_min_wh(path, output_path):
    fn = os.path.basename(path)
    img =Image.open(path)  
    npimg = np.array(img)
    crop_sz =min(img.size)
    npimg = albumentations.CenterCrop(crop_sz, crop_sz)(image=npimg)['image']
    img = Image.fromarray(npimg)
    img.save(os.path.join(output_path, fn))    


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", 
        default=None, 
        type=str, 
        required=True, 
        help="Input folder where images exist."
    )
    parser.add_argument(
        "--output_folder", 
        default=None, 
        type=str, 
        required=True, 
        help="Output folder for images."
    )
    parser.add_argument("--mantain_aspect_ratio", action='store_true', default=False, help="Whether to mantain aspect ratio of images.")
    parser.add_argument("--pad_resize", default=False, type=bool, help="Whether to pad and resize images.")
    parser.add_argument("--sz", default=256, type=int, help="Whether to pad and resize images.")
    parser.add_argument("--cc", default=False, action='store_true', help="Whether to do color constancy to the images.")
    parser.add_argument("--resize_and_save", default=False, action='store_true')
    parser.add_argument("--centercrop", default=False, action='store_true')

    args = parser.parse_args()

    if args.sz: 
        print("Images will be resized to {}".format(args.sz))
        args.sz= int(args.sz)

    images = glob.glob(os.path.join(args.input_folder, '*.jpg'))
    if (not args.mantain_aspect_ratio) and args.pad_resize:
        print("Adding padding to images if needed and resizing images to square of side {}px.".format(args.sz))
        Parallel(n_jobs=16)(
            delayed(pad_and_resize)(i, args.output_folder, (args.sz, args.sz)) for i in tqdm(images))
    elif args.resize_and_save:
        print("Resizing and saving images to size {}".format(args.sz))
        Parallel(n_jobs=32)(
            delayed(resize_and_save)(i, args.output_folder, (args.sz, args.sz), args.cc) for i in tqdm(images))
    elif args.centercrop:
        print("Will crop min(h,w) center and resize.")
        Parallel(n_jobs=32)(
            delayed(resize_min_wh)(i, args.output_folder) for i in tqdm(images))
    else:
        print("Resizing images to mantain aspect ratio in a way that the shorter side is {}px but images are rectangular.".format(args.sz))
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        print(args)
        Parallel(n_jobs=32)(
            delayed(resize_and_mantain)(i, args.output_folder, (args.sz, args.sz), args) for i in tqdm(images))