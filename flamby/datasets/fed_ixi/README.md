# IXI Dataset

## Data Citation and License

The IXI dataset is made available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use the IXI data please acknowledge the source of the IXI data, e.g. the following website: https://brain-development.org/ixi-dataset/

IXI Tiny is derived from the same source. Acknowledge the following reference on TorchIO : https://torchio.readthedocs.io/datasets.html#ixitiny

## Ethics
The dataset website does not provide any information regarding data collection ethics.
However, the original dataset was collected as part of the
IXI - Information eXtraction from Images (EPSRC GR/S21533/02) project,
and thus funded by [UK Research and Innovation (UKRI)](https://www.ukri.org/).
As part of its [terms and conditions](https://www.ukri.org/wp-content/uploads/2022/04/UKRI-050422-FullEconomicCostingGrantTermsConditions-Apr2022.pdf),
the UKRI demands that all funded
projects are "carried out in accordance with all applicable ethical, legal and
regulatory requirements" (RGC 2.2).


## Publication Citation

Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat]. 2020. https://doi.org/10.48550/arXiv.2003.04696

## Introduction

This repository highlights **IXI** (*Information eXtraction from Images*), a medical dataset focusing on brain images through Structural magnetic resonance imaging (MRI), a non-invasive technique for examining the anatomy and pathology of the brain.

In the same register, we highlight a particular dataset called **IXI Tiny**, which is composed of preprocessed images from the standard **IXI** dataset. The idea behind the use of this dataset is to take advantage of its lightness, as well as the labels it directly provides so it allows us to handle an interesting segmentation task.

We have chosen to give some insight about the standard dataset, although we will focus on the lighter one, **IXI Tiny**, as a starting point. Note that the requirements for the standard IXI dataloader have been implemented, but for a matter of clarity, we will limit ourselves to the code of IXI tiny on this part of the repository. Of course, standard IXI could be added in the near future on a separate place.

## Standard IXI Dataset

### Overview

The **IXI** dataset contains “nearly 600 MR images from normal, healthy subjects”, including “T1, T2 and PD-weighted images, MRA images and Diffusion-weighted images (15 directions)”.

The dataset contains data from three different hospitals in London :
- Hammersmith Hospital using a Philips 3T system ([details of scanner parameters](http://wp.doc.ic.ac.uk/brain-development/scanner-philips-medical-systems-intera-3t/)).
- Guy’s Hospital using a Philips 1.5T system ([details of scanner parameters](http://wp.doc.ic.ac.uk/brain-development/scanner-philips-medical-systems-gyroscan-intera-1-5t/)).
- Institute of Psychiatry using a GE 1.5T system (details of the scan parameters not available at the moment).

For information, here is the respective size of the different archives:

| Modality | Size |
| :------: | ------ |
| T1 | 4.51G |
| T2 | 3.59G |
| PD | 3.79G |
| MRA | 11.5G |
| DTI | 3.98G |

**Total size**: 27.37G

Datapoints inside the different archives (our 5 modalities) follow this naming convention:

**IXI**[*patient id*]**-**[*hospital name*]**-**[*id*]**-**[*modality*]**.nii.gz**

These files contain images in **NIFTI** format.

## IXI Tiny Dataset

### Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Dataset contains data from three different hospitals in London focusing on brain images through MRI.
| Dataset size       | 444 MB
| Centers            | 3 centers - Guys (Guy’s Hospital), HH (Hammersmith Hospital), IOP (Institute of Psychiatry).
| Records per center | Guys: 249/62, HH: 145/36, IOP: 59/15 (train/test).
| Inputs shape       | Image of shape (1, 48, 60, 48).
| Targets shape      | Image of shape (2, 48, 60, 48).
| Total nb of points | 566.
| Task               | Segmentation.

### Overview

**IXI Tiny** relies on **IXI**, a publicly available dataset of almost 600 subjects. This lighter version made by [TorchIO](https://torchio.readthedocs.io/datasets.html#ixitiny) is focusing on 566 T1-weighted brain MR images and comes with a set of corresponding labels (brain segmentations).

To produce the labels, ROBEX, an automatic whole-brain extraction tool for T1-weighted MRI data has been used.
Affine registration, which is a necessary prerequisite for many image processing tasks, has been performed using [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) putting all the brain images onto a common reference space (MNI template). An orientation tweak has finally been made with [ITK](https://itk.org/).

Volumes have a dimension of 83 x 44 x 55 voxels (compared to 256 x 256 x 140 in the standard dataset).

The total size of this tiny dataset is 444 MB.

The structure of the archive containing the dataset has been modified, making it adapted for particular cases of use where each subject is represented by a directory containing all the modalities associated with them.

E.g.
```
IXI_sample
│
└───IXI002-Guys-0828
│   │
│   └───label
│   │   │   IXI002-Guys-0828_label.nii.gz
│   │
│   └───T1
│   │   │   IXI002-Guys-0828_image.nii.gz
│   │
│   └───T2
│   │   │   IXI002-Guys-0828_image.nii.gz
│   │
│   └───...
│
└───IXI012-HH-1211
│   │
│   └───label
│   │   │   IXI012-HH-1211_label.nii.gz
│   │
│   └───T1
│   │   │   IXI012-HH-1211_image.nii.gz
│   │
│   └───T2
│   │   │   IXI012-HH-1211_image.nii.gz
│   │
│   └───...
│
│
└───...

```

### Download

To download the data, simply run the following commands:

1. cd into `dataset_creation_scripts` folder: `cd dataset_creation_scripts`

2. run the download script: `python download.py -o IXI-Dataset`

### Utilization

Once the dataset is ready for use, you can load it the following way:
```python
from flamby.datasets.fed_ixi import FedIXITiny

# To load the first center as a pytorch dataset
center0 = FedIXITiny(transform=None, center=0, train=True, pooled=False)
# To load the second center as a pytorch dataset
center1 = FedIXITiny(transform=None, center=1, train=True, pooled=False).

# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()
```
The following arguments can be passed to FedIXITiny:
- 'transform' allows to perform a specific transformation on the brain images (e. g. with the MONAI library).
- 'center' allows center indexation, must be in `[0, 1, 2]` or in `['Guys', 'HH', 'IOP']`.
- 'train', whether we want to load the train or test set
- 'pooled' loads data from all the centers (overwriting previous center argument)

More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)


### Benchmarking the baseline on a pooled setting

Once the download is completed and the federated classes are set up, we can benchmark the baseline regarding our prediction task on the pooled dataset: `python benchmark.py`

This will train and test a UNet model (see Prediction task section).

### Prediction task

As a first approach, what we can do with the **IXI Tiny** dataset is to set up a segmentation task using the T1 images:
Create a model which take T1 image as input and predict the binary mask of the brain (label). This process allows us to isolate the brain from the other head components, such as the eyes, skin, and fat.

We will use a UNet model (a kind of convolution neural network architecture with few changes), very popular in biomedical segmentation. UNet is specifically used to perform semantic segmentation, meaning that each voxel of our volume will be classified. We can also refer this task as a dense prediction.

Here are some information to give an insight into how the prediction is set up:

**Loss and metric formulas** : We use the DICE loss calculated the following way : `DICE_loss = 1 - DICE_score = 1 - (2 * TP / (2 * TP + FP + FN + epsilon))` and take the DICE score for the performance metric.

**UNet final hyperparameters** :
```python
in_channels: int = 1,
out_classes: int = 2,
dimensions: int = 3,
num_encoding_blocks: int = 3,
out_channels_first_layer: int = 8,
normalization: Optional[str] = "batch",
pooling_type: str = "max",
upsampling_type: str = "linear",
preactivation: bool = False,
residual: bool = False,
padding: int = 1,
padding_mode: str = "zeros",
activation: Optional[str] = "PReLU",
initial_dilation: Optional[int] = None,
dropout: float = 0,
monte_carlo_dropout: float = 0
```

**Batch size** : 2

**Learning rate** : 0.001 (AdamW optimizer)
