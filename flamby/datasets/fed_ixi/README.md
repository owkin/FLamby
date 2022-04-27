# IXI Dataset

### Data Citation

The IXI dataset is made available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use the IXI data please acknowledge the source of the IXI data, e.g. the following website: https://brain-development.org/ixi-dataset/

IXI Tiny is derived from the same source. Acknowledge the following reference on TorchIO : https://torchio.readthedocs.io/datasets.html#ixitiny

### Publication Citation

Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat]. 2020. https://doi.org/10.48550/arXiv.2003.04696

## Introduction

This repository highlights **IXI** (*Information eXtraction from Images*), a medical dataset focusing on brain images through Structural magnetic resonance imaging (MRI), a non-invasive technique for examining the anatomy and pathology of the brain.

In the same register, we highlight a particular dataset called **IXI Tiny**, which is composed of preprocessed images from the standard **IXI** dataset. The idea behind the use of this dataset is to take advantage of its lightness, as well as the labels it directly provides so it allows us to handle an interesting segmentation task.

We have chosen to include the standard dataset, fulfilling the requirements for the dataloader implementation, although we will focus on the lighter one, **IXI Tiny**, as a starting point.

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

### Overview

**IXI Tiny** relies on **IXI**, a publicly available dataset of almost 600 subjects. This lighter version made by [TorchIO](https://torchio.readthedocs.io/datasets.html#ixitiny) is focusing on 566 T1-weighted brain MR images and comes with a set of corresponding labels (brain segmentations).

To produce the labels, ROBEX, an automatic whole-brain extraction tool for T1-weighted MRI data has been used.
Affine registration, which is a necessary prerequisite for many image processing tasks, has been performed using [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) putting all the brain images onto a common reference space (MNI template). An orientation tweak has finally been made with [ITK](https://itk.org/).

The total size of this tiny dataset is 444 MB.

The structure of the archive containing the dataset has been modified by M. Lorenzi, making it adapted for particular cases of use where each subject is represented by a directory containing all the modalities associated with them.

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


### Utilization

As a first approach, what we can do with the **IXI Tiny** dataset is to set up a segmentation task using the T1 images:
Create a model which take T1 image as input and predict the binary mask of the brain (label). This process allows us to isolate the brain from the other head components, such as the eyes, skin, and fat.