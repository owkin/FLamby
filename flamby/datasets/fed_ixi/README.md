# IXI Dataset

## Overview

The Information eXtraction from Images (IXI) dataset contains “nearly 600 MR images from normal,
healthy subjects”, including “T1, T2 and PD-weighted images, MRA images and Diffusion-weighted images (15 directions)”.

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

## Utilization

As a first approach, what we can do is to set up a segmentation task using the T1 images:
Create a model which take T1 image as input and predict the binary mask of the brain (label). This process allows us to isolate the brain from the other head components, such as the eyes, skin, and fat.
To produce the labels, we can use an automatic whole-brain extraction tool for T1-weighted MRI data like ROBEX.

[TBD]

## Source

This dataset is made available under the Creative Commons CC BY-SA 3.0 license.
IXI website : https://brain-development.org/ixi-dataset/