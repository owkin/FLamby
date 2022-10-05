# LIDC-IDRI

This dataset comes from the [TCIA GDC data portal](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc).

## Terms of use
### License
We do not own the copyright of the data.
As indicated on the [dataset's data usage policy](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc),
users of this data must abide by the [TCIA Data Usage Policy](https://wiki.cancerimagingarchive.net/x/c4hF) and the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/),
and give proper attribution to the original authors (see the next sections).
### Data Citation

Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

### Publication Citation

Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

### TCIA Citation

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7


### Ethics
As per the [terms of use of TCIA](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions), data was anonymized prior to being submitted and
>users must agree not to generate and use information in a manner that could allow the
>identities of research participants to be readily ascertained.

In particular, as stated in the dataset publication, data was anonymized
in each local center before being uploaded to the central repository.

## Dataset description
The table below provides summary information on the dataset.
Please refer to the [original article](https://doi.org/10.1118/1.3528204)
for a more in-depth presentation
of the dataset.

|                   | Dataset description |
| ----------------- | -----------------------------------------------|
| Description       | This is the dataset from LIDC-IDRI study |
| Dataset           | 1009 CT-scans with masks (GE MEDICAL SYSTEMS 661 (530 / 131), SIEMENS 205 (164, 41), TOSHIBA 69 (55 / 14), Philips 74 (59 / 15) |
| Centers           | Manufacturer of the CT-scans (GE MEDICAL SYSTEMS, SIEMENS, TOSHIBA, Philips) |
| Permission        | Public |
| Task              | Segmentation |
| Format            | CT-scans and masks in the `nifti` format |

This dataset was used in the [Luna16 challenge](https://luna16.grand-challenge.org/Home/).
Note that contrary to the challenge, in which slides with thickness larger than 3 mm were removed,
the present dataset contains 1009 scans (almost all scans from
[LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), and 9 scans are removed due to missing slices).

The data is split in a training and testing sets containing 80% and 20% of the available scans respectively.
This split is stratified according to the centers (the manufacturers of the CT-scan apparatus),
in order to preserve the distribution of centers of origin within each split.

Centers (i.e., manufacturers) are encoded using a unique index, as follows: GE MEDICAL SYSTEMS: 0, Philips: 1, SIEMENS: 2, TOSHIBA: 3.


## Downloading and preprocessing

To download the data we will use the [official TCIA python client](https://github.com/nadirsaghar/TCIA-REST-API-Client/blob/master/tcia-rest-client-python/src/tciaclient.py).

Create a directory for the dataset(``MyDirectory``).
Make sure you have enough space (150G), ``cd`` to ``dataset_creation_scipts`` and run:
```
python download_ct_scans.py -o MyDirectory
```

This may take a few hours, depending on your download bandwidth and your machine.

The ``download_ct_scans.py`` script will download DICOM files corresponding to the CT-scans as well as XML files
containing annotations and segmentations from radiologists.

DICOM files will then be converted to the ``nifti`` format. Each ``nifti`` file contains a 3D image of variable size (roughly (380 x 380 x 380 on average)).

Nifti (.nii.gz) files can be conveniently handled using the [nibabel](https://nipy.org/nibabel/) package.

### Folder arborescence

The data is stored in the following way:

```
MyDirectory
│
└───LIDC-XML-only
│   │
│   └───tcia-lidc-xml
│       │
│       └─── ...
│       │     │ 158.xml
│       │     │ ...
│       │
│       └─── ...
│
└───1.1.3.6.1.4.1.14519.5.2.1.6279.6001.[SeriesInstanceUID]
│   │   mask_consensus.nii.gz
│   │   mask.nii.gz
│   │   patient.nii.gz
│
│
└───...

```

- ``LIDC-XML-only``: folder with xml files containing radiologist annotations/segmentations.
- ``1.1.3.6.1.4.1.14519.5.2.1.6279.6001.[SeriesInstanceUID]``: one folder per ct scan. Contains:
  - ``patient.nii.gz``: nifti file containing the ct scan.
  - ``mask.nii.gz``: nifti file containing all annotations from radiologists.
  - ``mask_consensus.nii.gz``: nifti file containing the average annotation for radiologists. Used as ground truth for segmentation.


### Troubleshooting

While running ``download_ct_scans.py``, it may happen that the TCIA client stalls and that files stop being downloaded.
In that case, you should kill the python process and run ``download_ct_scans.py`` again, with the same arguments.
Files that were correctly downloaded will not be downloaded again.

## Using the dataset

Now that the dataset is ready for use you can load it using the low or high-level API
by doing:
```python
from flamby.datasets.fed_lidc_idri import FedLidcIdri

# To load the first center as a pytorch dataset
center0 = FedLidcIdri(center=0, train=True)
# To load the second center as a pytorch dataset
center1 = FedLidcIdri(center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
from torch.utils.data import DataLoader as dl

X, y = iter(dl(center0, batch_size=2, shuffle=True, num_workers=0)).next()
```
More informations on how to train model and handle flamby datasets in general are available in the [Getting Started section](../../../Quickstart.md)


## Baseline model

The baseline model is a V-Net (see [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
](https://arxiv.org/abs/1606.04797)). It is trained by minimizing the DICE loss. Since lung scans are too large to fit in memory,
patches containing positive voxels are randomly sampled and fed to the network during training. At test time, however, the whole
image is processed. The current implementation assumes access to a GPU able to process 2 (128 x 128 x 128) patches at a time.
The code was tested on a Titan X (Pascal) GPU.
