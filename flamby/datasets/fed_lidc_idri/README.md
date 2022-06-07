# LIDC-IDRI

This dataset comes from the [TCIA GDC data portal](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc).
We do not own the copyright of the data, everyone using this dataset should abide by [its licence](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc) and give proper attribution to the original authors.
We therefore give credit here to:
### Data Citation

Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

### Publication Citation

Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

### TCIA Citation

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7

Additionally here we preprocess the dataset to create nifti files and retrieve the per-center metadata.

## Dataset description

|                   | Dataset description |
| ----------------- | -----------------------------------------------|
| Description       | This is the dataset from LIDC-IDRI study |
| Dataset           | 1018 CT-scans with masks (GE MEDICAL SYSTEMS 670, SIEMENS 205, TOSHIBA 69, Philips 74) |
| Centers           | Manufacturer of the CT-scans (GE MEDICAL SYSTEMS, SIEMENS, TOSHIBA, Philips) |
| Permission        | Public |
| Task              | Segmentation |
| Format            | CT-scans and masks in the `nifti` format |

This dataset was used in the [Luna16 challenge](https://luna16.grand-challenge.org/Home/).
Note that contrary to the challenge, in which slides with thickness larger than 3 mm were removed, 
the present dataset contains all 1018 slides from [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

The data is split in a training and testing sets containing 80% and 20% of the available scans respectively.
This split is stratified according to the centers (the manufacturers of the CT-scan apparatus),
in order to preserve the distribution of centers of origin within each split. 


### Downloading and preprocessing

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

## Baseline model

The baseline model is a V-Net (see [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
](https://arxiv.org/abs/1606.04797)). It is trained by minimizing the DICE loss. Since lung scans are too large to fit in memory,
patches containing positive voxels are randomly sampled and fed to the network during training. At test time, however, the whole 
image is processed. The current implementation assumes access to a GPU able to process 2 (128 x 128 x 128) patches at a time.
The code was tested on a Titan X (Pascal) GPU.
