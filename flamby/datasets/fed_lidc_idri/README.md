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

|                   | Dataset description 
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from LIDC-IDRI study
| Dataset           | **1018 CT-scans** with masks (GE MEDICAL SYSTEMS 670, SIEMENS 205, TOSHIBA 69, Philips 74)
| Centers           | Manufacturer of the CT-scans (GE MEDICAL SYSTEMS, SIEMENS, TOSHIBA, Philips)
| Permission        | Public
| Task              | Segmentation
| Format            | CT-scans and masks in the `nifti` format


To download the data we will use the [official TCIA python client](https://github.com/nadirsaghar/TCIA-REST-API-Client/blob/master/tcia-rest-client-python/src/tciaclient.py).

Make sure you have enough space (?G) and run:
```
pythoon download_ct_scans.py
```

