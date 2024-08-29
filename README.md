# SOPHIA

Signal Optimisation and Processing for whole Heart Image Analysis.

Used by Nicholas Cheverie in the process of obtaining his Master of Science.

## About SOPHIA

This repository contains the code used during the analysis of ratiometric optical maps obtained by Nicholas Cheverie during the course of his thesis. It consists of:

* dual_channel_analysis.py: this script analyses the voltage and calcium layers within the recording.
* calc_ecc_rcc_dual.py: this script gets the values obtained from dual_channel_analysis.py to calculate the ECC and RRC values.

It used voltage and calcium images that were denoised by [SUPPORT] (https://github.com/NICALab/SUPPORT). These images were then registered using the Gravity Registration method offered by [PIRT] (https://github.com/almarklein/pirt). 

Additional libraries used include:

* pandas
* numpy
* scipy
* uncertainties
* tqdm
* [pybaselines] (https://github.com/derb12/pybaselines)
* [tifffile] (https://github.com/cgohlke/tifffile)
* [sif_parser] (https://github.com/fujiisoup/sif_parser)


