# SOPHIA

Signal Optimisation and Processing for whole Heart Image Analysis.

Used by Nicholas Cheverie in the process of obtaining his Master of Science.

## About SOPHIA

This repository contains the code used during the analysis of ratiometric optical maps obtained by Nicholas Cheverie during the course of his thesis. It consists of:

* single_channel_analysis.py: this script analyses the voltage and calcium layers within the recording.
* calc_ecc_rcc_dual.py: this script gets the values obtained from dual_channel_analysis.py to calculate the ECC and RRC values.

It used voltage and calcium images that were denoised by [SUPPORT](https://github.com/NICALab/SUPPORT). These images were then registered using the Gravity Registration method offered by [PIRT](https://github.com/almarklein/pirt). 

Additional libraries used include:

* pandas
* numpy
* scipy
* uncertainties
* tqdm
* [pybaselines](https://github.com/derb12/pybaselines)
* [tifffile](https://github.com/cgohlke/tifffile)
* [sif_parser](https://github.com/fujiisoup/sif_parser)

## How to Use

single_channel_analysis.py requires:

* path_to_denoised_and_registered_imgs: folder containing images which have been denoised by SUPPORT and registered with PIRT's Gravity Registration images. Images should be in tif format
* path_to_lookup_table: an Excel file containing the columns `voltage_ch`, `calcium_ch`, `recording_nchannels`, `framerate_vol`, and `framerate_cal`.
* path_for_processed_excel: the path where the Excel file containing the results, in addition to a pickle file containing the results, would be written.

After the Excel result file has been created, the ECC and RRC can be calculated from the parameters within the Excel result file. Ergo, the script calc_ecc_rrc_dual.py requires:

* path_to_lookup_table: the Excel file created by single_channel_analysis.py.
* path_to_result_excel: the path where the Excel file containing the results should be written.

