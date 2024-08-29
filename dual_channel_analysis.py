import os.path

import numpy
import pandas
import tifffile


def analyse_voltage_and_calcium_layer(calcium_layer: numpy.ndarray, voltage_layer: numpy.ndarray,
                                      data_dataframe: pandas.DataFrame, ca_ch_name: str, vo_ch_name: str,
                                      voltage_framerate: float, calcium_framerate: float, roi_params: dict,
                                      debug_plot: bool, debug_plot_savepath: str):
    import numpy
    import uncertainties

    import utils.create_evenly_spaced_rois
    import utils.calcium_roi_analysis
    import utils.voltage_roi_analysis

    # Create ROIs
    calcium_roi_dict = utils.create_evenly_spaced_rois.create_rois_in_centre(calcium_layer,
                                                                             roi_size=(roi_params['roi_size'],
                                                                                       roi_params['roi_size']),
                                                                             num_rois=roi_params['number_of_rois'],
                                                                             margin_percentage=roi_params[
                                                                                 'margin_percentage'])

    voltage_roi_dict = utils.create_evenly_spaced_rois.create_rois_in_centre(voltage_layer,
                                                                             roi_size=(roi_params['roi_size'],
                                                                                       roi_params['roi_size']),
                                                                             num_rois=roi_params['number_of_rois'],
                                                                             margin_percentage=roi_params[
                                                                                 'margin_percentage'])

    # For calcium, calculate CaTD50, CaTD80, max df/dt
    for cal_keys in calcium_roi_dict:
        roi = calcium_roi_dict[cal_keys]['roi']

        # Write to dataframe
        data_dataframe.loc[data_dataframe['calcium_ch'] == ca_ch_name, f"{cal_keys}_cal_x_start_end"] = [calcium_roi_dict[cal_keys]['x_start_end']]
        data_dataframe[ca_ch_name][f"{cal_keys}_cal_x_start_end"] = calcium_roi_dict[cal_keys]['x_start_end']
        data_dataframe[f"{cal_keys}_cal_y_start_end"] = calcium_roi_dict[cal_keys]['y_start_end']

        # Calculate CaTD50
        CaTD80_in_seconds, CaTD50_in_seconds, cal_max_dfdt_in_seconds = utils.calcium_roi_analysis.analyse_single_roi(
            roi=roi,
            framerate=calcium_framerate,
            debug_plot=False)

        # Get values Calcium channel
        roi_CaTD80_mean = numpy.nanmean(CaTD80_in_seconds, axis=None)
        roi_CaTD50_mean = numpy.nanmean(CaTD50_in_seconds, axis=None)
        roi_Ca_max_dfdt_mean = numpy.nanmean(cal_max_dfdt_in_seconds, axis=None)

        roi_CaTD80_std = numpy.nanstd(CaTD80_in_seconds, axis=None)
        roi_CaTD50_std = numpy.nanstd(CaTD50_in_seconds, axis=None)
        roi_Ca_max_dfdt_std = numpy.nanstd(cal_max_dfdt_in_seconds, axis=None)

        # Pack values into an uncertainties float and save into dict
        calcium_roi_dict[cal_keys]['roi_CaTD80'] = uncertainties.ufloat(roi_CaTD80_mean, roi_CaTD80_std, tag='seconds')
        calcium_roi_dict[cal_keys]['roi_CaTD50'] = uncertainties.ufloat(roi_CaTD50_mean, roi_CaTD50_std, tag='seconds')
        calcium_roi_dict[cal_keys]['roi_Ca_max_dfdt'] = uncertainties.ufloat(roi_Ca_max_dfdt_mean, roi_Ca_max_dfdt_std,
                                                                             tag='seconds')

        # Write to dataframe
        data_dataframe[f"{cal_keys}_cal_roi_CaTD80"] = calcium_roi_dict[cal_keys]['roi_CaTD80']
        data_dataframe[f"{cal_keys}_cal_roi_CaTD50"] = calcium_roi_dict[cal_keys]['roi_CaTD50']
        data_dataframe[f"{cal_keys}_cal_roi_Ca_max_dfdt"] = calcium_roi_dict[cal_keys]['roi_Ca_max_dfdt']

    for vol_keys in voltage_roi_dict:
        roi = voltage_roi_dict[vol_keys]['roi']

        # Write to dataframe
        data_dataframe[f"{vol_keys}_cal_x_start_end"] = calcium_roi_dict[vol_keys]['x_start_end']
        data_dataframe[f"{vol_keys}_cal_y_start_end"] = calcium_roi_dict[vol_keys]['y_start_end']

        # Calculate CaTD50
        APD80_in_seconds, APD50_in_seconds, vol_max_dfdt_in_seconds = utils.voltage_roi_analysis.analyse_single_roi(
            roi=roi,
            framerate=voltage_framerate,
            debug_plot=False)

        voltage_roi_dict[vol_keys]['APD80'] = APD80_in_seconds
        voltage_roi_dict[vol_keys]['APD50'] = APD50_in_seconds
        voltage_roi_dict[vol_keys]['vol_max_dfdt'] = vol_max_dfdt_in_seconds

        # Get values Voltage channel
        roi_APD80_mean = numpy.nanmean(APD80_in_seconds, axis=None)
        roi_APD50_mean = numpy.nanmean(APD50_in_seconds, axis=None)
        roi_vol_max_dfdt_mean = numpy.nanmean(vol_max_dfdt_in_seconds, axis=None)

        roi_APD80_std = numpy.nanstd(APD80_in_seconds, axis=None)
        roi_APD50_std = numpy.nanstd(APD50_in_seconds, axis=None)
        roi_vol_max_dfdt_std = numpy.nanstd(vol_max_dfdt_in_seconds, axis=None)

        # Pack values into an uncertainties float and save into dict
        voltage_roi_dict[vol_keys]['roi_APD80'] = uncertainties.ufloat(roi_APD80_mean, roi_APD80_std, tag='seconds')
        voltage_roi_dict[vol_keys]['roi_APD50'] = uncertainties.ufloat(roi_APD50_mean, roi_APD50_std, tag='seconds')
        voltage_roi_dict[vol_keys]['roi_vol_max_dfdt'] = uncertainties.ufloat(roi_vol_max_dfdt_mean,
                                                                              roi_vol_max_dfdt_std, tag='seconds')

        # Write to dataframe
        data_dataframe[f"{vol_keys}_cal_roi_APD80"] = calcium_roi_dict[vol_keys]['roi_APD80']
        data_dataframe[f"{vol_keys}_cal_roi_APD50"] = calcium_roi_dict[vol_keys]['roi_APD50']
        data_dataframe[f"{vol_keys}_cal_roi_vol_max_dfdt"] = calcium_roi_dict[vol_keys]['roi_vol_max_dfdt']

    # Collate voltage and calcium rois
    # Get mean CaTD50 and CaTD80 and cal_max_dfdt
    list_of_roi_CaTD80 = []
    list_of_roi_CaTD50 = []
    list_of_roi_Ca_max_dfdt_mean = []

    for cal_keys in calcium_roi_dict:
        list_of_roi_CaTD80.append(calcium_roi_dict[cal_keys]['roi_CaTD80'])
        list_of_roi_CaTD50.append(calcium_roi_dict[cal_keys]['roi_CaTD50'])
        list_of_roi_Ca_max_dfdt_mean.append(calcium_roi_dict[cal_keys]['roi_Ca_max_dfdt'])

    slice_CaTD80 = numpy.nanmean(list_of_roi_CaTD80)
    slice_CaTD50 = numpy.nanmean(list_of_roi_CaTD50)
    slice_Ca_max_dfdt = numpy.nanmean(list_of_roi_Ca_max_dfdt_mean)

    # Write to dataframe
    data_dataframe[f"Ca_CaTD80"] = slice_CaTD80
    data_dataframe[f"Ca_CaTD50"] = slice_CaTD50
    data_dataframe[f"Ca_max_dfdt"] = slice_Ca_max_dfdt

    # Get mean APD80, APD50 and vol_max_dfdt
    list_of_roi_APD80 = []
    list_of_roi_APD50 = []
    list_of_roi_vol_max_dfdt_mean = []

    for vol_keys in voltage_roi_dict:
        list_of_roi_APD80.append(voltage_roi_dict[vol_keys]['roi_APD80'])
        list_of_roi_APD50.append(voltage_roi_dict[vol_keys]['roi_APD50'])
        list_of_roi_vol_max_dfdt_mean.append(voltage_roi_dict[vol_keys]['roi_vol_max_dfdt'])

    slice_APD80 = numpy.nanmean(list_of_roi_APD80)
    slice_APD50 = numpy.nanmean(list_of_roi_APD50)
    slice_vol_max_dfdt = numpy.nanmean(list_of_roi_vol_max_dfdt_mean)

    # Write to dataframe
    data_dataframe[f"Vol_APD80"] = slice_APD80
    data_dataframe[f"Vol_APD50"] = slice_APD50
    data_dataframe[f"Vol_max_dfdt"] = slice_vol_max_dfdt

    one_frame_duration = 1 / calcium_framerate
    ecc = slice_vol_max_dfdt - slice_Ca_max_dfdt + one_frame_duration
    rrc = ecc + slice_CaTD80 - slice_APD50

    # Write to dataframe
    data_dataframe[f"ecc"] = ecc
    data_dataframe[f"rrc"] = rrc

    return data_dataframe


roi_params = {'roi_size': 30,
              'number_of_rois': 9,
              'margin_percentage': 0.3}

path_to_denoised_and_registered_imgs = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/denoised_and_registered'
path_to_lookup_table = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/data/own_lookup.xlsx'

data = pandas.read_excel(path_to_lookup_table)

debug_plot_savepath = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/debug_plots'

for idx, recording_id in data.iterrows():
    voltage_ch_name = recording_id['voltage_ch']
    calcium_ch_name = recording_id['calcium_ch']
    n_channels = recording_id['recording_nchannels']
    voltage_framerate = recording_id['framerate_vol']
    calcium_framerate = recording_id['framerate_cal']

    if n_channels == 2:
        # Create paths for voltage and calcium layers
        calcium_filename_no_ext, calcium_fileext = os.path.splitext(calcium_ch_name)
        calcium_path = f"{path_to_denoised_and_registered_imgs}/{calcium_filename_no_ext}_calcium.tif"
        calcium_img = tifffile.imread(calcium_path)

        voltage_filename_no_ext, voltage_fileext = os.path.splitext(voltage_ch_name)
        voltage_path = f"{path_to_denoised_and_registered_imgs}/{voltage_filename_no_ext}_voltage.tif"
        voltage_img = tifffile.imread(voltage_path)

        processed_df = analyse_voltage_and_calcium_layer(calcium_layer=calcium_img,
                                                         voltage_layer=voltage_img,
                                                         ca_ch_name=calcium_ch_name,
                                                         vo_ch_name=voltage_ch_name,
                                                         data_dataframe=data,
                                                         calcium_framerate=calcium_framerate,
                                                         voltage_framerate=voltage_framerate,
                                                         roi_params=roi_params,
                                                         debug_plot=True,
                                                         debug_plot_savepath=debug_plot_savepath)

    else:
        pass
