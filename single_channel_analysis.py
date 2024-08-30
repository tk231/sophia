import os.path

import numpy
import pandas
import tifffile
import glob
import tqdm


def analyse_single_layer(img: numpy.ndarray, imgname: str, roi_parameters: dict, idx_for_dataframe: int,
                         framerate: float, img_type: str = 'voltage' or 'calcium'):
    import numpy
    import uncertainties.unumpy

    import utils.create_evenly_spaced_rois
    import utils.calcium_roi_analysis
    import utils.voltage_roi_analysis

    df_entry = pandas.DataFrame(
        columns=['imgname', 'framerate',
                 'slicemean_APD80', 'slicemean_APD50', 'slicemean_CaTD80', 'slicemean_CaTD50',
                 'slicemean_vol_maxdfdt', 'slicemean_cal_maxdfdt',
                 'ROI1_x_startend', 'ROI1_y_startend',
                 'ROI1_APD80', 'ROI1_APD50', 'ROI1_vol_max_dfdt',
                 'ROI1_CaTD80', 'ROI1_CaTD50', 'ROI1_cal_max_dfdt',
                 'ROI2_x_startend', 'ROI2_y_startend',
                 'ROI2_APD80', 'ROI2_APD50', 'ROI2_vol_max_dfdt',
                 'ROI2_CaTD80', 'ROI2_CaTD50', 'ROI2_cal_max_dfdt',
                 'ROI3_x_startend', 'ROI3_y_startend',
                 'ROI3_APD80', 'ROI3_APD50', 'ROI3_vol_max_dfdt',
                 'ROI3_CaTD80', 'ROI3_CaTD50', 'ROI3_cal_max_dfdt',
                 'ROI4_x_startend', 'ROI4_y_startend',
                 'ROI4_APD80', 'ROI4_APD50', 'ROI4_vol_max_dfdt',
                 'ROI4_CaTD80', 'ROI4_CaTD50', 'ROI4_cal_max_dfdt',
                 'ROI5_x_startend', 'ROI5_y_startend',
                 'ROI5_APD80', 'ROI5_APD50', 'ROI5_vol_max_dfdt',
                 'ROI5_CaTD80', 'ROI5_CaTD50', 'ROI5_cal_max_dfdt',
                 'ROI6_x_startend', 'ROI6_y_startend',
                 'ROI6_APD80', 'ROI6_APD50', 'ROI6_vol_max_dfdt',
                 'ROI6_CaTD80', 'ROI6_CaTD50', 'ROI6_cal_max_dfdt',
                 'ROI7_x_startend', 'ROI7_y_startend',
                 'ROI7_APD80', 'ROI7_APD50', 'ROI7_vol_max_dfdt',
                 'ROI7_CaTD80', 'ROI7_CaTD50', 'ROI7_cal_max_dfdt',
                 'ROI8_x_startend', 'ROI8_y_startend',
                 'ROI8_APD80', 'ROI8_APD50', 'ROI8_vol_max_dfdt',
                 'ROI8_CaTD80', 'ROI8_CaTD50', 'ROI8_cal_max_dfdt',
                 'ROI9_x_startend', 'ROI9_y_startend',
                 'ROI9_APD80', 'ROI9_APD50', 'ROI9_vol_max_dfdt',
                 'ROI9_CaTD80', 'ROI9_CaTD50', 'ROI9_cal_max_dfdt'], dtype=object)

    # Create ROIs
    roi_dict = utils.create_evenly_spaced_rois.create_rois_in_centre(img,
                                                                     img_id=imgname,
                                                                     roi_size=(roi_parameters['roi_size'],
                                                                               roi_parameters['roi_size']),
                                                                     num_rois=roi_parameters['number_of_rois'],
                                                                     margin_percentage=roi_parameters[
                                                                         'margin_percentage'])

    # Write to dataframe
    df_entry.at[idx_for_dataframe, 'imgname'] = imgname
    df_entry.at[idx_for_dataframe, 'framerate'] = framerate

    for roi_key in roi_dict:
        df_entry.at[idx_for_dataframe, f"{roi_key}_x_startend"] = roi_dict[roi_key]['x_start_end']
        df_entry.at[idx_for_dataframe, f"{roi_key}_y_startend"] = roi_dict[roi_key]['y_start_end']

        if img_type == 'calcium':
            # Calculate CaTD50
            CaTD80_in_seconds, CaTD50_in_seconds, cal_max_dfdt_in_seconds, Ca_debug_plot = utils.calcium_roi_analysis.analyse_single_roi(
                roi_dict=roi_dict[roi_key],
                img_name=imgname,
                framerate=framerate,
                debug_plot=False)

            filtered_cal_max_dfdt_list = [cal_maxdfdt_val for cal_maxdfdt_val, CaTD80_val in zip(cal_max_dfdt_in_seconds, CaTD80_in_seconds) if not numpy.isnan(CaTD80_val)]

            # Get values Calcium channel
            roi_CaTD80_mean = numpy.nanmean(CaTD80_in_seconds, axis=None) if not numpy.isnan(
                CaTD80_in_seconds).all() else numpy.nan
            roi_CaTD50_mean = numpy.nanmean(CaTD50_in_seconds, axis=None) if not numpy.isnan(
                CaTD50_in_seconds).all() else numpy.nan
            cal_max_dfdt_in_seconds_mean = numpy.nanmean(filtered_cal_max_dfdt_list, axis=None) if not numpy.isnan(
                filtered_cal_max_dfdt_list).all() else numpy.nan

            roi_CaTD80_std = numpy.nanstd(CaTD80_in_seconds, axis=None) if not numpy.isnan(
                CaTD80_in_seconds).all() else numpy.nan
            roi_CaTD50_std = numpy.nanstd(CaTD50_in_seconds, axis=None) if not numpy.isnan(
                CaTD50_in_seconds).all() else numpy.nan
            cal_max_dfdt_in_seconds_std = numpy.nanstd(cal_max_dfdt_in_seconds, axis=None) if not numpy.isnan(
                cal_max_dfdt_in_seconds).all() else numpy.nan


            # Pack values into an uncertainties float and save into dict
            df_entry.at[idx_for_dataframe, f"{roi_key}_CaTD80"] = uncertainties.ufloat(roi_CaTD80_mean, roi_CaTD80_std,
                                                                                       tag='seconds')
            df_entry.at[idx_for_dataframe, f"{roi_key}_CaTD50"] = uncertainties.ufloat(roi_CaTD50_mean, roi_CaTD50_std,
                                                                                       tag='seconds')
            df_entry.at[idx_for_dataframe, f"{roi_key}_cal_max_dfdt"] = uncertainties.ufloat(cal_max_dfdt_in_seconds_mean,
                                                                                             cal_max_dfdt_in_seconds_std,
                                                                                             tag='seconds')

        else:
            # Calculate APD80, APD50
            APD80_in_seconds, APD50_in_seconds, vol_max_dfdt_in_seconds, vol_debug_plot = utils.voltage_roi_analysis.analyse_single_roi(
                roi_dict=roi_dict[roi_key],
                img_name=imgname,
                framerate=framerate,
                debug_plot=False)

            filtered_vol_max_dfdt_list = [vol_maxdfdt_val for vol_maxdfdt_val, APD80_val in zip(vol_max_dfdt_in_seconds, APD80_in_seconds) if not numpy.isnan(APD80_val)]


            # Get values Voltage channel
            roi_APD80_mean = numpy.nanmean(APD80_in_seconds, axis=None) if not numpy.isnan(
                APD80_in_seconds).all() else numpy.nan
            roi_APD50_mean = numpy.nanmean(APD50_in_seconds, axis=None) if not numpy.isnan(
                APD50_in_seconds).all() else numpy.nan
            roi_vol_dfdt_max_mean = numpy.nanmean(filtered_vol_max_dfdt_list, axis=None) if not numpy.isnan(
                filtered_vol_max_dfdt_list).all() else numpy.nan

            roi_APD80_std = numpy.nanstd(APD80_in_seconds, axis=None) if not numpy.isnan(
                APD80_in_seconds).all() else numpy.nan
            roi_APD50_std = numpy.nanstd(APD50_in_seconds, axis=None) if not numpy.isnan(
                APD50_in_seconds).all() else numpy.nan
            roi_vol_dfdt_max_std = numpy.nanstd(filtered_vol_max_dfdt_list, axis=None) if not numpy.isnan(
                filtered_vol_max_dfdt_list).all() else numpy.nan

            # Pack values into an uncertainties float and save into dict
            df_entry.at[idx_for_dataframe, f"{roi_key}_APD80"] = uncertainties.ufloat(roi_APD80_mean, roi_APD80_std,
                                                                                      tag='seconds')
            df_entry.at[idx_for_dataframe, f"{roi_key}_APD50"] = uncertainties.ufloat(roi_APD50_mean, roi_APD50_std,
                                                                                      tag='seconds')
            df_entry.at[idx_for_dataframe, f"{roi_key}_vol_max_dfdt"] = uncertainties.ufloat(roi_vol_dfdt_max_mean,
                                                                                             roi_vol_dfdt_max_std,
                                                                                             tag='seconds')

    if img_type == 'calcium':
        calcium80_col_list = ['ROI1_CaTD80', 'ROI2_CaTD80', 'ROI3_CaTD80', 'ROI4_CaTD80', 'ROI5_CaTD80', 'ROI6_CaTD80',
                              'ROI7_CaTD80', 'ROI8_CaTD80', 'ROI9_CaTD80']

        calcium50_col_list = ['ROI1_CaTD50', 'ROI2_CaTD50', 'ROI3_CaTD50', 'ROI4_CaTD50', 'ROI5_CaTD50', 'ROI6_CaTD50',
                              'ROI7_CaTD50', 'ROI8_CaTD50', 'ROI9_CaTD50']

        calciummaxdfdt_col_list = ['ROI1_cal_max_dfdt', 'ROI2_cal_max_dfdt', 'ROI3_cal_max_dfdt', 'ROI4_cal_max_dfdt',
                                   'ROI5_cal_max_dfdt', 'ROI6_cal_max_dfdt', 'ROI7_cal_max_dfdt', 'ROI8_cal_max_dfdt',
                                   'ROI9_cal_max_dfdt']

        slicemean_CaTD80 = numpy.nanmean(df_entry[calcium80_col_list])
        slicemean_CaTD50 = numpy.nanmean(df_entry[calcium50_col_list])
        slicemean_cal_maxdfdt = numpy.nanmean(df_entry[calciummaxdfdt_col_list])

        if uncertainties.unumpy.isnan(slicemean_CaTD80) or uncertainties.unumpy.isnan(slicemean_CaTD50) or uncertainties.unumpy.isnan(slicemean_cal_maxdfdt):
            print(f"Programme failed on image {imgname}")
            pass
        else:
            df_entry.at[idx_for_dataframe, 'slicemean_CaTD80'] = slicemean_CaTD80
            df_entry.at[idx_for_dataframe, 'slicemean_CaTD50'] = slicemean_CaTD50
            df_entry.at[idx_for_dataframe, 'slicemean_cal_maxdfdt'] = slicemean_cal_maxdfdt

    else:
        voltage80_col_list = ['ROI1_APD80', 'ROI2_APD80', 'ROI3_APD80', 'ROI4_APD80', 'ROI5_APD80', 'ROI6_APD80',
                              'ROI7_APD80', 'ROI8_APD80', 'ROI9_APD80', ]

        voltage50_col_list = ['ROI1_APD50', 'ROI2_APD50', 'ROI3_APD50', 'ROI4_APD50', 'ROI5_APD50', 'ROI6_APD50',
                              'ROI7_APD50', 'ROI8_APD50', 'ROI9_APD50']

        voltagemaxdfdt_col_list = ['ROI1_vol_max_dfdt', 'ROI2_vol_max_dfdt', 'ROI3_vol_max_dfdt', 'ROI4_vol_max_dfdt',
                                   'ROI5_vol_max_dfdt', 'ROI6_vol_max_dfdt', 'ROI7_vol_max_dfdt', 'ROI8_vol_max_dfdt',
                                   'ROI9_vol_max_dfdt']

        slicemean_APD80 = numpy.nanmean(df_entry[voltage80_col_list].to_numpy())
        slicemean_APD50 = numpy.nanmean(df_entry[voltage50_col_list].to_numpy())
        slicemean_vol_maxdfdt = numpy.nanmean(df_entry[voltagemaxdfdt_col_list].to_numpy())

        if uncertainties.unumpy.isnan(slicemean_APD80) or uncertainties.unumpy.isnan(slicemean_APD50) or uncertainties.unumpy.isnan(slicemean_vol_maxdfdt):
            print(f"Programme failed on image {imgname}")
            pass
        else:
            df_entry.at[idx_for_dataframe, 'slicemean_APD80'] = slicemean_APD80
            df_entry.at[idx_for_dataframe, 'slicemean_APD50'] = slicemean_APD50
            df_entry.at[idx_for_dataframe, 'slicemean_vol_maxdfdt'] = slicemean_vol_maxdfdt

    # Note: ECC and RRC have to be calculated in another program, because this program can only handle single signals

    return df_entry


roi_params = {'roi_size': 30,
              'number_of_rois': 9,
              'margin_percentage': 0.25}

# path_to_denoised_and_registered_imgs = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/denoised_and_registered'
# path_to_lookup_table = ('/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/data/Master Sheet of Recordings for Motion '
#                         'Correction_processed.xlsx')
path_to_denoised_and_registered_imgs = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/denoised_and_registered'
path_to_lookup_table = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/master_mastersheet_processed.xlsx'
path_for_processed_excel = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results'

lookup_table = pandas.read_excel(path_to_lookup_table)

debug_plot_savepath = '/mnt/Aruba/PyCharmProjects/bre_omar_test_analysis2/results/debug_plots'

list_of_imgs = glob.glob(f"{path_to_denoised_and_registered_imgs}/*.tif")

processed_df = pandas.DataFrame()

for list_idx, img_path in tqdm.tqdm(enumerate(list_of_imgs)):
    img_basename, img_ext = os.path.splitext(os.path.basename(img_path))

    img = tifffile.imread(img_path)

    # Determine if calcium or voltage
    img_id, recording_type = img_basename.split('_')

    # Get framerate from lookup table, divide by 2 if double images obtained!
    single_or_double = lookup_table.loc[lookup_table['Filename'] == f"{img_id}.sif", 'Single/Double'].item()
    framerate_both = lookup_table.loc[lookup_table['Filename'] == f"{img_id}.sif", 'Framerate'].item()

    if single_or_double == 'Double':
        framerate = framerate_both / 2
    else:
        framerate = framerate_both

    row = analyse_single_layer(img=img, imgname=img_basename, roi_parameters=roi_params, idx_for_dataframe=list_idx,
                               framerate=framerate, img_type=recording_type)

    processed_df = pandas.concat([processed_df, row], ignore_index=True)

processed_df.to_excel(f'{path_for_processed_excel}/total_results.xlsx', index=False)
processed_df.to_pickle(f'{path_for_processed_excel}/total_results.pkl')
