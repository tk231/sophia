import typing

def analyse_single_roi(roi_dict: dict, t_roi: tuple or None, peak_heights: float or None, img_name: str or None,
                       framerate: float, expected_hr_ms: typing.Optional[int] = 250, debug_plot: bool = False):
    import numpy
    import scipy.signal
    import utils.filter_and_perform_baseline_correction
    import utils.find_closest_mins

    # TODO: find better peak distances
    if expected_hr_ms is not None:
        peak_distances = round(framerate * (expected_hr_ms / 1000))
    else:
        peak_distances = 14

    if t_roi is None:
        roi_intensities = numpy.mean(roi_dict['roi'], (1, 2))
    else:
        t_roi_start, t_roi_end = t_roi
        roi_intensities = numpy.mean(roi_dict['roi'][t_roi_start:t_roi_end], (1, 2))

    # Filter and fit baseline
    filtered_signal = utils.filter_and_perform_baseline_correction.savgol_filter_and_fit_baseline(
        signal_1d=roi_intensities,
        signal_type='Voltage')

    # Action potential peaks
    if peak_heights is None:
        ap_peaks, _ = scipy.signal.find_peaks(filtered_signal,  height=0.6, distance=peak_distances)
    else:
        ap_peaks, _ = scipy.signal.find_peaks(filtered_signal, height=peak_heights, distance=peak_distances)

    # Values of peaks
    ap_peak_values = filtered_signal[ap_peaks]

    # APDn calculation
    # For APD80, 50
    apd80_vals = ap_peak_values * 0.8
    apd50_vals = ap_peak_values * 0.5

    apd80_threshold = ap_peak_values - apd80_vals
    apd50_threshold = ap_peak_values - apd50_vals

    # Calculate start of action potentials using the min of peaks
    min_peaks, _ = scipy.signal.find_peaks(-filtered_signal, distance=peak_distances)

    # Check and make sure first peak is after first min, else get the next peak
    signal_lims = utils.find_closest_mins.find_closest_mins2(list_of_peaks=ap_peaks, list_of_mins=min_peaks)

    # Calculate ap80s and ap50s
    APD80_in_seconds = []
    APD50_in_seconds = []
    max_dfdt_in_seconds = []

    for i in range(len(signal_lims)):
        # Start at action potential
        signal_start_end = signal_lims[i]

        start_idx = signal_start_end[0] if signal_start_end[0] is not None else 0
        end_idx = signal_start_end[1] if signal_start_end[1] is not None else len(filtered_signal)

        # Segment of action potential
        actionpotential_segment = filtered_signal[start_idx:end_idx]

        # Interpolate signal
        # Create interpolated signal
        interpolation_points_n = 10 * len(actionpotential_segment)
        fine_x = numpy.linspace(0, len(actionpotential_segment), num=interpolation_points_n,
                                endpoint=True)  # Need to use linspace for num
        rough_x = numpy.arange(0, len(actionpotential_segment),
                               step=1)  # Use arange with steps, else values will be weird

        signal_cubic_spline = scipy.interpolate.CubicSpline(rough_x, actionpotential_segment)
        signal_fine = signal_cubic_spline(fine_x)

        # Find indices where actionpotential_segment is below to the threshold
        ap80_indices = numpy.where(signal_fine <= apd80_threshold[i])[0]
        ap50_indices = numpy.where(signal_fine <= apd50_threshold[i])[0]

        #
        if (len(ap80_indices) <= 1) or (len(ap50_indices) <= 1):
            APD80_in_seconds.append(numpy.nan)
            APD50_in_seconds.append(numpy.nan)
            max_dfdt_in_seconds.append(numpy.nan)
            continue

        else:
            # APD80 and APD50 values are biggest gap differences between values
            ap80_gap = numpy.diff(ap80_indices).max() / 10
            ap50_gap = numpy.diff(ap50_indices).max() / 10

            apd80_vals = ap80_gap / framerate
            APD80_in_seconds.append(apd80_vals if apd80_vals > 0.15 else numpy.nan)

            apd50_vals = ap50_gap / framerate
            APD50_in_seconds.append(apd50_vals if apd50_vals > 0.11 else numpy.nan)

            # Calculate max_dfdt
            # TODO: check max_dfdt
            max_in_actionpotential = numpy.argmax(signal_fine)
            if max_in_actionpotential == 0:
                max_dfdt_in_seconds.append(numpy.nan)
                pass
            else:
                max_diff_idx_in_actionpotential = numpy.argmax(numpy.diff(signal_fine[:max_in_actionpotential]))
                max_diff_in_actionpotential = fine_x[max_diff_idx_in_actionpotential] + start_idx
                max_dfdt_in_seconds.append(max_diff_in_actionpotential / framerate)


    # TODO: fix debug_plot
    if debug_plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        debug_fig, ax = plt.subplots()

        ax.plot(roi_intensities, label='Raw signal')
        ax.plot(filtered_signal, label='Filtered signal')
        ax.plot(ap_peaks, filtered_signal[ap_peaks], 'x', label='AP peaks')
        ax.plot(min_peaks, filtered_signal[min_peaks], 'x', label='Peak start/end')
        #plt.plot(apdn_list, filtered_signal[apdn_list], 'x')
        #plt.title(f"ROI{}")

        plt.show()
    else:
        debug_fig = None

    return APD80_in_seconds, APD50_in_seconds, max_dfdt_in_seconds, debug_fig
