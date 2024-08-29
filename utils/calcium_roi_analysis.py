import numpy


def analyse_single_roi(roi_dict: dict, t_roi: tuple or None, img_name: str or None, framerate: float,
                       peak_heights: float or None, expected_hr_ms: int = 250,
                       debug_plot: bool = False):
    import scipy.signal
    import scipy.interpolate

    import utils.filter_and_perform_baseline_correction
    import utils.find_closest_mins

    # TODO: find better peak distances
    peak_distances = round(framerate * (expected_hr_ms / 1000))

    if t_roi is None:
        roi_intensities = numpy.mean(roi_dict['roi'], (1, 2))
    else:
        t_roi_start, t_roi_end = t_roi
        roi_intensities = numpy.mean(roi_dict['roi'][t_roi_start:t_roi_end], (1, 2))

    # Filter and perform baseline correction<
    filtered_signal = utils.filter_and_perform_baseline_correction.savgol_filter_and_fit_baseline(
        signal_1d=roi_intensities,
        signal_type='Calcium')

    # Find peaks
    if peak_heights is None:
        calcium_peaks, _ = scipy.signal.find_peaks(filtered_signal, height=0.8, distance=peak_distances)
    else:
        calcium_peaks, _ = scipy.signal.find_peaks(filtered_signal, height=peak_heights, distance=peak_distances)

    # Get peak values
    calcium_peak_vals = filtered_signal[calcium_peaks]

    # Calculate Calcium Transient N values
    # For CaTD80
    catd80_vals = calcium_peak_vals * 0.8
    catd50_vals = calcium_peak_vals * 0.5

    catd80_threshold = calcium_peak_vals - catd80_vals
    catd50_threshold = calcium_peak_vals - catd50_vals

    # Calculate start of calcium transients using the min of peaks
    min_peaks, _ = scipy.signal.find_peaks(-filtered_signal, distance=peak_distances)

    # Check and make sure first peak is after first min, else get the next peak
    signal_lims = utils.find_closest_mins.find_closest_mins2(list_of_peaks=calcium_peaks, list_of_mins=min_peaks)

    # Calculate catd80s, catd50s, and max dfdts
    CaTD80_in_seconds = []
    CaTD50_in_seconds = []
    max_dfdt_in_seconds = []

    for i in range(len(signal_lims)):
        # Start of calcium transient peak is first min in min_peaks
        # End of calcium transient peak is next min in min_peaks, else end of filtered_signal
        signal_start_end = signal_lims[i]

        start_signal_idx = signal_start_end[0] if signal_start_end[0] is not None else 0
        end_signal_idx = signal_start_end[1] if signal_start_end[1] is not None else len(filtered_signal)

        # Get CaT80 or CaT50 start and end idx
        # Segment of calcium transient curve
        calciumtransient_segment = filtered_signal[start_signal_idx:end_signal_idx]

        # Interpolate signal
        # Create interpolated signal
        interpolation_points_n = 10 * len(calciumtransient_segment)
        fine_x = numpy.linspace(0, len(calciumtransient_segment), num=interpolation_points_n,
                                endpoint=True)  # Need to use linspace for num
        rough_x = numpy.arange(0, len(calciumtransient_segment),
                               step=1)  # Use arange with steps, else values will be weird

        signal_cubic_spline = scipy.interpolate.CubicSpline(rough_x, calciumtransient_segment)
        signal_fine = signal_cubic_spline(fine_x)

        # Find indices where calciumtransient_segment is below to the threshold
        cat80_idx = numpy.where(signal_fine <= catd80_threshold[i])[0]
        cat50_idx = numpy.where(signal_fine <= catd50_threshold[i])[0]

        if (len(cat80_idx) <= 1) or (len(cat50_idx) <= 1):
            CaTD80_in_seconds.append(numpy.nan)
            CaTD50_in_seconds.append(numpy.nan)
            max_dfdt_in_seconds.append(numpy.nan)
            break

        else:
            # CaTD80 and CaTD50 values are biggest gap differences between values
            cat80_gap = numpy.diff(cat80_idx).max() / 10
            cat50_gap = numpy.diff(cat50_idx).max() / 10

            catd80_vals = cat80_gap / framerate
            CaTD80_in_seconds.append(catd80_vals if catd80_vals > 0.15 else numpy.nan)

            catd50_vals = cat50_gap / framerate
            CaTD50_in_seconds.append(catd50_vals if catd50_vals > 0.11 else numpy.nan)

            # Calculate max_dfdt
            # TODO: check max_dfdt
            max_in_calcium_transient = numpy.argmax(signal_fine)
            # Get peak
            if max_in_calcium_transient == 0:
                max_dfdt_in_seconds.append(numpy.nan)
                pass
            else:
                max_diff_idx_in_calcium_transient = numpy.argmax(numpy.diff(signal_fine[:max_in_calcium_transient]))  # Get max dfdt
                max_diff_in_calcium_transient = fine_x[max_diff_idx_in_calcium_transient] + start_signal_idx
                max_dfdt_in_seconds.append(max_diff_in_calcium_transient / framerate)

    if debug_plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        debug_fig, ax = plt.subplots()

        ax.plot(roi_intensities, label='Raw signal')
        ax.plot(filtered_signal, label='Filtered signal')
        ax.plot(calcium_peaks, filtered_signal[calcium_peaks], 'x', label='Ca peaks')
        ax.plot(min_peaks, filtered_signal[min_peaks], 'x', label='Peak start/end')
        # TODO: make debug plots of CaTD50, CaTD80, and max dfdt
        plt.title()

        plt.show()
    else:
        debug_fig = None

    return CaTD80_in_seconds, CaTD50_in_seconds, max_dfdt_in_seconds, debug_fig
