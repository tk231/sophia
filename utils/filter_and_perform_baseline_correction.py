import numpy


def savgol_filter_and_fit_baseline(signal_1d: numpy.ndarray, signal_type: str = 'Voltage' or 'Calcium',
                                   debug_plot: bool = False) -> numpy.ndarray:
    import scipy.signal
    import pybaselines

    # Signal smoothing
    filtered_signals = scipy.signal.savgol_filter(x=signal_1d, window_length=5, polyorder=2)

    # Normalise intensities
    filtered_signals_max = filtered_signals.max()
    filtered_signals_min = filtered_signals.min()

    denominator_normalisation = filtered_signals_max - filtered_signals_min

    if signal_type == 'Voltage':
        filtered_signals_for_nextsteps = - (filtered_signals - filtered_signals_max) / denominator_normalisation
    elif signal_type == 'Calcium':
        filtered_signals_for_nextsteps = (filtered_signals - filtered_signals_min) / denominator_normalisation

    # Perform baseline fitting
    baseline_fitting_x = numpy.arange(0, len(filtered_signals_for_nextsteps), step=1)
    baseline_fitter = pybaselines.Baseline(x_data=baseline_fitting_x)

    bkg, params = baseline_fitter.iasls(filtered_signals_for_nextsteps, lam=5e3, p=0.02)

    normalised_corrected_signal = filtered_signals_for_nextsteps - bkg

    if debug_plot:
        import matplotlib.pyplot as plt

        plt.plot(signal_1d, label='Original signal')
        plt.plot(normalised_corrected_signal, label='Filtered signal')
        plt.plot(bkg, label='Background signal')
        plt.legend()

    return normalised_corrected_signal
