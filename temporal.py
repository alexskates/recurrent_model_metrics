import numpy as np
from numpy.lib.stride_tricks import as_strided as astr
import itertools


def sliding_window(seq, window_size, step_size, mirror=False):
    """
    Return a sliding window of size window_size over the first
    axis of a sequence seq. Returns in the form of an array, where the
    i-th element of an array.

    Uses as_strided, which is better than a loop as it doesn't actually
    have to allocate any more memory, all it does is give a view into seq.
    Basically it uses C code for the loops, instead of python.

    If mirror=True, then creates N windows centred on each element of the
    sequence. Padding at the start and end of the sequence will be mirrored
    data. Used for the time-varying hurst parameter.
    See the paper: 'Application of a Time-Scale Local Hurst Exponent
    analysis to time series' for details

    seq: np.array. 1D or higher
    window_size: size of the sliding window
    step_size: how big a step to take
    """
    if mirror:
        n_windows = seq.shape[0]
        step_size = 1  # Ensure step_size is 1
        # How much do we need to pad by?
        pad_front = int(np.floor(window_size - 1) / 2)
        pad_back = int(window_size / 2)
        data = np.concatenate(
            [np.flip(seq[:pad_front], axis=0),
             seq,
             np.flip(seq[-pad_back:], axis=0)],
            axis=0)
    else:
        n_windows = (seq.shape[0] - window_size) // step_size
        data = seq

    # Shape is the resulting shape of the array
    shape = (n_windows, window_size, *data.shape[1:])
    # Strides are the amount of memory needed to move to the next piece of
    # data. In this case, need to move step_size * seq.shape[-1] * bytes to
    # get to the next sliding window.
    strides = (data.strides[0] * step_size, *data.strides)
    return astr(data, shape=shape, strides=strides)


def state_lifetimes(sequence, s):
    """
    Given a sequence of state activations, calculate the state lifetimes for 
    state s
    To use with sliding windows, use the following format:
    intervals = list(map(lambda w: interval_times(w, s), windows))

    sequence: np.array of size [T,]
    s: int, state to calculate the lifetime of
    """
    if s in sequence:
        return [np.sum(1 for _ in y)
                for x, y in itertools.groupby(sequence) if x == s]


def interval_times(sequence, s):
    """
    Given a sequence of state activations, calculate the interval times
    between activations of state s
    To use with sliding windows, use the following format:
    intervals = list(map(lambda w: interval_times(w, s), windows))

    sequence: np.array of size [T,]
    s: int, state to calculate the inteval time of
    """
    if s in sequence:
        return [np.sum(1 for _ in y)
                for x, y in itertools.groupby(sequence, lambda z: z == s)
                if not x]


def fractional_occupancy(sequence, s):
    """
    Given a sequence of state activations, calculate the fractional occupancy 
    for state s.
    To use with sliding windows, use the following format:
    fo = fractional_occupancy(windows, s)

    sequence: np.array
    s: int, state to calculate the fractional occupancy of
    """
    if len(sequence.shape) == 2:
        return np.sum(sequence == s, axis=1) / sequence.shape[1]
    else:
        return np.sum(sequence == s) / len(sequence)


def total_switches(sequence):
    """
    Given a sequence of state activations, calculate the number of times the
    state changes.
    To use with sliding windows, use the following format:
    switches = list(map(total_switches, windows))
    """
    return sum(1 for _ in itertools.groupby(sequence)) - 1


def autocorr(x):
    """
    Given a sequence, calculate the normalised autocorrelation of the series.
    """
    result = np.correlate(x, x, mode='full')
    result = result[int(result.size / 2):]
    max_corr = np.max(result)
    return result / max_corr  # Normalised


def RS_func(series):
    """
    Rescaled range, using cumulative sum of deviation from the mean.

    series: array corresponding to [window_size x dim] of sequence
    """
    mean = np.mean(series, axis=0)
    deviations = series - mean
    Z = np.cumsum(deviations, axis=0)
    R = np.max(Z, axis=0) - np.min(Z, axis=0)
    S = np.var(series, axis=0)
    return R / S


