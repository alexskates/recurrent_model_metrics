import numpy as np
from temporal import sliding_window


def RS_func(series, ax=1):
    """
    Rescaled range, using cumulative sum of deviation from the mean.
    Calculate across axis ax (want it to be the sequence
    """
    mean = np.mean(series, axis=ax, keepdims=True)
    deviations = series - mean
    Z = np.cumsum(deviations, axis=ax)
    R = np.max(Z, axis=ax) - np.min(Z, axis=ax)
    S = np.std(series, axis=ax)
    return R / S

model_gamma = np.load('/Users/askates/Documents/Projects/model/results'
                      '/hyperparam_explore/40_200_L1_H64_D0.0_LR0'
                      '.000006125_E700/q_z_all.npy')
model_path = model_gamma[:1000, 0]

H = []

for w in range(20, 500, 1):
    windowed_data = sliding_window(model_path, w, 1, True)

    N = windowed_data.shape[1]
    min_k = 8
    max_k = int(np.floor(N / 2))

    # To calculate Hurst parameter, calculate R/S over a variety of partial time
    # series sizes
    all_k = np.arange(min_k, max_k + 1)
    rs_all = np.empty((windowed_data.shape[0], all_k.shape[0]))

    for idx, k in enumerate(all_k): # Zero indexed, so need to add 1.
        # Get each non-overlapping segment to calculate R/S value upon
        segments = np.array([windowed_data[:, i:i+k] for i in range(0, N+1-k, k)])
        # Calculate the R/S values across each segment
        rs_segments = RS_func(segments, 2)
        # Average over all partial series
        rs_mean = np.mean(rs_segments, axis=0)
        rs_all[:, idx] = rs_mean

    # Can use least squares to fit the slope of each segment
    # To fit y = mx + c, rewrite as y = Ap where A = [[x 1]] and p = [[m], [c]]
    # Then use np.linalg.lstsq to solve for p
    A = np.vstack([np.log10(all_k), np.ones(all_k.shape[0])]).T
    # The gradient returned is H
    Hurst, _ = np.linalg.lstsq(A, np.log10(rs_all.T), rcond=None)[0]
    H.append(Hurst)
    print('Done {}'.format(w))






