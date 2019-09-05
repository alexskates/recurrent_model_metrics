import numpy as np


def dice_binary(seq1, seq2, invalid_value=1.):
    """
    The Sorenson-Dice coefficient is a statistic used for comparing the
    similarity of two samples. This function takes two binary (one-hot)
    sequences. Invalid_value is used where both sequences are all zero.
    Defaults to 1.0 as if they are both completely zero, they are the same.

    Given by the vector operations:
    DICE = \frac{2 * | a \cdot b|}{|a|^2 + |b|^2}
    """
    assert seq1.shape == seq2.shape, "Sizes of both sequences must match"
    seq1 = np.asarray(seq1).astype(np.bool)
    seq2 = np.asarray(seq2).astype(np.bool)
    if len(seq1.shape) == 2:
        # If only two axes then is a single sequence
        seq_sum = seq1.sum() + seq2.sum()
        if seq_sum == 0.:
            return invalid_value
        return 2. * np.logical_and(seq1, seq2).sum() / seq_sum
    else:
        # Otherwise is sliding window so want to broadcast operations
        # Hacky solution for where both sequences have no activations of a state
        with np.errstate(divide='ignore', invalid='ignore'):
            dice = 2. * np.logical_and(seq1, seq2).sum(axis=(1, 2)) / (
                    seq1.sum(axis=(1, 2)) + seq2.sum(axis=(1, 2)))
            dice[np.isnan(dice)] = invalid_value
        return dice


def kl_divergence(mu_a, Sigma_a, mu_b, Sigma_b):
    """
    Calculate the KL Divergence between two multivariate normal distributions
    with parameters (mu_a, Sigma_a) and (mu_b, Sigma_b), e.g.

    D_kl (N_a || N_b) =

    1/2 (log(det Sigma_b / det Sigma_a) - n + tr(Sigma_b^-1 Sigma_a) +
    (mu_b - mu_a)'Sigma_b^-1(mu_b - mu_a))

    Derivation is given by http://stanford.edu/~jduchi/projects/general_notes.pdf
    """
    n = Sigma_a.shape[0]
    diff = mu_b - mu_a
    Sigma_b_inv = np.linalg.inv(Sigma_b)
    det_a = np.linalg.det(Sigma_a)
    det_b = np.linalg.det(Sigma_b)
    return 0.5 * (np.log(det_b / det_a) - n +
                  np.trace(np.dot(Sigma_b_inv, Sigma_a))
                  + np.dot(np.dot(diff.T, Sigma_b_inv), diff))
