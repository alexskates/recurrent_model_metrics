import numpy as np
from os import path
from similarity import dice_binary, kl_divergence


class Model(object):
    def __init__(self, name, dir=None, covariances=None, gamma=None, loss=None,
                 statepath=None):
        self.name = name.replace('_', ' ')
        if dir is not None:
            self.dir = dir
            self.covariances = np.load(path.join(dir, 'inferred_covariances.npy'))
            self.K = self.covariances.shape[0]
            self.D = self.covariances.shape[1]
            self.gamma = np.load(path.join(dir, 'q_z_all.npy'))
            if len(self.gamma.shape) > 2:
                self.gamma = np.concatenate(self.gamma, 0)
            # Don't currently have a way of finding most likely sequence so
            # just take the most likely state at each time
            if path.exists(path.join(dir, 'state_seq_all.npy')):
                self.statepath = np.load(path.join(dir, 'state_seq_all.npy'))
                if len(self.statepath.shape) > 1:
                    self.statepath = np.argmax(self.statepath, 1)
            else:
                self.statepath = np.argmax(self.gamma, 1)
            if path.exists(path.join(dir, 'train_loss.npy')):
                self.loss = np.load(path.join(dir, 'train_loss.npy'))
            else:
                self.loss = None
        else:
            self.covariances = covariances
            self.K = covariances.shape[0]
            self.D = covariances.shape[1]
            self.gamma = gamma
            self.statepath = statepath
            self.loss = loss
        self.statepath_onehot = convert_to_onehot(self.statepath, self.K)

    def reorder_states(self, master_statepath_onehot=None, master_cov=None):
        if master_statepath_onehot is not None:
            munkres_inds = munkres_algorithm_indices(
                master_statepath_onehot, self.statepath_onehot, False)
        else:
            munkres_inds = munkres_algorithm_indices_kl(
                master_cov, self.covariances, False)

        self.covariances = self.covariances[munkres_inds]
        self.gamma = self.gamma[:, munkres_inds]
        self.statepath_onehot = self.statepath_onehot[:, munkres_inds]
        # For the sequence of states, could replace them all, but easier to do:
        self.statepath = np.argmax(self.statepath_onehot, 1)


def convert_to_onehot(sequence, num_states):
    """
    Takes a 1D sequence and converts to to a 2D one-hot sequence.
    """
    assert np.max(sequence) + 1 <= num_states, "Not enough states"
    one_hot = np.zeros((len(sequence), num_states))
    one_hot[np.arange(len(sequence)), sequence.astype(np.int)] = 1
    return one_hot


def munkres_algorithm_indices(seq1, seq2, verbose=True):
    """
    Returns the optimal permutation of the states of the first sequence so as
    to best match the states of the second sequence on the basis of maximal
    overlaps of activated state sequences (ensure seq1 and seq2 are one-hot).
    Uses the Munkres/Hungarian algorithm.

    For details, see:
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    https://pypi.org/project/munkres/ for usage of Munkres package
    """
    from munkres import Munkres, print_matrix
    m = Munkres()

    def similarity_matrix(seq1, seq2):
        """
        Calculates the similarity matrix of two active state sequences (where the
        sequences themselves are one-hot). For use with the Munkres/Hungarian
        algorithm.
        Rows correspond to seq1, columns to seq2,
        e.g. sim_mat[0, 1] would refer to 0th state of seq1 and 1st state of seq2
        """
        return np.array(
            [[dice_binary(seq1[:, j, np.newaxis], seq2[:, i, np.newaxis])
              for i in range(seq2.shape[1])] for j in range(seq1.shape[1])]
        )

    # Similarity: high = good, Cost: low = good.
    # Solution: Cost = Similarity * -1
    cost_matrix = -similarity_matrix(seq1, seq2)
    indexes = m.compute(cost_matrix)
    if verbose:
        print('Cost matrix:\n')
        print(cost_matrix)
        print_matrix(cost_matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = cost_matrix[row][column]
        total += value
        if verbose:
            print('({}, {}) -> {}'.format(row, column, value))
            print('total cost: {}'.format(total))
    return np.array(indexes)[:, 1]


def munkres_algorithm_indices_kl(cov1, cov2, verbose=True):
    """
    Returns the optimal permutation of the states of the first sequence so as
    to best match the states of the second sequence on the basis of maximal
    overlaps of activated state sequences (ensure seq1 and seq2 are one-hot).
    Uses the Munkres/Hungarian algorithm.

    For details, see:
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    https://pypi.org/project/munkres/ for usage of Munkres package
    """
    from munkres import Munkres, print_matrix
    m = Munkres()

    # Similarity: high = good, Cost: low = good.
    # Solution: Cost = KL Divergence
    kl = np.array(
            [[kl_divergence(np.zeros(cov1.shape[1]), cov1[j, :, :],
                            np.zeros(cov1.shape[1]), cov2[i, :, :])
              for i in range(cov1.shape[0])] for j in range(cov1.shape[0])]
        )
    cost_matrix = kl
    indexes = m.compute(cost_matrix)
    if verbose:
        print('Cost matrix:\n')
        print(cost_matrix)
        print_matrix(cost_matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = cost_matrix[row][column]
        total += value
        if verbose:
            print('({}, {}) -> {}'.format(row, column, value))
            print('total cost: {}'.format(total))
    return np.array(indexes)[:, 1]