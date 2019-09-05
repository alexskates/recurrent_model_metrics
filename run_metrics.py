import os
import argparse
import numpy as np
import scipy.io as sio
from utils import Model
from visualisation import plot_loss, plot_covariances, plot_global_stats, \
    plot_windowed_fo, plot_kl_matrix, plot_correlation_matrix, \
    plot_timecourses, plot_autocorrelation


def main(params):
    np.random.seed(params.rand_seed)

    models = []
    for name, dir in zip(params.models[::2], params.models[1::2]):
        if 'hmm' in name.lower():
            hmm = sio.loadmat(os.path.join(dir, 'envelope_HMM_K8.mat'))
            models.append(
                Model(name.replace('_', ' '),
                      covariances=hmm['cov'],
                      gamma=hmm['Gamma'],
                      statepath=hmm['vpath'].flatten() - 1))
            models[-1].dir = dir
        else:
            models.append(Model(name, dir=dir))

    fig_dir = os.path.join(models[0].dir, 'figures')

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # # Just look at first subject for now
    # for m in models:
    #     m.gamma = m.gamma[:38000]
    #     m.statepath = m.statepath[:38000]
    #     m.statepath_onehot = m.statepath_onehot[:38000]

    # Match up the models using the munkres algorithm
    for m in models[1:]:
        m.reorder_states(None, models[0].covariances)

    # Export covariance matrices to matlab to get spatial maps
    for m in models:
        sio.savemat(
            os.path.join(models[0].dir, 'model_{}_cov.mat'.format(m.name)),
            {'C': m.covariances})

    # Plot all the figures
    plot_loss(models, fig_dir)
    plot_covariances(models, fig_dir)
    plot_timecourses(models, fig_dir, start=2000, length=10000,
                     freq=params.sample_freq)
    plot_timecourses(models, fig_dir, start=2000, length=10000,
                     freq=params.sample_freq, hard=False)

    if len(models) > 1:
        plot_correlation_matrix(models, fig_dir)
        plot_kl_matrix(models, fig_dir)

    plot_global_stats(models, fig_dir,
                      freq=params.sample_freq)
    plot_windowed_fo(models, fig_dir, params.sample_freq)
    plot_autocorrelation(models, fig_dir)
    # plot_hidden(model_gamma_full, model_hidden, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--models', '-m', nargs='+', required=True,
                        help='<Required> Pass in list, of name directory for '
                             'each model to be compared')
    parser.add_argument('--sample-freq', '-s', default=1000,
                        help='Sampling frequency in Hz (default 1000)')
    parser.add_argument('--rand-seed', type=int, default=42)
    parser.add_argument('--states', type=int, default=8)
    args = parser.parse_args()
    main(args)
