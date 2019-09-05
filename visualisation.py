import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
from typing import List
from utils import Model
from itertools import combinations
from similarity import kl_divergence
from temporal import sliding_window, state_lifetimes, interval_times, \
    fractional_occupancy, autocorr


def plot_covariances(models: List[Model], fig_dir):
    # Plot covariance matrices
    num_states = models[0].K
    f, axarr = plt.subplots(len(models), num_states,
                            figsize=[20, 3 + 1.5 * len(models)])
    f.suptitle('Covariance Matrices')
    for i in range(num_states):
        if len(models) > 1:
            for j, m in enumerate(models):
                m_cov = ma.masked_where(np.eye(m.D), m.covariances[i])
                # m_cov = m.covariances[i]
                axarr[j, i].set_title('State {}, {}'.format((i + 1), m.name))
                axarr[j, i].imshow(m_cov, interpolation='nearest')
                axarr[j, i].set_axis_off()
        else:
            m_cov = ma.masked_where(np.eye(models[0].D), models[0].covariances[i])
            axarr[i].set_title('State {}, {}'.format((i + 1), models[0].name))
            axarr[i].imshow(m_cov, interpolation='nearest')
            axarr[i].set_axis_off()
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    plt.savefig(os.path.join(fig_dir, 'covariances.pdf'))
    plt.close()


def plot_loss(models: List[Model], fig_dir):
    plt.figure(figsize=[6, 4])
    for m in models:
        if m.loss is not None:
            plt.plot(m.loss, label=m.name)
    plt.title('Training Loss')
    plt.ylabel('VFE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'loss.pdf'))
    plt.close()


def plot_global_stats(models: List[Model], fig_dir, freq):
    num_models = len(models)
    num_states = models[0].K

    # State Lifetimes
    appended_data = []
    for m in models:
        lifetimes = [state_lifetimes(m.statepath, i) for i in range(
            num_states)]
        for j in range(num_states):
            if lifetimes[j] is None:
                appended_data.append(
                    pd.DataFrame({
                        'Duration': [0],
                        'State': [j + 1],
                        'Model': [m.name]
                    }))
            else:
                appended_data.append(
                    pd.DataFrame({
                        'Duration': [float(l) / freq * 1000 for l in lifetimes[j]],
                        'State': np.repeat(j + 1, len(lifetimes[j])),
                        'Model': np.repeat(m.name, len(lifetimes[j]))
                    }))
    lifetimes_df = pd.concat(appended_data)

    # State Interval Times
    appended_data = []
    for m in models:
        intervals = [interval_times(m.statepath, i) for i in range(
            num_states)]
        for j in range(num_states):
            if intervals[j] is None:
                appended_data.append(
                    pd.DataFrame({
                        'Duration': [0],
                        'State': [j + 1],
                        'Model': [m.name]
                    }))
            else:
                appended_data.append(
                    pd.DataFrame({
                        'Duration': [float(l) / freq * 1000 for l in intervals[j]],
                        'State': np.repeat(j + 1, len(intervals[j])),
                        'Model': np.repeat(m.name, len(intervals[j]))
                    }))
    intervals_df = pd.concat(appended_data)

    # State Fractional Occupancy
    appended_data = []
    for m in models:
        fos = [fractional_occupancy(m.statepath, i) for i in range(num_states)]
        appended_data.append(
            pd.DataFrame({
                'Percent': fos,
                'State': np.arange(1, num_states + 1),
                'Model': np.repeat(m.name, num_states)
            }))
    fo_df = pd.concat(appended_data)

    f, axarr = plt.subplots(1, 5, figsize=[22, 5])

    axarr[0].set_title('State Lifetimes')
    sns.boxplot(x='State', y='Duration', hue='Model', data=lifetimes_df,
             ax=axarr[0], showfliers=False)

    axarr[1].set_title('State Lifetimes')
    sns.violinplot(x='State', y='Duration', hue='Model', data=lifetimes_df,
                   split=(num_models == 2), cut=0, ax=axarr[1])

    axarr[2].set_title('State Interval Times')
    sns.boxplot(x='State', y='Duration', hue='Model', data=intervals_df,
             ax=axarr[2], showfliers=False)

    axarr[3].set_title('State Interval Times')
    sns.violinplot(x='State', y='Duration', hue='Model', data=intervals_df,
                   split=(num_models == 2), cut=0, ax=axarr[3])

    axarr[4].set_title('State Fractional Occupancy')
    g = sns.catplot(x='State', y='Percent', hue='Model', data=fo_df,
                    kind='bar', ax=axarr[4])

    for ax in axarr[1:]:
        ax.get_legend().remove()
    plt.close(g.fig)  # Catplot makes its own figure, close this.
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'temporal.pdf'))
    plt.close()


def plot_windowed_fo(models: List[Model], fig_dir, freq):
    # Fractional Occupancy variance vs sliding window size
    num_states = models[0].K
    windows = np.arange(100, 5000, 10)

    # Get ticks
    tick_diff = (5000 - 100) / 10
    tick_diff_secs = int(np.ceil(tick_diff / freq))
    num_ticks = int(5000 / (tick_diff_secs * freq))
    ticks = np.arange(1, num_ticks + 1) * freq * tick_diff_secs
    tick_labels = np.arange(tick_diff_secs, (num_ticks+1)*tick_diff_secs,
                            tick_diff_secs)

    fo_windows = [[[fractional_occupancy(sliding_window(m.statepath,
                                                        window_size,
                                                        int(window_size / 10)), i)
                    for window_size in windows]
                   for i in range(num_states)]
                  for m in models]

    fo_window_mean = [[np.array([np.mean(fo_windows[m][i][w])
                                 for w in range(len(windows))])
                       for i in range(num_states)]
                      for m in range(len(models))]
    fo_window_var = [[np.array([np.var(fo_windows[m][i][w])
                                for w in range(len(windows))])
                      for i in range(num_states)]
                     for m in range(len(models))]

    for i in range(num_states):

        fig, ax1 = plt.subplots()
        ax1.set_title('Sliding Window Fractional Occupancy')
        ax1.set_xlabel('Window Size (s)')
        ax1.set_ylabel('Fractional Occupancy')
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels)

        for j, m in enumerate(models):
            ax1.plot(windows, fo_window_mean[j][i], label=m.name)
            ax1.fill_between(windows,
                             fo_window_mean[j][i] + fo_window_var[j][i] / 2,
                             fo_window_mean[j][i] - fo_window_var[j][i] / 2,
                             alpha=0.5)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('Variance')
        for j, m in enumerate(models):
            ax2.plot(windows, fo_window_var[j][i], label=m.name, linestyle='--')

        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, 'sliding_fo_state_{}.pdf'.format(i + 1)))
        plt.close()


def plot_kl_matrix(models: List[Model], fig_dir):
    num_states = models[0].K
    comparisons = list(combinations(models, 2))

    f, axarr = plt.subplots(1, len(comparisons),
                            figsize=[3 + 1.5 * len(comparisons), 3])

    f.suptitle('Negative Log KL Divergence Between Models')
    for i, comparison in enumerate(comparisons):
        m1 = comparison[0]
        m2 = comparison[1]
        kl_matrix = np.zeros((num_states, num_states))
        for j in range(num_states):
            for k in range(num_states):
                kl_matrix[j, k] = -np.log(kl_divergence(
                        np.zeros(m1.D), m1.covariances[j],
                        np.zeros(m2.D), m2.covariances[k]))
        if len(comparisons) > 1:
            axarr[i].imshow(kl_matrix, interpolation='nearest')
            axarr[i].set_xlabel('{} States'.format(m2.name))
            axarr[i].set_ylabel('{} States'.format(m1.name))
            axarr[i].set_xticks(np.arange(num_states))
            axarr[i].set_xticklabels(np.arange(1, num_states + 1))
            axarr[i].set_yticks(np.arange(num_states))
            axarr[i].set_yticklabels(np.arange(1, num_states + 1))
        else:
            axarr.imshow(kl_matrix, interpolation='nearest')
            axarr.set_xlabel('{} States'.format(m2.name))
            axarr.set_ylabel('{} States'.format(m1.name))
            axarr.set_xticks(np.arange(num_states))
            axarr.set_xticklabels(np.arange(1, num_states + 1))
            axarr.set_yticks(np.arange(num_states))
            axarr.set_yticklabels(np.arange(1, num_states + 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(fig_dir, 'kl_matrix.pdf'))
    plt.close()



def plot_correlation_matrix(models: List[Model], fig_dir):
    num_states = models[0].K
    comparisons = list(combinations(models, 2))

    f, axarr = plt.subplots(1, len(comparisons),
                            figsize=[3 + 1.5 * len(comparisons), 3])

    f.suptitle('Temporal Correlations Between Models')
    for i, comparison in enumerate(comparisons):
        m1 = comparison[0]
        m2 = comparison[1]
        corr_matrix = np.zeros((num_states, num_states))
        for j in range(num_states):
            for k in range(num_states):
                if not (np.std(m1.statepath_onehot[:, j]) == 0. or
                        np.std(m2.statepath_onehot[:, k]) == 0.):
                    corr_matrix[j, k] = np.corrcoef(
                        m1.statepath_onehot[:, j],
                        m2.statepath_onehot[:, k])[1, 0]
                else:
                    corr_matrix[i, j] = np.nan
        if len(comparisons) > 1:
            axarr[i].imshow(corr_matrix, interpolation='nearest')
            axarr[i].set_xlabel('{} States'.format(m2.name))
            axarr[i].set_ylabel('{} States'.format(m1.name))
            axarr[i].set_xticks(np.arange(num_states))
            axarr[i].set_xticklabels(np.arange(1, num_states + 1))
            axarr[i].set_yticks(np.arange(num_states))
            axarr[i].set_yticklabels(np.arange(1, num_states + 1))
        else:
            axarr.imshow(corr_matrix, interpolation='nearest')
            axarr.set_xlabel('{} States'.format(m2.name))
            axarr.set_ylabel('{} States'.format(m1.name))
            axarr.set_xticks(np.arange(num_states))
            axarr.set_xticklabels(np.arange(1, num_states + 1))
            axarr.set_yticks(np.arange(num_states))
            axarr.set_yticklabels(np.arange(1, num_states + 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(fig_dir, 'correlation_matrix.pdf'))
    plt.close()


def plot_timecourses(models: List[Model], fig_dir, start=100, length=500,
                     freq=250, hard=True):
    f, axarr = plt.subplots(len(models), 1, figsize=[15, 3 + len(models)])
    if len(models) > 1:
        for i, m in enumerate(models):
            axarr[i].set_title('{} State Sequence'.format(m.name))
            if hard:
                axarr[i].stackplot(np.arange(start, start+length),
                                   m.statepath_onehot[start:start+length].T)
            else:
                axarr[i].stackplot(np.arange(start, start+length),
                                   m.gamma[start:start+length].T)
            axarr[i].set_ylim(0, 1)
            axarr[i].set_xlim(start, start+length)
            axarr[i].set_xlabel('Time (s)')
    else:
        axarr.set_title('{} State Sequence'.format(models[0].name))
        if hard:
            axarr.stackplot(np.arange(start, start + length),
                               models[0].statepath_onehot[start:start + length].T)
        else:
            axarr.stackplot(np.arange(start, start + length),
                               models[0].gamma[start:start + length].T)
        axarr.set_ylim(0, 1)
        axarr.set_xlim(start, start + length)
        axarr.set_xlabel('Time (s)')
    plt.setp(axarr, xticks=np.arange(start, start+length, length/10),
             xticklabels=np.arange(start, start+length, length/10)/freq)
    plt.tight_layout()
    if hard:
        path = os.path.join(fig_dir, 'timecourses_hard.pdf')
    else:
        path = os.path.join(fig_dir, 'timecourses.pdf')
    plt.savefig(path)
    plt.close()

#
# def plot_timecourses(models: List[Model], fig_dir, start=100, length=500,
#                      freq=250):
#     f, axarr = plt.subplots(len(models)+1, 1, figsize=[15, 3 + len(models)])
#     i = 0
#     m = models[i]
#     axarr[i].set_title('{} State Sequence'.format(m.name))
#     axarr[i].stackplot(np.arange(start, start+length),
#                        m.statepath_onehot[start:start+length].T)
#     axarr[i].set_ylim(0, 1)
#     axarr[i].set_xlim(start, start+length)
#     axarr[i].set_xlabel('Time (s)')
#     i = 1
#     m = models[i]
#     axarr[i].set_title('{} State Sequence'.format(m.name))
#     axarr[i].stackplot(np.arange(start, start+length),
#                        m.statepath_onehot[start:start+length].T)
#     axarr[i].set_ylim(0, 1)
#     axarr[i].set_xlim(start, start+length)
#     axarr[i].set_xlabel('Time (s)')
#     i = 1
#     m = models[i]
#     axarr[i].set_title('HMM_A State Sequence'.format(m.name))
#     axarr[i].stackplot(np.arange(start, start+length),
#                        m.statepath_onehot[start:start+length].T)
#     axarr[i].set_ylim(0, 1)
#     axarr[i].set_xlim(start, start+length)
#     axarr[i].set_xlabel('Time (s)')
#     i = 2
#     m = models[1]
#     axarr[i].set_title('{} Gammas'.format(m.name))
#     axarr[i].stackplot(np.arange(start, start+length), m.gamma[start:start+length].T)
#     axarr[i].set_ylim(0, 1)
#     axarr[i].set_xlim(start, start+length)
#     axarr[i].set_xlabel('Time (s)')
#     plt.setp(axarr, xticks=np.arange(start, start+length+1, length/10),
#              xticklabels=np.arange(start, start+length+1, length/10)/freq)
#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, 'timecourses.pdf'))


def plot_autocorrelation(models: List[Model], fig_dir):
    num_states = models[0].K
    model_autocorr = [[autocorr(m.gamma[:, i]) for i in range(num_states)]
                      for m in models]
    confidence_interval = 1.96 / models[0].gamma.shape[0]

    for i in range(num_states):
        plt.figure(figsize=[5, 5])
        plt.title('Autocorrelation of Model Gammas, State {}'.format(i+1))
        # axarr[i].set_title('State {}'.format(i+1))
        for a, m in zip(model_autocorr, models):
            plt.plot(a[i], label=m.name)
        plt.axhline(confidence_interval, c='r', linestyle='--', label='95% CI')
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(fig_dir, 'autocorrelation_{}.pdf'.format(i+1)))
        plt.close()


# def plot_hidden(model_gamma, model_hidden, fig_dir):
#     [layers, cells] = model_hidden.shape[-2:]
#     states = model_gamma.shape[-1]
#     hidden_correlation = np.empty([layers * cells, states])
#
#     for l in range(layers):
#         for h in range(cells):
#             for z in range(states):
#                 corr = np.corrcoef(model_hidden[:, l, h], model_gamma[:, z])
#                 hidden_correlation[l * h + h, z] = corr[1, 0]
#
#     if layers > 1:
#         f, axarr = plt.subplots(1, layers, figsize=[layers * 2, 5], sharex=True,
#                                 sharey=True)
#         fb = ['Forward', 'Backwards']
#         for l in range(layers):
#             axarr[l].set_title('{} Layer {}'.format(fb[l], 1))
#             axarr[l].imshow(hidden_correlation[l * cells:(l + 1) * cells, :],
#                             cmap='RdBu_r',
#                             interpolation='nearest',
#                             vmax=np.max(hidden_correlation),
#                             vmin=np.min(hidden_correlation), origin='lower',
#                             aspect='auto')
#         axarr[0].set_xlabel('States')
#         axarr[0].set_ylabel('Hidden States')
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, 'correlation_mat_hidden.pdf'))
#     else:
#         plt.figure(figsize=[3, 5])
#         plt.imshow(hidden_correlation, cmap='RdBu_r', interpolation='nearest',
#                    vmax=np.max(hidden_correlation),
#                    vmin=np.min(hidden_correlation),
#                    origin='lower', aspect='auto')
#         plt.savefig(os.path.join(fig_dir, 'correlation_mat_hidden.pdf'))
#
#     for k in range(8):
#         start = 500
#         length = 500
#         ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
#         ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1)
#         ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
#         ax1.set_title('State {} Gamma Timecourse'.format(k + 1))
#         ax1.plot(np.arange(start, start + length),
#                  model_gamma[start:start + length, k])
#
#         top_n = 3
#         max_idx = hidden_correlation[:cells, k].argsort()[-1:-top_n - 1:-1]
#         max_vals = hidden_correlation[max_idx, k]
#         corr_range = np.linspace(np.min(hidden_correlation),
#                                  np.max(hidden_correlation), 100)
#
#         colors = plt.cm.RdBu_r(np.linspace(0, 1, 100))
#         # Get closest color for each
#         color_idx = np.abs(max_vals[:, np.newaxis] - corr_range).argmin(axis=1)
#         ax2.set_title('Top {} Correlated GRU Hidden States'.format(top_n))
#         for i in range(1, top_n + 1):
#             ax2.plot(np.arange(start, start + length),
#                      model_hidden[start:start + length,
#                      int(np.floor(max_idx[top_n - i] / cells)),
#                      max_idx[top_n-i] % cells],
#                      color=colors[color_idx[top_n - i]])
#
#         min_idx = hidden_correlation[:cells, k].argsort()[:top_n]
#         min_vals = hidden_correlation[min_idx, k]
#
#         min_color_idx = np.abs(min_vals[:, np.newaxis] - corr_range).argmin(
#             axis=1)
#         ax3.set_title('Top {} Anti-Correlated GRU Hidden States'.format(top_n))
#         for i in range(1, top_n + 1):
#             ax3.plot(np.arange(start, start + length),
#                      model_hidden[start:start + length,
#                      int(np.floor(min_idx[top_n - i] / cells)),
#                      min_idx[top_n-i] % cells],
#                      color=colors[min_color_idx[
#                          top_n - i]])
#
#         ax1.set_xticklabels([])
#         ax1.set_xlim(start, start + length)
#         ax2.set_xticklabels([])
#         ax2.set_xlim(start, start + length)
#         ax3.set_xlim(start, start + length)
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, 'state_{}_hidden.pdf'.format(k)))





