"""Shared plotting utilities: bootstrap CI, ranked TF-weight bar plots, and
significant-TF selection."""
import numpy as np
import matplotlib.pyplot as plt


def calculate_ci(bootstrap_weights, confidence_level=95):
    '''
    Function to calculate lower and upper bounds of an n% confidence interval (e.g., a 95% CI)
    '''
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile

    # Calculate the percentiles
    lower_bound = np.percentile(bootstrap_weights, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_weights, upper_percentile, axis=0)

    return lower_bound, upper_bound




def plot_x0s0_ws(coef_bs, tfs, CI, **kwargs):
    '''
    Function to plot the average of fitted weights and their n% CI.
    Only Significant positive or negative weights are shown (e.i., lower_ci>0 or upper_ci<0).
    Slices the top_tfs based on absolute weight magnitude.
    '''

    pos_color = kwargs.get("pos_barcolor", 'crimson')
    neg_color = kwargs.get("neg_barcolor", 'mediumblue')
    label_fontsize = kwargs.get("label_fontsize", 20)
    tick_fontsize = kwargs.get("tick_fontsize", 12)
    figure_height = kwargs.get('fig_size', 6)
    figure_length_coef = kwargs.get('fig_length_coef', 2)
    axis_pad = kwargs.get('axis_pad', 2.5)
    ylim_coef = kwargs.get('ylim_coef', 0.35)
    alpha = kwargs.get('alpha', 0.75)
    top_tfs = kwargs.get('top_tfs', None) # Default to 50

    lower_ci, upper_ci = calculate_ci(coef_bs, confidence_level=CI)
    ws_mean = np.mean(coef_bs, axis=0)

    isort = np.argsort(ws_mean)
    sorted_tfs = tfs[isort]
    sorted_ws = ws_mean[isort]

    # Indices for positive and negative weights
    w_p = np.where(sorted_ws >= 0)[0][::-1] # Ranked high to low
    w_n = np.where(sorted_ws < 0)[0]        # Ranked low to high (most negative first)

    # Identify the indices for significant values
    sig_pos_idx = np.where(lower_ci[isort][w_p] > 0)[0]
    sig_neg_idx = np.where(upper_ci[isort][w_n] < 0)[0]

    # --- NEW: Slice for top_tfs ---
    # We take the first 'top_tfs' significant ones
    sig_pos = sig_pos_idx[:top_tfs] if top_tfs is not None else sig_pos_idx
    sig_neg = sig_neg_idx[:top_tfs] if top_tfs is not None else sig_neg_idx

    # Dynamic limits based on filtered TFs
    ylim_max = np.max(ws_mean) + ylim_coef*np.max(ws_mean)
    ylim_min = np.min(ws_mean) + ylim_coef*np.min(ws_mean)

    # Calculate x-limits based on whichever side has more bars
    n_pos = len(sig_pos)
    n_neg = len(sig_neg)
    xlim_max = max(n_pos, n_neg) + axis_pad
    xlim_min = -axis_pad

    len_xaxis = max(n_pos, n_neg) / 8 * figure_length_coef

    fig, ax1 = plt.subplots(figsize=(len_xaxis, figure_height))

    # Plot the positive bar plot
    ax1.bar(
        range(n_pos),
        sorted_ws[w_p][sig_pos],
        yerr=[
            sorted_ws[w_p][sig_pos] - lower_ci[isort][w_p][sig_pos],
            upper_ci[isort][w_p][sig_pos] - sorted_ws[w_p][sig_pos]
        ],
        ecolor='black',
        color=pos_color,
        alpha=alpha
    )

    ax1.set_ylabel('Fitted Weights (W>0)', fontsize=label_fontsize, color=pos_color)
    ax1.tick_params(axis='y', labelcolor=pos_color, labelsize=tick_fontsize)
    ax1.set_xlabel('Transcription Factors (W>0)', fontsize=label_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize, labelcolor='#404040')
    ax1.set_xticks(range(n_pos))
    ax1.set_xticklabels(sorted_tfs[w_p][sig_pos], rotation=90)
    ax1.set_ylim(0, ylim_max)

    # Create a secondary y-axis for negative values
    ax2 = ax1.twinx()

    # Plot the negative bar plot (Reversed to match visual ranking)
    ax2.bar(
        range(n_neg),
        sorted_ws[w_n][sig_neg][::-1],
        yerr=[
            (sorted_ws[w_n][sig_neg] - lower_ci[isort][w_n][sig_neg])[::-1],
            (upper_ci[isort][w_n][sig_neg] - sorted_ws[w_n][sig_neg])[::-1]
        ],
        ecolor='black',
        color=neg_color,
        alpha=alpha
    )

    ax2.set_ylabel('Fitted Weights (W<0)', fontsize=label_fontsize, color=neg_color)
    ax2.tick_params(axis='y', labelcolor=neg_color, labelsize=tick_fontsize)
    ax2.set_ylim(ylim_min, 0)

    # Add upper x-axis labels for ax2
    ax3 = ax1.twiny()
    ax3.set_xticks(range(n_neg))
    ax3.set_xticklabels(sorted_tfs[w_n][sig_neg][::-1], rotation=90)
    ax3.set_xlabel('Transcription Factors (W<0)', fontsize=label_fontsize, color='black')
    ax3.tick_params(axis='x', labelsize=tick_fontsize, labelcolor='#404040')

    # Align x-limits for all axes
    ax1.set_xlim(xlim_min, xlim_max)
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_xlim(ax1.get_xlim())

    plt.tight_layout()

    return fig

def return_sig_tfs(coef_bs, tfs, CI):

    lower_ci, upper_ci = calculate_ci(coef_bs, confidence_level=CI)

    ws_mean = np.mean(coef_bs, axis = 0)

    isort = np.argsort(ws_mean)
    sorted_tfs = tfs[isort]
    sorted_ws = ws_mean[isort]

    w_p = np.where(sorted_ws>=0)[0][::-1]
    w_n = np.where(sorted_ws<0)[0]

    # Identify the indices for significant positive and negative values
    sig_pos = np.where(lower_ci[isort][w_p] > 0)[0]
    sig_neg = np.where(upper_ci[isort][w_n] < 0)[0]

    sig_neg_tfs = sorted_tfs[w_n][sig_neg]
    sig_pos_tfs = sorted_tfs[w_p][sig_pos]

    return sig_pos_tfs, sig_neg_tfs
