'''
Functions for plotting results

Authors: Pat Sweeney, Jaime Ruiz Serra
Date:    2024/08
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

DPI = 72
SHOW_LEGEND = True
ONLY_LEFT_Y_LABEL = False
TIGHT_LAYOUT = False
MARGIN = 0.05  # Percentage margin for ylim
LEGEND_IDX = 1

# Define consistent color settings
VFE_COLOR = '#ff00ff'
ENERGY_COLOR = '#ccccff'
COMPLEXITY_COLOR = '#8a2be2'
ELBO_COLOR = '#000080'
ENTROPY_COLOR = '#468499'
ACCURACY_COLOR = '#00ced1'
EFE_COLOR = '#990000'
RISK_COLOR = '#ff7f50'
AMBIGUITY_COLOR = '#f6546a'
EXP_ELBO_COLOR = '#065535'
PRAGMATIC_COLOR = '#66cdaa'
SALIENCE_COLOR = '#00ff00'
NOVELTY_COLOR = '#fc03d0'
PRECISION_COLOR = '#000000'
POLICY_ENTROPY_COLOR = '#1F948B'
ACTION_COLORS = ['#03dffc', '#f70ce4', '#fcad03', '#fc03d0', '#03fcad', '#ad03fc', '#adfc03', '#fcad03', '#fc03ad', '#03adfc']

label_font_size = 8  # Define a consistent font size for labels
LINEWIDTH = 0.7

# Set consistent font size and style for all labels and titles
plt.rc('font', size=label_font_size)  # Default text size
plt.rc('axes', titlesize=14, labelsize=label_font_size)  # Axes titles and labels
plt.rc('xtick', labelsize=label_font_size)  # X-tick labels
plt.rc('ytick', labelsize=label_font_size)  # Y-tick labels
plt.rc('legend', fontsize=label_font_size)  # Legend font size
plt.rc('font', family='serif')  # Use serif fonts
plt.rc('text', usetex=False)  # Disable LaTeX rendering for simplicity (enable if needed)

# ==============================================================================
# Helper functions
# ==============================================================================

def unstack(a, axis=0):
    '''Source: https://stackoverflow.com/a/64097936/21740893'''
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]

def get_action_labels(num_actions):
    '''Define action labels based on the number of actions'''
    if num_actions == 2:
        return ['$q(\hat u = \mathtt{c})$', '$q(\hat u = \mathtt{d})$']
    elif num_actions == 3:
        return ['High', 'Medium', 'Low']
    else:
        # return [f'Action {i}' for i in range(num_actions)]
        # Return binary representation of actions
        return [f'{i:0{int(np.log2(num_actions))}b}' for i in range(num_actions)]
    
# def get_figure_size(num_players, num_actions, base_width=6, base_height=4):
#     '''Determine figure size based on the number of players and actions'''
#     width = base_width + num_players * 2
#     height = base_height + num_actions * 2
#     return (width, height)

def highlight_transitions(game_transitions, ax, t_min=0, t_max=None):
    
    durations = [g[-1] for g in game_transitions]
    t_max = sum(durations) if t_max is None else t_max
    t = np.arange(t_min, t_max)
    
    for game_idx, (label, payoffs, duration) in enumerate(game_transitions):
        # Highlight game transitions (every other game)
        if game_idx % 2 == 0:
            ax.fill_between(
                t, 
                ax.get_ylim()[0], 
                ax.get_ylim()[1], 
                where=(sum(durations[:game_idx]) <= t) & (t < sum(durations[:game_idx+1])), 
                color='gray', 
                edgecolor='none',
                alpha=0.1)
        # Game labels
        y_mid = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
        ax.text(
            sum(durations[:game_idx]) + (1/2)*durations[game_idx],  # x position
            y_mid,  # y position: halfway up the y-axis
            label, 
            color='gray', 
            alpha=0.4,
            ha='center')

# ==============================================================================
# Main plotting function 
# ==============================================================================

def make_default_config(variables_history):

    # Data preprocessing
    required_variables_names = [
        'VFE', 
        'energy', 
        'entropy', 
        'accuracy', 
        'complexity', 
        'EFE', 
        'EFE_terms',
        'gamma',
        'q_s',
        'q_u', 
        'u', 
        'A', 
        'B',
        # 'payoff',
    ]
    for required_variable_name in required_variables_names:
        if not isinstance(variables_history[required_variable_name], np.ndarray):
            variables_history[required_variable_name] = np.array(
                variables_history[required_variable_name]
            ).squeeze()
        else:
            variables_history[required_variable_name] = (
                variables_history[required_variable_name].copy().squeeze()
            )

    (
        ambiguity,   # each of shape (T, num_actions**policy_length, policy_length)
        risk, 
        salience, 
        pragmatic_value, 
        novelty
    ) = unstack(variables_history['EFE_terms'], axis=-1)

    # variables_history['EFE_terms'][0][0].shape == torch.Size([2, 1, 5])

    # Define what plots we want to compute and show
    plot_configs = [
        # {'plot_fn': plot_precision, 
        # 'args': (variables_history['gamma'],)},
        {'plot_fn': plot_vfe, 
        'args': (
            variables_history['VFE'], 
            variables_history['energy'], 
            variables_history['complexity'], 
            variables_history['entropy'], 
            variables_history['accuracy'])},
        {'plot_fn': plot_efe,      
        'args': (
            variables_history['EFE'], 
            risk,
            ambiguity,
            pragmatic_value,
            salience,
            novelty,)},
        {'plot_fn': plot_expected_efe,      
        'args': (
            variables_history['q_u'],
            variables_history['EFE'],
            risk,
            ambiguity,
            pragmatic_value,
            salience,
            novelty,)},
        {'plot_fn': plot_policy_heatmap, 
        'args': (variables_history['q_u'], )},
        {'plot_fn': plot_policy_entropy, 
        'args': (variables_history['q_u'], )},
        # {'plot_fn': plot_inferred_policy_heatmap, 
        # 'args': (variables_history['q_s'], )},
        # {'plot_fn': plot_A, 
        # 'args': (variables_history['A'], )},
    ]

    return plot_configs

def plot(plot_configs=None, num_players=0, 
         suptitle=None, game_transitions=None, figsize=(14, 22), height_ratios=None,
         t_min=0, t_max=None):
    
    # Create a figure with subplots for each agent, arranged in a (num_plots)x(num_agents) grid
    n_rows = len(plot_configs)
    fig, axes = plt.subplots(n_rows, num_players, figsize=figsize, dpi=DPI, height_ratios=height_ratios)
    # fig.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust spacing

    for row_idx in range(n_rows):
        for col_idx in range(num_players):
            ax = axes[row_idx, col_idx]
            plot_configs[row_idx]['plot_fn'](
                *plot_configs[row_idx]['args'], 
                ax=ax, i=col_idx,
                t_min=t_min, t_max=t_max,
            )
            if (
                game_transitions 
                and not t_max
                and plot_configs[row_idx]['plot_fn'].__name__ not in [
                    'plot_policy_heatmap', 
                    'plot_inferred_policy_heatmap', 
                    'plot_A'
                ]
            ):  # TODO: show transitions even if t_max provided
                highlight_transitions(game_transitions, ax)
            if TIGHT_LAYOUT and row_idx < n_rows - 1:
                ax.set_xlabel(None)

    # Adjust layout for better spacing and overall title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for the suptitle
    if TIGHT_LAYOUT:
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()

    return fig

# ==============================================================================
# Plotting functions
# ==============================================================================

def plot_vfe(
        vfe_history, 
        energy_history, complexity_history, 
        entropy_history, accuracy_history, 
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(vfe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    summed_vfe = vfe_history[t_min:t_max, i].sum(axis=1)  
    summed_energy = energy_history[t_min:t_max, i].sum(axis=1)
    summed_entropy = entropy_history[t_min:t_max, i].sum(axis=1)
    summed_complexity = complexity_history[t_min:t_max, i].sum(axis=1)
    summed_accuracy = accuracy_history[t_min:t_max, i].sum(axis=1)
    
    vfe_plot = ax.plot(
        x_range, 
        summed_vfe, 
        label='$F[q, o]$', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)
    
    ax2 = ax#.twinx()
    
    energy_plot = ax2.plot(
        x_range, 
        summed_energy, 
        label='Energy', 
        color=ENERGY_COLOR, 
        linestyle=':', 
        linewidth=LINEWIDTH,
        alpha=0.7)
    entropy_plot = ax2.plot(
        x_range, 
        summed_entropy, 
        label='Entropy', 
        color=ENTROPY_COLOR, 
        linestyle=':', 
        linewidth=LINEWIDTH,
        alpha=0.7)
    
    # complexity_plot = ax2.plot(
    #     x_range, 
    #     summed_complexity, 
    #     label='Complexity', 
    #     color=COMPLEXITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    # accuracy_plot = ax2.plot(
    #     x_range, 
    #     -summed_accuracy, 
    #     label='Accuracy', color=ACCURACY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    ax.set_title(f'VFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = vfe_history[t_min:t_max].sum(axis=2).min()
    ymax = vfe_history[t_min:t_max].sum(axis=2).max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$F[q, o]$', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
        # ax2.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Combine both legends in one box
        lines = [
            vfe_plot[0], 
            # complexity_plot[0], accuracy_plot[0], 
            energy_plot[0], entropy_plot[0]
            ]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='lower right')
        # ax.legend(loc='upper left')
        # ax2.legend(loc='upper right')

def plot_vfe_per_factor(
        vfe_history, 
        energy_history, complexity_history, 
        entropy_history, accuracy_history, 
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(vfe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)
    
    for j in range(vfe_history.shape[-1]):
        vfe_plot = ax.plot(
            x_range, 
            vfe_history[t_min:t_max, i, j], 
            label='VFE', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)
    
    ax.set_title(f'VFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = vfe_history[t_min:t_max].min()
    ymax = vfe_history[t_min:t_max].max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Combine both legends in one box
        lines = [vfe_plot[0]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='lower right')

def plot_vfe_complexity_energy(vfe_history, energy_history, complexity_history, ax, i):
    summed_vfe = vfe_history[:, i].sum(axis=1)  
    summed_energy = energy_history[:, i].sum(axis=1)
    summed_complexity = complexity_history[:, i].sum(axis=1)
    
    ax.plot(
        range(len(vfe_history)), 
        summed_vfe, 
        label='VFE', color=VFE_COLOR, linewidth=LINEWIDTH)
    
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_complexity)), 
        summed_complexity, 
        label='Complexity', 
        color=COMPLEXITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_energy)), 
        summed_energy, 
        label='Energy', color=ENERGY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    ax.set_title(f'VFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_vfe_accuracy_entropy(vfe_history, entropy_history, accuracy_history, ax, i):
    summed_vfe = vfe_history[:, i].sum(axis=1)  
    summed_entropy = entropy_history[:, i].sum(axis=1)
    summed_accuracy = accuracy_history[:, i].sum(axis=1)
    
    ax.plot(
        range(len(summed_vfe)), 
        summed_vfe, 
        label='VFE', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)
    
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_accuracy)), 
        -summed_accuracy, 
        label='Accuracy', color=ACCURACY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_entropy)), 
        summed_entropy, 
        label='Entropy', color=ENTROPY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    ax.set_title(f'VFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_efe(
        efe_history, 
        risk,
        ambiguity,
        pragmatic_value,
        salience,
        novelty,
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(efe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # Only plot EFE terms if they are non-zero (i.e. if actual data was provided)
    # (as a way to stub out the plotting of EFE terms)
    # FIXME: Use kwargs instead?
    plot_EFE_terms = (not np.all(risk == 0))

    if plot_EFE_terms:
        ax2 = ax#.twinx()

    num_actions = efe_history.shape[-1]
    for a in range(num_actions):
        efe_plot = ax.plot(
            x_range, 
            efe_history[t_min:t_max, i, a], 
            label='$G[\hat u=\mathtt{'+['c', 'd'][a]+'}]$',  color=ACTION_COLORS[a], linewidth=LINEWIDTH)
        
        if plot_EFE_terms:
            # risk_plot = ax2.plot(
            #     x_range, 
            #     risk[t_min:t_max, i, a], 
            #     label='$\mathcal{R}[\hat u=\mathtt{'+['c', 'd'][a]+'}]$', 
            #     color=ACTION_COLORS[a], 
            #     linestyle=':', linewidth=LINEWIDTH, alpha=0.5)
            # ambiguity_plot = ax2.plot(
            #     x_range, 
            #     ambiguity[t_min:t_max, i, a], 
            #     label='$\mathcal{A}[\hat u=\mathtt{'+['c', 'd'][a]+'}]$', 
            #     color=ACTION_COLORS[a], 
            #     linestyle='--', linewidth=LINEWIDTH, alpha=0.5)
            pv_plot = ax2.plot(
                x_range, 
                -pragmatic_value[t_min:t_max, i, a], 
                label=r'$-\rho[\hat u=\mathtt{'+['c', 'd'][a]+'}]$', 
                color=ACTION_COLORS[a], linestyle=(0, (1, 5)), linewidth=LINEWIDTH, alpha=0.7)
            salience_plot = ax2.plot(
                x_range, 
                salience[t_min:t_max, i, a], 
                label=r'$\varsigma[\hat u=\mathtt{'+['c', 'd'][a]+'}]$', 
                color=ACTION_COLORS[a], linestyle=(0, (5, 7)), linewidth=LINEWIDTH, alpha=0.7)
            novelty_plot = ax.plot(
                x_range, 
                novelty[t_min:t_max, i, a], 
                label=r'$\eta[\hat u=\mathtt{'+['c', 'd'][a]+'}]$', color=ACTION_COLORS[a], linestyle=':', linewidth=LINEWIDTH
            )

    ax.set_title(f'EFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = min(
        efe_history[t_min:t_max].min(), 
        risk[t_min:t_max].min(), 
        ambiguity[t_min:t_max].min(),
        -pragmatic_value[t_min:t_max].max(),  # negative
        salience[t_min:t_max].min(),
        novelty[t_min:t_max].min()
    )
    ymax = max(
        efe_history[t_min:t_max].max(), 
        risk[t_min:t_max].max(), 
        ambiguity[t_min:t_max].max(),
        -pragmatic_value[t_min:t_max].min(),  # negative 
        salience[t_min:t_max].max(),
        novelty[t_min:t_max].max()
    )
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$G[\hat u]$', color='black', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Combine both legends in one box
        lines = [
            efe_plot[0], 
            # risk_plot[0], ambiguity_plot[0], 
            pv_plot[0], 
            salience_plot[0], 
            novelty_plot[0]
            ]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right')
        # ax.legend(loc='upper left')
        # ax2.legend(loc='upper right')


def plot_expected_efe(
        q_u_history, efe_history,
        risk_history, ambiguity_history,
        pragmatic_value_history, salience_history,
        novelty_history,
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(efe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # Only plot EFE terms if they are non-zero (i.e. if actual data was provided)
    # (as a way to stub out the plotting of EFE terms)
    plot_EFE_terms = (not np.all(risk_history == 0))
    plot_EFE_terms = False

    # Get EFE, Pragmatic Value, Salience, Risk, and Ambiguity over all actions
    # Calculate the dot product with q_u_history for each agent (i.e. expected value undre action dist)
    expected_efe = np.empty((q_u_history.shape[0], q_u_history.shape[1]))
    weighted_pragmatic_value = np.empty_like(expected_efe)
    weighted_salience = np.empty_like(expected_efe)
    weighted_risk = np.empty_like(expected_efe)
    weighted_ambiguity = np.empty_like(expected_efe)
    weighted_novelty = np.empty_like(expected_efe)

    for t in range(q_u_history.shape[0]):
        for a in range(q_u_history.shape[1]):
            expected_efe[t, a] = np.dot(q_u_history[t, a], efe_history[t, a])
            weighted_pragmatic_value[t, a] = np.dot(q_u_history[t, a], pragmatic_value_history[t, a])
            weighted_salience[t, a] = np.dot(q_u_history[t, a], salience_history[t, a])
            weighted_risk[t, a] = np.dot(q_u_history[t, a], risk_history[t, a])
            weighted_ambiguity[t, a] = np.dot(q_u_history[t, a], ambiguity_history[t, a])
            weighted_novelty[t, a] = np.dot(q_u_history[t, a], novelty_history[t, a])
            
    summed_efe = expected_efe[t_min:t_max, i]  
    summed_risk = weighted_risk[t_min:t_max, i]
    summed_ambiguity = weighted_ambiguity[t_min:t_max, i]
    summed_salience = weighted_salience[t_min:t_max, i]
    summed_pragmatic_value = -weighted_pragmatic_value[t_min:t_max, i]
    summed_novelty = weighted_novelty[t_min:t_max, i]
    
    # Plot EFE on the primary y-axis
    efe_plot = ax.plot(
        x_range, 
        summed_efe, 
        label=r'$\langle \boldsymbol{\mathsf{G}} \rangle$', color=EFE_COLOR, linewidth=LINEWIDTH)
    
    if plot_EFE_terms:
        # Create a secondary y-axis for Risk, Ambiguity, Salience, Pragmatic Value, and Novelty)
        ax2 = ax#.twinx()
        # risk_plot = ax2.plot(
        #     x_range, 
        #     summed_risk, 
        #     label='r', color=RISK_COLOR, linestyle=':', linewidth=LINEWIDTH)
        # ambiguity_plot = ax2.plot(
        #     x_range, 
        #     summed_ambiguity, 
        #     label='a', color=AMBIGUITY_COLOR, linestyle='--', linewidth=LINEWIDTH)
        pragmatic_value_plot = ax2.plot(
            x_range, 
            summed_pragmatic_value, 
            label='$-$Pragmatic value', color=PRAGMATIC_COLOR, linestyle='--', linewidth=LINEWIDTH)
        salience_plot = ax2.plot(
            x_range, 
            summed_salience, 
            label='Salience', color=SALIENCE_COLOR, linestyle=':', linewidth=LINEWIDTH)
        # novelty_plot = ax2.plot(
        #     x_range, 
        #     summed_novelty, 
        #     label='Novelty', color=NOVELTY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    # Set labels and title
    ax.set_title(f'Expected EFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = min(
        expected_efe[t_min:t_max].min(), 
        summed_risk[t_min:t_max].min(), 
        summed_ambiguity[t_min:t_max].min(),
        summed_salience[t_min:t_max].min(),
        summed_pragmatic_value[t_min:t_max].min(),
        summed_novelty[t_min:t_max].min()
    )
    ymax = max(
        expected_efe[t_min:t_max].max(), 
        summed_risk[t_min:t_max].max(), 
        summed_ambiguity[t_min:t_max].max(),
        summed_salience[t_min:t_max].max(),
        summed_pragmatic_value[t_min:t_max].max(),
        summed_novelty[t_min:t_max].max()
    )
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel(r'$\langle \boldsymbol{\mathsf{G}} \rangle$', color='black', fontsize=label_font_size)
        # if plot_EFE_terms:
        #     ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        if plot_EFE_terms:
            lines = [
                efe_plot[0], 
                # risk_plot[0], ambiguity_plot[0], 
                pragmatic_value_plot[0], salience_plot[0],
                # novelty_plot[0]
                ]
            labels = [line.get_label() for line in lines]
            ax2.legend(lines, labels, loc='lower right')
            # ax2.legend(loc='upper right')
            # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            pass
            # ax.legend(loc='upper left')


def plot_efe_risk_ambiguity(expected_efe, weighted_risk, weighted_ambiguity, ax, i):
    summed_efe = expected_efe[:, i]  
    summed_risk = weighted_risk[:, i]
    summed_ambiguity = weighted_ambiguity[:, i]
    
    ax.plot(
        range(len(summed_efe)), 
        summed_efe, 
        label='EFE', color=EFE_COLOR, linewidth=LINEWIDTH)
    
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_risk)), 
        summed_risk, 
        label='Risk', color=RISK_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_ambiguity)), 
        summed_ambiguity, 
        label='Ambiguity', color=AMBIGUITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    ax.set_title(f'EFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('EFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_efe_salience_pragmatic_value(expected_efe, weighted_salience, weighted_pragmatic_value, weighted_novelty=None, ax=None, i=0):
    summed_efe = expected_efe[:, i]
    summed_salience = -weighted_salience[:, i]
    summed_pragmatic_value = -weighted_pragmatic_value[:, i]
    summed_novelty = -weighted_novelty[:, i] if weighted_novelty is not None else None
    
    # Plot EFE on the primary y-axis
    ax.plot(
        range(len(summed_efe)), 
        summed_efe, 
        label='EFE', color=EXP_ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)
    
    # Create a secondary y-axis for Salience, Pragmatic Value, and Novelty
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_salience)), 
        summed_salience, 
        label='Salience', color=SALIENCE_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_pragmatic_value)), 
        summed_pragmatic_value, 
        label='Pragmatic value', color=PRAGMATIC_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    # Plot Novelty if provided
    if summed_novelty is not None:
        ax2.plot(
            range(len(summed_novelty)), 
            summed_novelty, 
            label='Novelty', color=NOVELTY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    # Set labels and title
    ax.set_title(f'EFE Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('EFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Set legends for both axes
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_policy_heatmap(
        q_pi_history, 
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(q_pi_history) if t_max is None else t_max

    num_actions = q_pi_history.shape[-1]
    action_labels = get_action_labels(num_actions)
    cax = ax.imshow(
        q_pi_history[t_min:t_max, i, :].T, 
        vmin=0, vmax=1,
        origin='upper', 
        aspect='auto', 
        interpolation='nearest'
    )
    
    ax.set_title(f'Policy Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max-t_min, int((t_max-t_min)/5)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max-t_min)/5)))
    ax.set_yticks(range(len(action_labels)))
    ax.set_yticklabels(action_labels)
    # if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
    #     ax.set_ylabel('$q(\hat u)$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_policy_entropy(
        q_pi_history, 
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(q_pi_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # num_actions = q_pi_history.shape[-1]
    # action_labels = get_action_labels(num_actions)
    entropy = -np.sum(q_pi_history[t_min:t_max, i, :] * np.log(q_pi_history[t_min:t_max, i, :] + 1e-9), axis=1)
    ax.plot(
        x_range,
        entropy, 
        label='Policy entropy', color=POLICY_ENTROPY_COLOR, linewidth=LINEWIDTH
    )
    max_ent = np.log(q_pi_history.shape[-1])
    ax.hlines(max_ent, t_min, t_max, color='#ccc', linestyle='--', linewidth=LINEWIDTH, label='Max entropy')
    
    ax.set_title(f'Policy entropy Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylim(-0.1, max_ent+0.1)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$H[q(\hat u)]$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])

def plot_inferred_policy_heatmap(
        s_history, 
        ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(s_history) if t_max is None else t_max

    num_players = s_history.shape[1]
    num_actions = s_history.shape[-1]
    action_labels = get_action_labels(num_actions)
    cax = ax.imshow(
        s_history[t_min:t_max, i, :, :].reshape(-1, num_players*num_actions).T,
        vmin=0, vmax=1,
        origin='upper', 
        aspect='auto', 
        interpolation='nearest'
    )
    
    ax.set_title(f'Agent {chr(105+i)}\'s inferred hidden state (each factor)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max-t_min, int((t_max-t_min)/10)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max-t_min)/10)))
    ax.set_yticks(range(num_actions*num_players))
    ax.set_yticklabels(action_labels * num_players)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$q(s)$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])

# def plot_log_C_modality(log_C_modality_history, ax, i, t_max=None, max_legend_items=9):

#     num_players = log_C_modality_history.shape[1]
#     num_actions = log_C_modality_history.shape[-1]
#     action_labels = get_action_labels(num_actions)
#     cax = ax.imshow(
#         log_C_modality_history[:, i, :, :].reshape(-1, num_players*num_actions).T,
#         # vmin=0, vmax=1,
#         origin='upper', 
#         aspect='auto', 
#         interpolation='nearest'
#     )
#     print(f'log_C_modality (agent {i}) (min, max):\t{log_C_modality_history[:, i, :, :].min():0.4f}, {log_C_modality_history[:, i, :, :].max():0.4f}')
    
#     ax.set_title(f'Agent {chr(105+i)}\'s preferred observation (per factor)', fontsize=label_font_size)
#     ax.set_xlabel('Time step (t)', fontsize=label_font_size)
#     ax.set_yticks(range(num_actions*num_players))
#     ax.set_yticklabels(action_labels * num_players)
#     if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
#         ax.set_ylabel('log_C_modality', fontsize=label_font_size)

# def plot_log_C_opp(log_C_opp_history, ax, i, t_max=None, max_legend_items=9):
#     num_players = log_C_opp_history.shape[1]
#     num_actions = log_C_opp_history.shape[2]
#     all_joint_actions_enum = np.array(np.meshgrid(*[np.arange(num_actions)] * num_players)).T.reshape(-1, num_players)

#     t_max = min(t_max, log_C_opp_history.shape[0]) if t_max else log_C_opp_history.shape[0]  # Ensure t_max is within bounds
#     log_C_values = log_C_opp_history[:t_max, i].reshape(-1, num_actions ** num_players)
    
#     # Calculate the average log C values for each joint action
#     avg_log_C = log_C_values.mean(axis=0)
    
#     # Get the indices of the top max_legend_items joint actions
#     top_indices = np.argsort(avg_log_C)[-max_legend_items:]
    
#     # Plot only the top max_legend_items joint actions
#     for j in top_indices:
#         current_action = all_joint_actions_enum[j]
#         ax.plot(
#             range(t_max), 
#             log_C_values[:, j], 
#             label=str(current_action.tolist()),
#             linewidth=LINEWIDTH
#         )
    
#     ax.set_title(f'Learnt Preferences for Agent {chr(105+i)}', fontsize=label_font_size)
#     ax.set_xlabel('Time step (t)', fontsize=label_font_size)
#     if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
#         ax.set_ylabel('log C (opponent)', fontsize=label_font_size)
#     if SHOW_LEGEND and i == LEGEND_IDX:
#         ax.legend(title='Joint actions', fontsize=8, loc='upper right')


def plot_utility(log_C_opp_history, ax, i, t_max=None, num_samples=1000, epsilon=np.finfo(float).eps, max_legend_items=9):
    num_players = log_C_opp_history.shape[1]
    num_actions = log_C_opp_history.shape[2]
    all_joint_actions_enum = np.array(np.meshgrid(*[np.arange(num_actions)] * num_players)).T.reshape(-1, num_players)

    t_max = min(t_max, log_C_opp_history.shape[0]) if t_max else log_C_opp_history.shape[0]  # Ensure t_max is within bounds
    utility_mean = np.zeros((t_max, num_actions ** num_players))
    utility_std = np.zeros((t_max, num_actions ** num_players))
    
    for t in range(t_max):
        alpha = log_C_opp_history[t, i].reshape(-1)
        samples = np.array([dirichlet.rvs(alpha, size=1) for _ in range(num_samples)]).reshape(num_samples, -1)
        log_samples = np.log(samples + epsilon)
        utility_mean[t, :] = log_samples.mean(axis=0)
        utility_std[t, :] = log_samples.std(axis=0)
    
    # Calculate the average utility values for each joint action
    avg_utility = utility_mean.mean(axis=0)
    
    # Get the indices of the top max_legend_items joint actions
    top_indices = np.argsort(avg_utility)[-max_legend_items:]
    
    # Plot only the top max_legend_items joint actions
    for j in top_indices:
        current_action = all_joint_actions_enum[j]
        ax.plot(
            range(t_max), 
            utility_mean[:, j], 
            label=str(current_action.tolist()),
            color=f'C{j % 10}', 
            linewidth=LINEWIDTH
        )
        ax.fill_between(
            range(t_max), 
            utility_mean[:, j] - utility_std[:, j], 
            utility_mean[:, j] + utility_std[:, j],
            color=f'C{j % 10}', 
            alpha=0.3
        )
    
    ax.set_title(f'Learnt Reward for Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('Reward (log prob)', fontsize=label_font_size)
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(title='Joint actions', fontsize=8, loc='upper right')


def plot_precision(
        gamma_history, ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(gamma_history) if t_max is None else t_max
    x_range = range(t_min, t_max)
    ax.plot(
        x_range, 
        gamma_history[t_min:t_max, i], 
        label='Precision', color=PRECISION_COLOR, linewidth=LINEWIDTH
    )
    
    ax.set_title(f'Precision for Agent {chr(105+i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = gamma_history[t_min:t_max].min()
    ymax = gamma_history[t_min:t_max].max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('Precision', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_A(A_history, ax, i, 
        t_min=0, t_max=None
    ):

    t_max = len(A_history) if t_max is None else t_max

    num_players = A_history.shape[1]
    num_actions = A_history.shape[-1]
    cax = ax.imshow(
        A_history[t_min:t_max, i].reshape(-1, num_players*num_actions*num_actions).T,
        # vmin=0, vmax=1,
        origin='upper', 
        aspect='auto', 
        interpolation='nearest'
    )
    labels = [f'[{chr(105+j)}, {o}, {s}]' 
              for j in range(num_players) 
              for o in range(num_actions) 
              for s in range(num_actions)]
    
    ax.set_title(f'Likelihood Agent {chr(105+i)} (factor, o, s)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max-t_min, int((t_max-t_min)/10)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max-t_min)/10)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('A', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_B(B_history, i, t_min=0, t_max=None):

    t_max = B_history.shape[0] if t_max is None else t_max
    num_players = B_history.shape[1]
    num_actions = B_history.shape[-1]

    labels = [f'[{chr(105+j)}, {o}, {s}]' 
        for j in range(num_players) 
        for o in range(num_actions) 
        for s in range(num_actions)]

    data_per_factor_and_action = [
        B_history[t_min:t_max, i, :, u].reshape((t_max - t_min), -1).T
        for u in range(num_actions)
    ]

    for u_idx, data in enumerate(data_per_factor_and_action):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        cax = ax.imshow(
            data,
            # vmin=0, vmax=1,
            origin='upper', 
            aspect='auto', 
            interpolation='nearest'
        )

        ax.set_title(r'$\boldsymbol{\mathsf{B}}_'+str(chr(105+i))+r'[u=\mathtt{'+str(chr(99+u_idx))+'}]$', fontsize=label_font_size)
        ax.set_xlabel('Time step (t)', fontsize=label_font_size)
        ax.set_xticks(range(0, t_max-t_min, int((t_max-t_min)/10)))
        ax.set_xticklabels(range(t_min, t_max, int((t_max-t_min)/10)))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
            ax.set_ylabel(r'$\boldsymbol{\mathsf{B}}$', fontsize=label_font_size)

        plt.show()



# ==============================================================================
# ENSEMBLE PLOTS
# ==============================================================================

def plot_vfe_ensemble(vfe_history, energy_history, complexity_history, entropy_history, accuracy_history, ax):

    summed_vfe = vfe_history.sum(axis=(1, 2))
    summed_energy = energy_history.sum(axis=(1, 2))
    summed_complexity = complexity_history.sum(axis=(1, 2))
    summed_entropy = entropy_history.sum(axis=(1, 2))
    summed_accuracy = accuracy_history.sum(axis=(1, 2))
    
    ax.plot(
        range(len(summed_vfe)), 
        summed_vfe, 
        label='VFE', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)
    
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_energy)), 
        summed_energy, 
        label='Energy', color=ENERGY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_entropy)), 
        summed_entropy, 
        label='Entropy', color=ENTROPY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_complexity)), 
        summed_complexity, 
        label='Complexity', 
        color=COMPLEXITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_accuracy)), 
        -summed_accuracy, 
        label='Accuracy', color=ACCURACY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    ax.set_title(f'VFE Ensemble ({vfe_history.shape[1]} agents)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
    # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if SHOW_LEGEND:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

def plot_expected_efe_ensemble(
        q_u_history,
        efe_history,
        risk_history,
        ambiguity_history,
        salience_history,
        pragmatic_value_history,
        novelty_history,
        ax,
        ):
    # Get EFE, Pragmatic Value, Salience, Risk, and Ambiguity over all actions
    # Calculate the dot product with q_u_history for each agent (i.e. expected value undre action dist)
    expected_efe = np.empty((q_u_history.shape[0], q_u_history.shape[1]))
    weighted_pragmatic_value = np.empty_like(expected_efe)
    weighted_salience = np.empty_like(expected_efe)
    weighted_risk = np.empty_like(expected_efe)
    weighted_ambiguity = np.empty_like(expected_efe)
    weighted_novelty = np.empty_like(expected_efe)

    for t in range(q_u_history.shape[0]):  # timesteps
        for a in range(q_u_history.shape[1]):  # agents
            expected_efe[t, a] = np.dot(q_u_history[t, a], efe_history[t, a])
            weighted_pragmatic_value[t, a] = np.dot(q_u_history[t, a], pragmatic_value_history[t, a])
            weighted_salience[t, a] = np.dot(q_u_history[t, a], salience_history[t, a])
            weighted_risk[t, a] = np.dot(q_u_history[t, a], risk_history[t, a])
            weighted_ambiguity[t, a] = np.dot(q_u_history[t, a], ambiguity_history[t, a])
            weighted_novelty[t, a] = np.dot(q_u_history[t, a], novelty_history[t, a])
            
    summed_efe = expected_efe.sum(axis=1)  
    summed_risk = weighted_risk.sum(axis=1)
    summed_ambiguity = weighted_ambiguity.sum(axis=1)
    summed_salience = -weighted_salience.sum(axis=1)
    summed_pragmatic_value = -weighted_pragmatic_value.sum(axis=1)
    summed_novelty = -weighted_novelty.sum(axis=1) if weighted_novelty is not None else None
    
    # Plot EFE on the primary y-axis
    ax.plot(
        range(len(summed_efe)), 
        summed_efe, 
        label=r'$\langle \boldsymbol{\mathsf{G}} \rangle$', color=EFE_COLOR, linewidth=LINEWIDTH)
    
    # Create a secondary y-axis for Risk, Ambiguity, Salience, Pragmatic Value, and Novelty
    ax2 = ax#.twinx()
    ax2.plot(
        range(len(summed_risk)), 
        summed_risk, 
        label='Risk', color=RISK_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_ambiguity)), 
        summed_ambiguity, 
        label='Ambiguity', color=AMBIGUITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_salience)), 
        summed_salience, 
        label='Salience', color=SALIENCE_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_pragmatic_value)), 
        summed_pragmatic_value, 
        label='Pragmatic Value', color=PRAGMATIC_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    # Plot Novelty if provided
    if summed_novelty is not None:
        ax2.plot(
            range(len(summed_novelty)), 
            summed_novelty, 
            label='Novelty', color=NOVELTY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    
    # Set labels and title
    ax.set_title(f'Expected EFE of Ensemble ({q_u_history.shape[1]} agents)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel(r'\langle \boldsymbol{\mathsf{G}} \rangle', color='black', fontsize=label_font_size)
    # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if SHOW_LEGEND:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


def plot_state_space_trajectory(
        trajectory, 
        ax=None, 
        legend=False, 
        gradient=False,
        lines=True,
        vertex_markers=True,
        additional_trajectory_kwargs:dict={}):
    '''
    Args:
        trajectory: np.ndarray of shape (T, num_agents)

    >>> q_u_hist = np.array(variables_history['q_u'])
    >>> plotting.plot_state_space_trajectory(q_u_hist[:, :, 0])
    '''

    num_agents = trajectory.shape[1]
    T = trajectory.shape[0]

    corner_kwargs = dict(
            alpha=1.,
            marker='^', 
            s=50,
            zorder=3,
        )
    
    trajectory_kwargs = dict(
        label='Trajectory', linewidth=0.5, marker='o', markersize=2
    ) | additional_trajectory_kwargs  # Merge the defaults and user-provided dictionaries
    
    if gradient:
        from matplotlib.cm import viridis
        from matplotlib.colors import Normalize

        # Create a colormap (viridis) and normalize the time steps
        norm = Normalize(vmin=0, vmax=T-1)
        colors = viridis(norm(np.arange(T)))

    if num_agents == 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Plot the 3D trajectory -----------------------------------------------
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if gradient:
            for t in range(T-1):
                ax.plot(
                    trajectory[t, 0], trajectory[t, 1], trajectory[t, 2], 
                    color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1], trajectory[t:t+2, 2], 
                        color=colors[t], alpha=0.4)
        else:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                **trajectory_kwargs)

        # Plot the corners of the cube -----------------------------------------
        if vertex_markers:
            ax.scatter(
                0, 0, 0,
                label='0C', color='#fc03ad', **corner_kwargs)
            ax.scatter(
                [1, 0, 0],
                [0, 1, 0], 
                [0, 0, 1],
                label='1C', color='#c90411', **corner_kwargs)
            ax.scatter(
                [0, 1, 1],
                [1, 0, 1], 
                [1, 1, 0],
                label='2C', color='#fc6f03', **corner_kwargs)
            ax.scatter(
                1, 1, 1,
                label='3C', color='#39c904', **corner_kwargs)

        # Set labels and limits to define the cube -----------------------------
        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')
        ax.set_zlabel('$q(u_k = c)$')

        # Setting the limits of the cube to be between 0 and 1
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

    elif num_agents == 2:
        
        if ax is None:
            fig, ax = plt.subplots()

        if gradient:
            for t in range(T-1):
                ax.plot(
                    trajectory[t, 0], trajectory[t, 1],
                    color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1], 
                        color=colors[t], alpha=0.4)
        else:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                **trajectory_kwargs)
        
        # Plot the corners of the square ---------------------------------------
        if vertex_markers:
            ax.scatter(
                0, 0,
                label='0C', color='#fc03ad',  **corner_kwargs)
            ax.scatter(
                [1, 0],
                [0, 1], 
                label='1C', color='#fc6f03',  **corner_kwargs)
            ax.scatter(
                1, 1,
                label='2C', color='#39c904',  **corner_kwargs)
            
        # Set labels, limits, etc ----------------------------------------------
        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')
        
        # set axis ratio to equal (square)
        ax.set_aspect('equal', adjustable='datalim')

    if legend:
        ax.legend()
    ax.set_title('State-space trajectory')
    # plt.show()


def plot_rolling_payoffs(data, game_transitions, ax=None):
    '''
    Args:
        data (ndarray): Payoff history of shape (T, num_players)

    >>> plt.figure(figsize=(8, 2), dpi=plotting.DPI)
    >>> ax = plt.gca()
    >>> plotting.plot_rolling_payoffs(np.array(variables_history['payoff']), game_transitions, ax)
    '''

    num_players = data.shape[1]
    T = data.shape[0]
    if ax is None:
        plt.figure(figsize=(8, 2), dpi=DPI)
        ax = plt.gca()

    # Individual payoffs
    for i in range(num_players):
        # plt.scatter(np.arange(len(data[:, i])), data[:, i])
        window = 20
        rolling_mean = np.convolve(data[:, i], np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window, T+1), rolling_mean)

    # Mean payoffs
    plt.plot(np.arange(len(data)), data.mean(axis=1), 
             marker='o', color='gray', linewidth=0.5, markersize=3, alpha=0.2)

    # Plot rolling mean
    window = 20
    rolling_mean = np.convolve(data.mean(axis=1), np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window, T+1), rolling_mean, color='red')

    highlight_transitions(game_transitions, ax)

    plt.title(f'Payoffs: mean and rolling mean ({window}-step window)')
    plt.xlabel('Time step')
    plt.ylabel('Payoff')
    # plt.show()



def plot_action_history(action_history, num_players, num_actions, ax=None):
    '''Plot the action history of the agents with custom colors.
    
    Args:
        action_history (np.array): Action history of the agents shape (num_players, T)
        num_players (int): Number of agents.
        num_actions (int): Number of actions.
    '''

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Custom colors
    colors = ACTION_COLORS[:num_actions]
    cmap = mcolors.ListedColormap(colors)

    if ax is None:
        plt.figure(figsize=(8, 2), dpi=DPI)
        ax = plt.gca()

    # Display the action history with custom colors
    im = ax.imshow(
        action_history.T,
        origin='upper', 
        aspect='auto', 
        interpolation='nearest',
        cmap=cmap  # Apply the custom colormap
    )

    # Set y-ticks and labels
    ax.set_yticks(np.arange(num_players))
    ax.set_yticklabels([f'Agent {i+1}' for i in range(num_players)])
    ax.set_xlabel('Time step')

    # Create custom legend
    defect_patch = mpatches.Patch(color=colors[0], label='Cooperate')
    cooperate_patch = mpatches.Patch(color=colors[1], label='Defect')
    ax.legend(handles=[defect_patch, cooperate_patch], loc='upper right')

    ax.set_title('Action history')

def plot_ensemble_policies(q_u_hist, game_transitions, ax=None):
    
    import utils.timeseries as timeseries

    # Compute metrics
    num_agents = q_u_hist.shape[1]
    if num_agents == 2:  # Two agents
        data = np.hstack((
            q_u_hist[:, :, 0], 
            np.zeros_like(q_u_hist[:, 0, 0])[:, None],
        ))
    elif num_agents == 3:  # Three agents
        data = q_u_hist[:, :, 0]

    t_steady = timeseries.get_t_steady_state(data)
    t_bifurcation = timeseries.get_t_bifurcation(data)

    if ax is None:
        plt.figure(figsize=(8, 2), dpi=DPI)
        ax = plt.gca()
    
    # Plot the ensemble policies
    ax.plot(q_u_hist[:, :, 0])

    # Highlight the steady state and bifurcation points
    ax.vlines(
        [t_bifurcation, t_steady], 
        0, 1, 
        linestyle='--', linewidth=LINEWIDTH, color='red', alpha=0.4,)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('q(u=c)')

# ==============================================================================
# State-space trajectory plots (3D)
# ==============================================================================
def plot_reference_hull(
        ax, 
        face_colors=['#cfaf59', '#ffdc7d', '#ffffff', '#ffffff'],
        face_alphas=[1, 1, 0, 0],
        legend=True, 
    ):

    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    # Define the 8 vertices of a cube
    vertices = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1]])

    # Define the edges of the cube by connecting vertices
    edges = [
        (0, 1),
        (1, 4),
        (0, 4),
        (4, 7),
        (1, 7),
        (0, 7),
    ]

    # Extract the edge coordinates from the vertices
    edge_lines = [(vertices[start], vertices[end]) for start, end in edges]

    # Define triangular faces to shade (based on vertex indices)
    faces = [
        [vertices[0], vertices[1], vertices[4]],  # Bottom
        [vertices[1], vertices[4], vertices[7]],  # Side
        [vertices[0], vertices[4], vertices[7]],  # Ceiling back
        [vertices[0], vertices[1], vertices[7]],  # Ceiling front
    ]

    # Create a Poly3DCollection for the faces and add shading
    for i, face in enumerate(faces):
        face_collection = Poly3DCollection(
            [face], 
            color=face_colors[i], alpha=face_alphas[i], 
            linewidths=1, edgecolors='k', zsort='min')
        face_collection.set_edgecolor('k')
        face_collection.set_zorder(1)
        ax.add_collection3d(face_collection)

    # Plot each edge as a line segment
    for edge in edge_lines:
        ax.plot(*zip(*edge), color="k", linewidth=LINEWIDTH, zorder=2)

    # Plot the corners of the cube
    CORNER_MARKER = '^'
    CORNER_MARKER_SIZE = 80
    ax.scatter(0, 0, 0, label='0C', color='#fc03ad', alpha=1.,
            marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 0, 0, label='1C', color='#c90411', alpha=1.,
            marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 1, 0, label='2C', color='#fc6f03', alpha=1.,
            marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 1, 1, label='3C', color='#39c904', alpha=1.,
            marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)

    # Set the labels and aspect ratio
    ax.set_xlabel('$q(u_i = c)$')
    ax.set_ylabel('$q(u_j = c)$')
    ax.set_zlabel('$q(u_k = c)$')

    # Setting the limits of the cube to be between 0 and 1
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_zlim([-0.05, 1.05])

    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    if legend:
        ax.legend()


def plot_trajectory_flat(trajectory):
    
    boundary_kwargs = dict(
        color='gray', linestyle='--', zorder=-1, alpha=0.5,
    )
    trajectory_kwargs = dict(
        alpha=0.6,
        marker='o',
        linewidth=LINEWIDTH,
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(10, 5), 
        gridspec_kw={'width_ratios': [1, 1], 'wspace': 0})

    # First plot (left side)
    ax1.plot(
        [p[0] for p in trajectory],
        [p[1] for p in trajectory],
        c='#323bf0',
        **trajectory_kwargs
    )
    ax1.plot([0, 1], [0, 1], **boundary_kwargs)
    ax1.plot([0, 1], [0, 0], **boundary_kwargs)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Second plot (right side)
    ax2.plot(
        [p[2] for p in trajectory],
        [p[1] for p in trajectory],
        c='#1b2080',
        **trajectory_kwargs
    )
    ax2.plot([0, 1], [0, 1], **boundary_kwargs)
    ax2.plot([0, 1], [1, 1], **boundary_kwargs)
    ax2.plot([0, 0], [0, 1], **boundary_kwargs)
    # ax2.set_ylabel('Y')
    ax2.set_xlabel('Z')

    # Hide the right spine of the first plot and the left spine of the second plot
    ax1.set_xlim(None, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.set_xlim(0, None)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Share the y-axis between the plots by aligning their limits
    ax2.set_yticks([])  # Optionally hide the y-axis ticks on the second plot

    # Display the plot
    # plt.show()

