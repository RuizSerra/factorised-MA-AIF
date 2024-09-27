'''
Retrieve simulation data from database, plot, and store plots to disk.

Author: Jaime Ruiz Serra
Date:   2024-09-23
'''

import argparse
import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import utils.plotting
import utils.database
import utils.timeseries

utils.plotting.DPI = 120
utils.plotting.SHOW_LEGEND = False
utils.plotting.ONLY_LEFT_Y_LABEL = True
utils.plotting.TIGHT_LAYOUT = True


def main(args):

    # Retrieve data from database --------------------------------------------------
    logging.info(f'Retrieving data matching {args.timestamp} from database {args.db_path}')
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=(
            'SELECT * FROM metadata '
            f'WHERE timestamp LIKE "%{args.timestamp}%" '
            # 'AND description LIKE "%{DESCRIPTION}%" '
        )
    )
    if len(metadata) == 0:
        logging.info(f'❌ No data found matching timestamp {args.timestamp}. Aborting.')
        exit()
    game_transitions = pickle.loads(metadata.iloc[0]['game_transitions'])
    agent_kwargs = pickle.loads(metadata.iloc[0]['agent_kwargs'])
    commit_sha = metadata.iloc[0]['commit_sha']
    description = metadata.iloc[0]['description']
    timestamp = metadata.iloc[0]['timestamp']
    num_players = metadata.iloc[0]['num_agents']
    num_actions = metadata.iloc[0]['num_actions']

    experiments = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=(
            'SELECT * FROM timeseries '
            f'WHERE timestamp LIKE "%{timestamp}%" '
            f'AND commit_sha LIKE "%{commit_sha}%" '
        )
    )
    logging.info(f'Found {len(experiments)} matching experiments')

    # Create output directory ------------------------------------------------------
    if len(experiments) > 0:
        output_dir = os.path.join(
            args.figures_dir, 
            f'{timestamp}-{num_players}x{num_actions}-{description.replace(" ", "_").replace("/", "_")}'
        )
        try:
            os.makedirs(output_dir, exist_ok=False)
        except:
            raise FileExistsError(f'❌ Output directory {output_dir} already exists. Aborting.')
        logging.info(f'Created output directory {output_dir}')
    else:
        raise ValueError('No matching experiments found.')

    # Store metadata in readable format --------------------------------------------
    with open(os.path.join(output_dir, 'metadata.md'), 'w') as f:
        f.write(f'# Metadata\n\n')
        f.write(f'**Description**: {description}\n\n')
        f.write(f'**Timestamp**: `{timestamp}`\n\n')
        f.write(f'**Commit SHA**: `{commit_sha}`\n\n')
        f.write(f'**Game transitions**:\n\n')
        for i, (name, _, duration) in enumerate(game_transitions):
            f.write(f'{i+1}. {name} ({duration} steps)\n')
        f.write('\n')
        f.write(f'**Agent kwargs**:\n\n')
        for i, kwargs in enumerate(agent_kwargs):
            f.write(f'- Agent _{chr(105+i)}_\n')
            for k, v in kwargs.items():
                f.write(f'    - {k}: {v}\n')
        f.write('\n')

    # First plot all the experiment trials, superimposed ---------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in range(len(experiments)):
        # Load data
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        seed = loaded_vars['seed']
        q_u_hist = loaded_vars['q_u']
        # trajectory = q_u_hist[T_MIN:T_MAX, :, 0].copy()

        (
            ambiguity,   # each of shape (T, num_actions**policy_length, policy_length)
            risk, 
            salience, 
            pragmatic_value, 
            novelty
        ) = utils.plotting.unstack(loaded_vars['EFE_terms'], axis=-1)

        utils.plotting.plot_expected_efe_ensemble(
            loaded_vars['q_u'], 
            loaded_vars['EFE'], 
            np.zeros_like(risk),    # Placeholder arrays for the other terms to reduce clutter
            np.zeros_like(ambiguity), 
            np.zeros_like(salience), 
            np.zeros_like(pragmatic_value), 
            np.zeros_like(novelty), 
            ax1
        )

        utils.plotting.plot_vfe_ensemble(
            loaded_vars['VFE'],
            np.zeros_like(loaded_vars['energy']),
            np.zeros_like(loaded_vars['complexity']),
            np.zeros_like(loaded_vars['entropy']),
            np.zeros_like(loaded_vars['accuracy']),
            ax2,
        )

    utils.plotting.highlight_transitions(game_transitions, ax1)
    utils.plotting.highlight_transitions(game_transitions, ax2)

    # Save the figure to disk
    fig.savefig(os.path.join(output_dir, 'ts-ensemble-all-trials.png'))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/ts-ensemble-all-trials.png')

    # Plot a single experiment -----------------------------------------------------
    i = np.random.randint(len(experiments))
    loaded_vars = utils.database.load_single_timeseries(experiments, i)

    variables_history = loaded_vars
    num_actions = metadata.iloc[0]['num_actions']
    seed = loaded_vars['seed']

    t_min = 0
    t_max = None
    suptitle = f'Factorised MA-AIF (seed={seed})'

    plot_configs = [
        {'plot_fn': utils.plotting.plot_vfe_ensemble,
        'args': (
            np.array(variables_history['VFE']), 
            np.array(variables_history['energy']), 
            np.array(variables_history['complexity']), 
            np.array(variables_history['entropy']), 
            np.array(variables_history['accuracy']), )},
        {'plot_fn': utils.plotting.plot_expected_efe_ensemble,
        'args': (
            np.array(variables_history['q_u']),
            np.array(variables_history['EFE']),
            risk,
            ambiguity,
            salience,
            pragmatic_value,
            novelty,)},
        {'plot_fn': utils.plotting.plot_ensemble_policies, 
        'args': (np.array(variables_history['q_u']), game_transitions,)},   
        {'plot_fn': utils.plotting.plot_action_history,
        'args': (np.array(variables_history['u']), num_players, num_actions, )},
    ]

    # Create a figure with subplots for each agent, arranged in a (num_plots)x(num_agents) grid
    n_rows = len(plot_configs)
    figsize = (10, int(n_rows*2.2)+2)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, dpi=utils.plotting.DPI)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust spacing

    for row_idx in range(n_rows):
        ax = axes[row_idx] if n_rows > 1 else axes
        plot_configs[row_idx]['plot_fn'](
            *plot_configs[row_idx]['args'], 
            ax=ax,
            # t_min=t_min, t_max=t_max,  # TODO
        )
        if game_transitions and not t_max:  # TODO: show transitions even if t_max provided
            utils.plotting.highlight_transitions(game_transitions, ax)
        if utils.plotting.TIGHT_LAYOUT and row_idx < n_rows - 1:
            ax.set_xlabel(None)

    # Adjust layout for better spacing and overall title
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for the suptitle
    if utils.plotting.TIGHT_LAYOUT:
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Save the figure to disk
    fig.savefig(os.path.join(output_dir, f'ts-ensemble-{seed}.png'))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/ts-ensemble-{seed}.png')

    # Plot per-agent timeseries ----------------------------------------------------
    plot_configs = utils.plotting.make_default_config(loaded_vars)
    seed = loaded_vars['seed']

    fig = utils.plotting.plot(
        plot_configs=plot_configs, 
        game_transitions=game_transitions,
        num_players=num_players, 
        suptitle=(
            f'Factorised MA-AIF (seed={seed})'
        ),
        figsize=(14, 22),
        # t_min=725, t_max=745,
    )

    # Save the figure to disk
    fig.savefig(os.path.join(output_dir, f'ts-individual-{seed}.png'))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/ts-individual-{seed}.png')

    # Plot state-space trajectory --------------------------------------------------
    from functools import reduce

    logging.info(f'Proecessing {len(experiments)} experiment trajectories')

    fig = plt.figure()
    if num_players == 2:
        # raise NotImplementedError('This code is not ready for 2-player games')
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    elif num_players == 3:
        ax = fig.add_subplot(211, projection='3d')
        ax2 = fig.add_subplot(212, projection='3d')

    agent_kwargs_same = reduce(
        lambda x, y: x == y and x,
        agent_kwargs
    )
    logging.info((
        'Homogeneous agents: will permute space where applicable.' 
        if agent_kwargs_same else 
        'Heterogeneous agents: will not permute space.'
    ))

    # Reference trajectory
    i = 0
    loaded_vars = utils.database.load_single_timeseries(experiments, i)
    seed = loaded_vars['seed']
    q_u_hist = loaded_vars['q_u']

    dataset = np.empty((len(experiments), len(q_u_hist[args.t_min:args.t_max]), num_players))
    distances = []
    seeds = []
    for i in range(len(experiments)):
        
        logging.info(f'{i+1}/{len(experiments)}')
        
        # Load data
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        seed = loaded_vars['seed']
        q_u_hist = loaded_vars['q_u']
        trajectory = q_u_hist[args.t_min:args.t_max, :, 0].copy()

        # Plot ORIGINAL trajectory
        utils.plotting.plot_state_space_trajectory(
            np.array(trajectory), 
            ax, 
            gradient=True,
            # lines=False,
            additional_trajectory_kwargs={
                'alpha': 0.3, 
                # 'color': 'red',
            })
        
        # Use reduce operation to check if all dicts in agent_kargs are the same
        if agent_kwargs_same:
            # Permute agents to a "canonical" order
            trajectory = utils.timeseries.permute_agents(trajectory.copy())
        
        # Save trajectory and seed
        dataset[i] = trajectory
        seeds.append(seed)

        # Plot trajectory
        utils.plotting.plot_state_space_trajectory(
            np.array(trajectory), 
            ax2, 
            gradient=True,
            # lines=False,
            additional_trajectory_kwargs={
                'alpha': 0.3, 
                # 'color': 'red',
            })
        
        # Highlight specific points
        # ax.scatter(*trajectory[300], color='red', s=40, zorder=10)
        # ax.scatter(*trajectory[950], color='orange', s=40, zorder=10)

    args.t_min = 0 if args.t_min is None else args.t_min
    args.t_max = len(q_u_hist) if args.t_max is None else args.t_max
    ax.set_title(f"Original trajectories ({args.t_min} < t < {args.t_max})")
    ax2.set_title(f"Normalised trajectories ({args.t_min} < t < {args.t_max})")
    if num_players == 3:
        # Change azimuth and elevation
        ax.view_init(azim=40, elev=30)
        ax2.view_init(azim=40, elev=30)

    # Save the figure to disk
    fig.savefig(os.path.join(output_dir, f'ss-transform.png'))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/ss-transform.png')

    # Clustering -------------------------------------------------------------------
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    n_clusters = min(args.n_clusters, len(dataset))

    # Normalize the data
    scaler = TimeSeriesScalerMeanVariance()
    time_series_data_scaled = scaler.fit_transform(dataset)

    # KMeans clustering with DTW
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
    labels = model.fit_predict(time_series_data_scaled)

    unique_labels, counts = np.unique(labels, return_counts=True)
    logging.info(f'Cluster member counts {[(u, c) for u, c in zip(unique_labels, counts)]}')

    # Plot clusters
    if num_players == 3:
        from sklearn.decomposition import PCA

        # Fit PCA to all trajectories at once
        all_data_3d = dataset.reshape(-1, num_players)
        pca = PCA(n_components=2)
        pca.fit(all_data_3d)

    # Plotting --------------------------------------------------------------------
    fig, axs = plt.subplots(1, n_clusters, figsize=(3*n_clusters, 3))
    fig.suptitle('Clusters')
    axs = axs.flatten()

    for i in range(len(experiments)):
        
        if num_players == 3:
            # Retrieve 3D trajectory data and project to 2D
            data_3d = dataset[i]
            data_2d = pca.transform(data_3d)
        elif num_players == 2:
            data_2d = dataset[i]
        
        cluster = labels[i]

        # Plot the 2D projection of the trajectory in its cluster plot
        axs[cluster].plot(data_2d[:, 0], data_2d[:, 1], alpha=0.3, marker='o', markersize=1) 
        axs[cluster].set_title(f'{counts[cluster]} match' + ('es' if counts[cluster] > 1 else ''))
        axs[cluster].axis('off')

        # Plot (the 2D projection of) the vertices of the unit cube
        corner_kwargs = dict(
            marker='^', s=40, zorder=10, alpha=1.,
        )

        if num_players == 3:
            # 3C -------------------------------------------------
            reference_points = pca.transform(np.array([
                [1., 1., 1.],
            ]))
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='3C', color='#39c904', **corner_kwargs)

            # 2C -------------------------------------------------
            reference_points = pca.transform(np.array([
                [1., 1., 0.],
                [1., 0., 1.],
                [0., 1., 1.],
            ]))
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='2C', color='#fc6f03', **corner_kwargs)

            # 1C -------------------------------------------------
            reference_points = pca.transform(np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]))
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='1C', color='#c90411', **corner_kwargs)

            # 0C -------------------------------------------------
            reference_points = pca.transform(np.array([
                [0., 0., 0.],
            ]))
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='0C', color='#fc03ad', **corner_kwargs)
        
        elif num_players == 2:
            # 2C -------------------------------------------------
            reference_points = np.array([
                [1., 1.],
            ])
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='2C', color='#39c904', **corner_kwargs)

            # 1C -------------------------------------------------
            reference_points = np.array([
                [1., 0.],
                [0., 1.],
            ])
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='1C', color='#fc6f03', **corner_kwargs)

            # 0C -------------------------------------------------
            reference_points = np.array([
                [0., 0.],
            ])
            axs[cluster].scatter(
                reference_points[:, 0], reference_points[:, 1], 
                label='0C', color='#fc03ad', **corner_kwargs)

    # Save the figure to disk
    fig.savefig(os.path.join(output_dir, f'ss-clusters.png'))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/ss-clusters.png')

    logging.info('\n😎 Done!')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--db-path', type=str, default='experiment-results.db')
    argparser.add_argument('--timestamp', type=str, default='20240923-145722')
    argparser.add_argument('--figures-dir', type=str, default=os.path.join(os.environ['HOME'], 'Desktop/MAAIF-figures'))
    argparser.add_argument('--t-min', type=int, default=None)
    argparser.add_argument('--t-max', type=int, default=None)
    argparser.add_argument('--n-clusters', type=int, default=6)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f'\nArguments: {args}')

    main(args)