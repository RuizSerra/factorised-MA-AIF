'''
Run multiple simulations of the game and store the results in a database.

Author: Jaime Ruiz Serra
Date:   2024-09-23
'''

from itertools import combinations
import argparse
import logging
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append('../')

from games import *
import utils.simulation


def tile_matrices(A, B, C, D):
    '''
    Given four matrices A, B, C, D, return a new matrix with the following structure:
        A B
        C D
    '''

    top_row = torch.cat([A, B], dim=1)   # Shape: (2, 4)
    bottom_row = torch.cat([C, D], dim=1)  # Shape: (2, 4)

    return torch.cat([top_row, bottom_row], dim=0)  # Shape: (4, 4)

GAME_DURATION = 500

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-repeats', type=int, default=20)
    argparser.add_argument('--db-path', type=str, default='experiment-results.db')
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Game configuration ----------------------------------------------------------

    GAME_NAMES = {
        'Ha': harmony_2player,
        'PD': prisoners_dilemma_2player,
        'Ch': chicken_2player,
        'SH': stag_hunt_2player,
    }

    game_pairs = (
        list(combinations(GAME_NAMES.keys(), 2))   # Ha-PD, Ha-Ch, Ha-SH, PD-Ch, PD-SH, Ch-SH
        + [(g, )*2 for g in GAME_NAMES.keys()]     # Ha-Ha, PD-PD, Ch-Ch, SH-SH
    )

    for game_pair in game_pairs:
        
        # Combine the two games into a single "games-within-games" matrix
        GWG = tile_matrices(
            GAME_NAMES[game_pair[0]],
            torch.ones(2, 2),
            torch.ones(2, 2),
            GAME_NAMES[game_pair[1]],
        )

        # Run simulations --------------------------------------------------------------
        game_transitions = [     # TODO: No game transitions for now
            ('-'.join(game_pair), GWG, GAME_DURATION),
        ]

        # Agent configuration ---------------------------------------------------------
        num_players = game_transitions[0][1].ndim

        # Define the start and end points --------------------------------------
        # Hardcoded for 2 actions
        # start = np.array([0, 1])
        # end = np.array([1, 0])
        # steps = 11
        # priors = np.linspace(start, end, steps)
        
        META_AGENT_KWARGS = [
            [
                # dict(
                #     A_prior_type='identity',
                #     B_prior_type='uniform',
                #     A_learning=False,
                #     B_learning=True,
                #     # beta_1=10,
                #     # interoception=True,
                #     # E_prior=torch.tensor([e, 1-e])
                #     D_prior=[torch.tensor(p).float() for _ in range(num_players)],
                # ),
                dict(
                    beta_1=30,
                    # interoception=True,
                    A_prior=99,
                    A_learning=False,
                    B_prior=0,
                    B_learning=True,
                    B_BMR='softmax',
                    decay=0.8,
                    B_learning_rate=1,
                    compute_novelty=True,
                    # D_prior=[torch.tensor([0.001, 0.999]).float() for _ in range(num_players)]
                ),
            ]
            # for b1 in [10, 50, 100, 200, 500]
            # for p in priors
            # for d in np.arange(0, 2.1, 0.2)
            # for b in [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1]
        ]
        
        logging.info(f'Running simulation for game transitions: {game_transitions}')
        description = '-'.join([game[0] for game in game_transitions])
        
        for agent_kwargs in META_AGENT_KWARGS:
            
            logging.debug(f'Agent configuration: {agent_kwargs}')

            # If only one set of agent_kwargs is provided, use it for all players
            if len(agent_kwargs) == 1:
                agent_kwargs = agent_kwargs * num_players
            elif len(agent_kwargs) != num_players:
                raise ValueError('"agent_kwargs" must have length equal to the number of players, or length 1.')

            # Run simulation --------------------------------------------------------------
            utils.simulation.simulate_parallel(
                game_transitions, 
                agent_kwargs=agent_kwargs,
                num_repeats=args.num_repeats,
                results_db_path=args.db_path,
                description=description,
            )

    logging.info('\n😎 Done!')