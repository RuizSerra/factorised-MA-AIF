'''
Run multiple simulations of the game and store the results in a database.

Author: Jaime Ruiz Serra
Date:   2024-09-23
'''

import argparse
import logging
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append('../')

from games import *
import utils.simulation

GAME_DURATION = 500

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-repeats', type=int, default=20)
    argparser.add_argument('--db-path', type=str, default='experiment-results.db')
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Game configuration ----------------------------------------------------------

    # TODO: possibly move these into separate yaml files (e.g. one for 2-agent games, one for 3-agent games)
    # META_GAME_TRANSITIONS = [
    #     # "Neighbour" game transitions
    #     [   
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #         ('Ha', harmony_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, GAME_DURATION),
    #         ('Ch', chicken_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ch', chicken_2player, GAME_DURATION),
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #     ],
    #     # "Neighbour" game transitions (reversed)
    #     [   
    #         ('Ch', chicken_2player, GAME_DURATION),
    #         ('Ha', harmony_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, GAME_DURATION),
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #     ],
    #     [
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #         ('Ch', chicken_2player, GAME_DURATION),
    #     ],
    #     # "Across" game transitions
    #     [   
    #         ('Ch', chicken_2player, GAME_DURATION),
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, GAME_DURATION),
    #         ('Ch', chicken_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #         ('Ha', harmony_2player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, GAME_DURATION),
    #         ('PD', prisoners_dilemma_2player, GAME_DURATION),
    #     ],
    # ]

    # META_GAME_TRANSITIONS = [
    #     # "Neighbour" game transitions
    #     [   
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #         ('Ha', harmony_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #         ('Ha', harmony_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, GAME_DURATION),
    #         ('Ch', chicken_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ch', chicken_3player, GAME_DURATION),
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #     ],
    #     # "Neighbour" game transitions (reversed)
    #     [   
    #         ('Ch', chicken_3player, GAME_DURATION),
    #         ('Ha', harmony_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, GAME_DURATION),
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, GAME_DURATION),
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #     ],
    #     [
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #         ('Ch', chicken_3player, GAME_DURATION),
    #     ],
    #     # "Across" game transitions
    #     [   
    #         ('Ch', chicken_3player, GAME_DURATION),
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, GAME_DURATION),
    #         ('Ch', chicken_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ch', chicken_3player, GAME_DURATION),
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, GAME_DURATION),
    #         ('Ch', chicken_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #         ('Ha', harmony_3player, GAME_DURATION),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, GAME_DURATION),
    #         ('PD', prisoners_dilemma_3player, GAME_DURATION),
    #     ],
    # ]

    META_GAME_TRANSITIONS = [
        # [   
        #     ('Ch', chicken_2player, GAME_DURATION),
        # ],
        [   
            ('Ch', chicken_2player, GAME_DURATION),
            ('SH', stag_hunt_2player, GAME_DURATION),
        ],
        [   
            ('Ch', chicken_3player, GAME_DURATION),
            ('SH-g', stag_hunt_3player_M2, GAME_DURATION),
        ],
        [   
            ('Ch', chicken_3player, GAME_DURATION),
            ('SH-r', stag_hunt_3player_M3, GAME_DURATION),
        ],
        [   
            ('Ch', chicken_3player, GAME_DURATION),
            ('SH-g', stag_hunt_3player_M2, GAME_DURATION//2),
            ('SH-r', stag_hunt_3player_M3, GAME_DURATION//2),
        ],
    ]

    # Run simulations --------------------------------------------------------------
    for game_transitions in META_GAME_TRANSITIONS:

        # Agent configuration ---------------------------------------------------------
        num_players = game_transitions[0][1].ndim

        # Define the start and end points --------------------------------------
        # Hardcoded for 2 actions
        start = np.array([0, 1])
        end = np.array([1, 0])
        steps = 11
        priors = np.linspace(start, end, steps)
        
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
                    beta_1=100,
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