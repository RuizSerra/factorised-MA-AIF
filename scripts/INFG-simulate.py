'''
Run multiple simulations of the game and store the results in a database.

Author: Jaime Ruiz Serra
Date:   2024-09-23
'''

import argparse
import logging

import sys
sys.path.append('../')

from games import *
import utils.simulation

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-repeats', type=int, default=20)
    argparser.add_argument('--db-path', type=str, default='experiment-results.db')
    args = argparser.parse_args()

    # Game configuration ----------------------------------------------------------

    # TODO: possibly move these into separate yaml files (e.g. one for 2-agent games, one for 3-agent games)
    # META_GAME_TRANSITIONS = [
    #     # "Neighbour" game transitions
    #     [   
    #         ('PD', prisoners_dilemma_2player, 500),
    #         ('SH', stag_hunt_2player, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, 500),
    #         ('Ha', harmony_2player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, 500),
    #         ('Ch', chicken_2player, 500),
    #     ],
    #     [   
    #         ('Ch', chicken_2player, 500),
    #         ('PD', prisoners_dilemma_2player, 500),
    #     ],
    #     # "Neighbour" game transitions (reversed)
    #     [   
    #         ('Ch', chicken_2player, 500),
    #         ('Ha', harmony_2player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, 500),
    #         ('SH', stag_hunt_2player, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, 500),
    #         ('PD', prisoners_dilemma_2player, 500),
    #     ],
    #     [
    #         ('PD', prisoners_dilemma_2player, 500),
    #         ('Ch', chicken_2player, 500),
    #     ],
    #     # "Across" game transitions
    #     [   
    #         ('Ch', chicken_2player, 500),
    #         ('SH', stag_hunt_2player, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_2player, 500),
    #         ('Ch', chicken_2player, 500),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_2player, 500),
    #         ('Ha', harmony_2player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_2player, 500),
    #         ('PD', prisoners_dilemma_2player, 500),
    #     ],
    # ]

    # META_GAME_TRANSITIONS = [
    #     # "Neighbour" game transitions
    #     [   
    #         ('PD', prisoners_dilemma_3player, 500),
    #         ('SH', stag_hunt_3player_M3, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, 500),
    #         ('Ha', harmony_3player, 500),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_3player, 500),
    #         ('SH2', stag_hunt_3player_M2, 500),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, 500),
    #         ('Ha', harmony_3player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, 500),
    #         ('Ch', chicken_3player, 500),
    #     ],
    #     [   
    #         ('Ch', chicken_3player, 500),
    #         ('PD', prisoners_dilemma_3player, 500),
    #     ],
    #     # "Neighbour" game transitions (reversed)
    #     [   
    #         ('Ch', chicken_3player, 500),
    #         ('Ha', harmony_3player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, 500),
    #         ('SH', stag_hunt_3player_M3, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, 500),
    #         ('SH2', stag_hunt_3player_M2, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, 500),
    #         ('PD', prisoners_dilemma_3player, 500),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, 500),
    #         ('PD', prisoners_dilemma_3player, 500),
    #     ],
    #     [
    #         ('PD', prisoners_dilemma_3player, 500),
    #         ('Ch', chicken_3player, 500),
    #     ],
    #     # "Across" game transitions
    #     [   
    #         ('Ch', chicken_3player, 500),
    #         ('SH', stag_hunt_3player_M3, 500),
    #     ],
    #     [   
    #         ('SH', stag_hunt_3player_M3, 500),
    #         ('Ch', chicken_3player, 500),
    #     ],
    #     [   
    #         ('Ch', chicken_3player, 500),
    #         ('SH2', stag_hunt_3player_M2, 500),
    #     ],
    #     [   
    #         ('SH2', stag_hunt_3player_M2, 500),
    #         ('Ch', chicken_3player, 500),
    #     ],
    #     [   
    #         ('PD', prisoners_dilemma_3player, 500),
    #         ('Ha', harmony_3player, 500),
    #     ],
    #     [   
    #         ('Ha', harmony_3player, 500),
    #         ('PD', prisoners_dilemma_3player, 500),
    #     ],
    # ]

    META_GAME_TRANSITIONS = [
        [   
            ('SH3', stag_hunt_3player_M3, 500),
        ],
        [   
            ('SH2', stag_hunt_3player_M2, 500),
        ],
    ]

    # Agent configuration ---------------------------------------------------------
    META_AGENT_KWARGS = [
        [
            dict(
                beta_1=10.,
                learn_every_t_steps=24,
                learning_offset=6,
                # E_prior=torch.tensor([e, 1-e])
                theta_prior=[torch.tensor([e, 10.-e]) for _ in range(3)],
            ),
        ]
        for e in torch.arange(1., 10., 0.5)
    ]

    # Run simulations --------------------------------------------------------------
    for game_transitions in META_GAME_TRANSITIONS:
        
        logging.info(f'Running simulation for game transitions: {game_transitions}')
        num_players = game_transitions[0][1].ndim
        description = '-'.join([game[0] for game in game_transitions])
        
        for agent_kwargs in META_AGENT_KWARGS:
            
            logging.debug(f'Agent configuration: {agent_kwargs}')

            # If only one set of agent_kwargs is provided, use it for all players
            if len(agent_kwargs) == 1:
                agent_kwargs = agent_kwargs * num_players
            elif len(agent_kwargs) != num_players:
                raise ValueError('"agent_kwargs" must have length equal to the number of players, or length 1.')

            # Run simulation --------------------------------------------------------------
            experiment_results = utils.simulation.simulate_parallel(
                game_transitions, 
                agent_kwargs=agent_kwargs,
                num_repeats=args.num_repeats,
                results_db_path=args.db_path,
                description=description,
            )

    logging.info('\n😎 Done!')