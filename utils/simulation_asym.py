import numpy as np
import torch
import torch.nn.functional as F
import random
import concurrent.futures
from datetime import datetime
import logging
from typing import Union

import sys
sys.path.append('../')

from agent import Agent
import utils.database

logging.basicConfig(level=logging.INFO)

def run_single_simulation(game_transitions, agent_kwargs, T, seed):
    # Initialisation -------------------------------------------------------
    agents = [
        Agent(
            id=i,
            game_matrix=game_transitions[0][1][i],  # Assign individual game matrix to each agent
            **agent_kwargs[i]
        ) for i in range(len(agent_kwargs))
    ]

    ig = IteratedGame(
        agents=agents, 
        game_transitions=game_transitions, 
        seed=seed
    )

    # Simulation -----------------------------------------------------------
    variables_history = ig.run(T)
    
    # Return results to main process
    return (seed, variables_history)

def simulate_parallel(
        game_transitions, 
        agent_kwargs:Union[list, None]=None,
        num_repeats=1,
        results_db_path=None,
        description=None,
    ):
    '''
    Args:
        game_transitions (list) List of tuples containing (label, list of game payoff matrices for each agent, duration)
        agent_kwargs (list) List of dictionaries containing agent configuration. 
            It can either be None (to use defaults), or a list of dictionaries of 
            length equal to the number of players.
        num_repeats (int) Number of simulations to run
        results_db_path (str) Path to the database file
        description (str) Description of the experiment
    '''

    T = sum([g[-1] for g in game_transitions])  # Total number of time steps
    num_players = len(game_transitions[0][1])   # Number of agents, based on the number of matrices
    num_actions = game_transitions[0][1][0].shape[0]  # Number of actions, assuming symmetric action space

    if agent_kwargs is None:
        agent_kwargs = [{} for i in range(num_players)]
    
    # Experiment repeats (parallel execution) ----------------------------------
    experiment_results = []
    seeds = np.random.choice(500, num_repeats, replace=False).tolist()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_simulation, 
                game_transitions, agent_kwargs, T, seeds[repeat_idx]
            ) for repeat_idx in range(num_repeats)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            experiment_results.append(future.result())

    return experiment_results

def simulate(
        game_transitions, 
        agent_kwargs:Union[list, None]=None,
        num_repeats=1,
        results_db_path=None,
        description=None,
    ):
    '''
    '''

    T = sum([g[-1] for g in game_transitions])  # Total number of time steps
    num_players = len(game_transitions[0][1])  # Number of agents, based on number of game matrices
    num_actions = game_transitions[0][1][0].shape[0]  # Number of actions

    if agent_kwargs is None:
        agent_kwargs = [{} for i in range(num_players)]
    
    experiment_results = []
    seeds = np.random.choice(500, num_repeats, replace=False).tolist()
    for repeat_idx in range(num_repeats):
        logging.info(f'Simulation {repeat_idx+1}/{num_repeats}')
        
        # Initialize agents with their respective game matrices
        agents = [
            Agent(
                id=i, 
                game_matrix=game_transitions[0][1][i],  # Each agent gets their own game matrix
                **agent_kwargs[i]
            ) for i in range(num_players)
        ]

        # Instantiate the simulator object
        ig = IteratedGame(
            agents=agents, 
            game_transitions=game_transitions, 
            seed=seeds[repeat_idx]
        )

        # Simulation -----------------------------------------------------------
        variables_history = ig.run(T)
        experiment_results.append((ig.seed, variables_history))

    return experiment_results


class IteratedGame:
    
    def __init__(
            self, 
            agents,
            game_transitions,
            seed=None):
        '''
        Initialize the simulation
        
        Args:
            agents (list) List of agents
            game_transitions (list) List of tuples containing (label, list of game payoff matrices for each agent, duration)
            seed (int or None) Seed for random number generators
        '''
        
        self.agents = agents
        self.game_transitions = game_transitions
        self.num_players = len(agents)
        self.num_actions = game_transitions[0][1][0].shape[0]  # Assuming symmetric action space for now

        # Seeds ----------------------------------------------------------------
        seed = np.random.randint(0, 100) if seed is None else seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self, 
            T:int, 
            transition_duration:int=200, 
            collect_variables:list=[
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
            ]
        ):
        '''
        Run the simulation for T time steps

        Args:
            T (int) Total number of time steps
            transition_duration (int) Duration of transitions between games (steps)

        Returns:
            variables_history (dict) History of variables for each agent
        '''

        variables_history = {v: [] for v in collect_variables}

        transition_count = 0
        logging.info(f'Simulation started with seed {self.seed}')
        for t in range(T):

            # Select actions for all agents
            u_all = [agent.select_action() for agent in self.agents]
            u_all_one_hot = F.one_hot(torch.tensor(u_all), self.num_actions).to(torch.float)

            for agent in self.agents:
                o_ego = u_all_one_hot[agent.id].unsqueeze(0)
                o_rest = torch.cat((u_all_one_hot[:agent.id], u_all_one_hot[agent.id+1:]), dim=0)
                o = torch.cat((o_ego, o_rest), dim=0)

                agent.payoff = agent.log_C[
                    tuple([torch.argmax(o_i).item() for o_i in o])
                ]
                
                agent.infer_state(o)
                agent.learn()

            for varname in variables_history.keys():
                variables_history[varname].append([
                    getattr(a, varname) for a in self.agents
                ])

        logging.info(f'Simulation complete')
        return variables_history


if __name__ == '__main__':

    from games import prisoners_dilemma_2player, chicken_3player

    # Initialisation with different game matrices for each agent
    game_transitions = [   
       ('PD', [prisoners_dilemma_2player, prisoners_dilemma_2player], 200),
       ('Ch', [chicken_3player, chicken_3player], 300)
    ]

    num_players = len(game_transitions[0][1])  # Number of agents based on matrices
    agent_kwargs = [{} for i in range(num_players)]  # Example agent configurations

    experiment_results = simulate(game_transitions, agent_kwargs, num_repeats=2)
    print('Simulation Results:', experiment_results)