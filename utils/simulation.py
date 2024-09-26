'''
Simulation
'''

import numpy as np
import torch
import torch.nn.functional as F
import random
import concurrent.futures
from datetime import datetime
# import git
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
            game_matrix=game_transitions[0][1],
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
        game_transitions (list) List of tuples containing (label, game payoff matrix, duration)
        agent_kwargs (list) List of dictionaries containing agent configuration. 
            It can either be None (to use defaults), or a list of dictionaries of 
            length equal to the number of players.
        num_repeats (int) Number of simulations to run
        results_db_path (str) Path to the database file
        description (str) Description of the experiment
    '''

    T = sum([g[-1] for g in game_transitions])  # Total number of time steps
    num_players = game_transitions[0][1].ndim  # Number of agents
    num_actions = game_transitions[0][1].shape[0]  # Number of actions

    if agent_kwargs is None:
        agent_kwargs = [{} for i in range(num_players)]
    
    if results_db_path is not None:
        git_repo = git.Repo(search_parent_directories=True)
        commit_sha = git_repo.head.object.hexsha[:8]
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        logging.info(f'Commit: {commit_sha}')
        logging.info(f'Timestamp: {timestamp}')
        logging.info(f'Storing results to database: {results_db_path}')
        utils.database.store_metadata(
            commit_sha, 
            timestamp, 
            description, 
            agent_kwargs, 
            game_transitions, 
            results_db_path
        )
        
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

    # Write results to the database in the main process ------------------------
    if results_db_path is not None:
        for result in experiment_results:
            seed, variables_history = result
            utils.database.store_timeseries(
                commit_sha, 
                timestamp, 
                seed, 
                variables_history, 
                results_db_path
            )

        logging.info(f'Stored results for {num_repeats} simulations in database: {results_db_path}')
        logging.info(f'Experiment ID: {commit_sha} {timestamp}')
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
    num_players = game_transitions[0][1].ndim  # Number of agents
    num_actions = game_transitions[0][1].shape[0]  # Number of actions

    if agent_kwargs is None:
        agent_kwargs = [{} for i in range(num_players)]
    
    if results_db_path is not None:
        git_repo = git.Repo(search_parent_directories=True)
        commit_sha = git_repo.head.object.hexsha[:8]
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        logging.info(f'Commit: {commit_sha}')
        logging.info(f'Timestamp: {timestamp}')
        logging.info(f'Storing results to database: {results_db_path}')
        utils.database.store_metadata(
            commit_sha, 
            timestamp, 
            description, 
            agent_kwargs, 
            game_transitions, 
            results_db_path
        )

    # A list of seeds for each simulation, with no repeated seeds
    experiment_results = []
    seeds = np.random.choice(500, num_repeats, replace=False).tolist()
    for repeat_idx in range(num_repeats):
        logging.info(f'Simulation {repeat_idx+1}/{num_repeats}')
        
        # Initialisation -------------------------------------------------------
        # Initialize agents with their respective game matrices
        agents = [
            Agent(
                id=i, 
                game_matrix=game_transitions[0][1],
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
        
        # Store results --------------------------------------------------------
        if results_db_path is None:  # Store results in memory (return at the end)
            experiment_results.append((ig.seed, variables_history))
        else:  # Store results in database
            utils.database.store_timeseries(
                commit_sha, 
                timestamp, 
                ig.seed, 
                variables_history, 
                results_db_path
            )
            experiment_results.append((ig.seed, commit_sha, timestamp))

    if results_db_path is not None:
        logging.info(f'Stored results for {num_repeats} simulations in database: {results_db_path}')
        logging.info(f'Experiment ID: {commit_sha} {timestamp}')
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
            game_transitions (list) List of tuples containing (label, game payoff matrix, duration)
            seed (int or None) Seed for random number generators
        '''
        
        self.agents = agents
        self.game_transitions = game_transitions
        self.num_players = len(agents)
        self.num_actions = game_transitions[0][1].shape[0]

        # Seeds ----------------------------------------------------------------
        seed = np.random.randint(0, 100) if seed is None else seed
        self.seed = seed
        random.seed(seed)         # Set seed for Python's random module
        np.random.seed(seed)      # Set seed for NumPy
        torch.manual_seed(seed)   # Set seeds for PyTorch
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # For reproducibility in PyTorch (especially on CUDA)
        torch.backends.cudnn.benchmark = False

    def run(self, 
            T:int, 
            transition_duration:int=10, 
            collect_variables:list=[
                's', 
                'u', 
                'gamma', 
                'theta', 
                'VFE', 
                'energy', 
                'entropy', 
                'accuracy', 
                'complexity', 
                'EFE', 
                'risk', 
                'ambiguity', 
                'salience', 
                'pragmatic_value', 
                'novelty', 
                'q_u', 
                'A', 
                'B',
                'payoff',
            ]
        ):
        '''
        Run the simulation for T time steps

        Args:
            T (int) Total number of time steps
            transition_duration (int) Duration of transitions between games (steps)
            num_iterations (int) Number of iterations for variational optimization
            num_samples (int) Number of samples for variational inference

        Returns:
            variables_history (dict) History of variables for each agent
        '''

        # Initialize history variables for data collection
        variables_history = {v: [] for v in collect_variables}

        # Run simulation ---------------------------------------------------------------
        transition_count = 0
        logging.info(f'Simulation started with seed {self.seed}')
        logging.info(f't=0/{T} Initial game is {self.game_transitions[transition_count][0]} ({transition_count+1}/{len(self.game_transitions)} games)')
        for t in range(T):

            # # Print summary for each agent -------------------------------------------
            if t % 50 == 0:
                logging.info(f't={t}/{T}')
                logging.debug('')
                for agent in self.agents:
                    logging.debug(agent)
                    logging.debug("-" * 40)  
            
            # Iterated game logic ------------------------------------------------------
            # Select actions for all agents
            u_all = [agent.select_action() for agent in self.agents]
            u_all_one_hot = F.one_hot(torch.tensor(u_all), self.num_actions).to(torch.float)  # Convert actions to one-hot encoding

            # Each agent infers the state based on their observation 
            for agent in self.agents:
                # Egocentric observation
                o_ego = u_all_one_hot[agent.id].unsqueeze(0)  # Shape (1, n_actions)
                o_rest = torch.cat((u_all_one_hot[:agent.id], u_all_one_hot[agent.id+1:]), dim=0)  # Shape (n_agents-1, n_actions)
                o = torch.cat((o_ego, o_rest), dim=0)  # Shape (n_agents, n_actions)

                # Compute payoffs
                agent.payoff = agent.log_C[
                    tuple([torch.argmax(o_i).item() for o_i in o])  # Indexing the game matrix
                ]  # Payoff for the selected actions
                
                agent.infer_state(o)

                agent.learn()  # Update the agent's model (only does so every agent.learn_every_t_steps steps)

            # Data collection ----------------------------------------------------------
            for varname in variables_history.keys():
                if isinstance(getattr(self.agents[0], varname), list):
                    variables_history[varname].append([
                        [_ for _ in getattr(a, varname)] for a in self.agents  # Deep copy for lists
                    ])
                elif isinstance(getattr(self.agents[0], varname), torch.Tensor):
                    variables_history[varname].append([
                        getattr(a, varname).clone().detach() for a in self.agents  # Deep copy for tensors
                    ])
                else:
                    variables_history[varname].append([
                        getattr(a, varname) for a in self.agents  # Copy for other types
                    ])
            
            # Interpolate game matrices ------------------------------------------------
            # for transitions between different games
            # t_transition = T * (transition_count+1) / len(self.game_transitions)
            t_transition = sum([g[-1] for g in self.game_transitions[:transition_count+1]])  # Cumulative transition time
            if (
                    transition_count != len(self.game_transitions) - 1  # Not the last game
                    and
                    (t > t_transition - transition_duration//2)  # t is after transition start
                    and 
                    (t < t_transition + transition_duration//2)  # t is before transition end
                ):
                l = (t - t_transition) / transition_duration  # Mixing parameter value
                for agent in self.agents:
                    agent.log_C = (     # Linear interpolation of game matrices
                        (1 - l) * self.game_transitions[transition_count][1] 
                        + l * self.game_transitions[transition_count+1][1]
                    )

            elif t == t_transition + transition_duration//2:  # Last time step of transition
                for agent in self.agents:  # Ensure the interpolation completes
                    agent.log_C = self.game_transitions[transition_count+1][1]
                transition_count += 1
                logging.info(f't={t}/{T} Transitioned to {self.game_transitions[transition_count][0]} ({transition_count+1}/{len(self.game_transitions)} games)')

        logging.info(f't={t+1}/{T} Simulation complete')  # hehe
        return variables_history


if __name__ == '__main__':

    from games import prisoners_dilemma_2player

    # Initialisation
    game_transitions = [   
       ('PD', prisoners_dilemma_2player, 200)
    ]

    num_players = game_transitions[0].ndim  # Number of agents
    num_actions = game_transitions[0].shape[0]  # Number of actions
    T = sum([g[-1] for g in game_transitions])  # Total number of time steps
    beta_1_prior = (10, ) * num_players  # Prior alpha (precisionum_players) values for each agent
    decay = (0.8, ) * num_players  # Decay rate for opponent model

    # Initialize agents with their respective game matrices
    agents = [
        Agent(
            id=i, 
            beta_1=beta_1_prior[i],
            decay=decay[i],
            dynamic_precision=True,
            game_matrix=game_transitions[0],
        ) for i in range(num_players)
    ]

    ig = IteratedGame(agents=agents, game_transitions=game_transitions)
    variables_history = ig.run(T)
    print('q_u at t=0', variables_history['q_u'][0])
