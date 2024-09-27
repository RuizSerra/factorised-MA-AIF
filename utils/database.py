'''
Database utilities

Author: Jaime Ruiz Serra
Date:   2024-09-20
'''

import logging
import pickle
import sqlite3
import numpy as np
import pandas as pd


# =============================================================================
# Write to database
# =============================================================================
def store_metadata(
        commit_sha, 
        timestamp, 
        description, 
        agent_kwargs, 
        game_transitions, 
        results_db_path
    ):
    '''Store the metadata of the experiment in the database (metadata table)
    
    Args:
        commit_sha (str): The commit SHA of the experiment
        timestamp (str): The timestamp of the experiment
        description (str): The description of the experiment
        agent_kwargs (dict): The agent configuration
        game_transitions (list): The game transitions
        results_db_path (str): The path to the database
    '''
    # Pickle the agent configuration to store in the database
    num_agents = game_transitions[0][1].ndim  # Number of players (rank of game tensor)
    num_actions = game_transitions[0][1].shape[0]  # Number of actions (assuming symmetrical actions)
    pickles = {
        'commit_sha': commit_sha,
        'timestamp': timestamp,
        'description': description,
        'num_agents': num_agents,
        'num_actions': num_actions,
    }
    pickles['agent_kwargs'] = [pickle.dumps(agent_kwargs)]
    pickles['game_transitions'] = [pickle.dumps(game_transitions)]
    # Store in database
    df = pd.DataFrame.from_dict(pickles)
    con = sqlite3.connect(results_db_path)
    df.to_sql('metadata', con, if_exists='append', index=False)
    con.close()

def store_timeseries(commit_sha, timestamp, seed, variables_history, results_db_path):
    # Pickle all iterables in variables_history
    pickles = {
        'commit_sha': commit_sha,
        'timestamp': timestamp,
        'seed': seed,
    }
    for key, value in variables_history.items():
        pickles[key] = [pickle.dumps(np.array(value))]

    # Store in database
    df = pd.DataFrame.from_dict(pickles)
    con = sqlite3.connect(results_db_path)
    df.to_sql('timeseries', con, if_exists='append', index=False)
    con.close()


# =============================================================================
# Read from database
# =============================================================================
def retrieve_timeseries_matching(
        sql_query="SELECT * FROM timeseries", 
        db_path=None):
    '''Retrieve the experiments from the database

    Args:
        sql_query (str): The SQL query to retrieve the experiments
        db_path (str): The path to the database

    Returns:
        pd.DataFrame: The experiments table
    '''
    if db_path is None:
        raise ValueError("Please provide a path to the database")
    conn = sqlite3.connect(db_path)
    experiments = pd.read_sql_query(sql_query, conn)
    conn.close()
    return experiments

def retrieve_timeseries_matching_metadata(
        db_path, 
        table='metadata',
        timestamp_query=None, 
        description_query=None,
        sql_query=None):
    '''Retrieve the experiments from the database

    Args:
        db_path (str): The path to the database
        timestamp_query (str): The timestamp to query
        description_query (str): The description to query
        sql_query (str): The SQL query to retrieve the experiments (overrides the other queries)
    '''   

    if sql_query is None:
        sql_query = f"SELECT * FROM {table}"
        addendums = []
        if timestamp_query is not None:
            addendums.append(f"timestamp LIKE '%{timestamp_query}%'")
        if description_query is not None:
            addendums.append(f"description LIKE '%{description_query}%'")
        if addendums:
            sql_query += ' WHERE ' + ' AND '.join(addendums)

    metadata = retrieve_timeseries_matching(sql_query=sql_query, db_path=db_path)
    assert len(metadata) > 0, f"No matches for '{sql_query}' in the {db_path} database"
    logging.info(f"Found {len(metadata)} matches for '{sql_query}' in the {db_path} database")

    experiments = pd.DataFrame()
    for idx in range(len(metadata)):

        commit_sha = metadata.iloc[idx]['commit_sha']
        timestamp = metadata.iloc[idx]['timestamp']

        sql_query = f"SELECT * FROM timeseries WHERE timestamp LIKE '%{timestamp}%' AND commit_sha LIKE '%{commit_sha}%'"
        experiments = pd.concat(
            [experiments, 
             retrieve_timeseries_matching(sql_query=sql_query, db_path=db_path)], 
            ignore_index=True
        )

    return experiments 

def load_single_timeseries(experiments, idx):
    
    loaded_vars = {}
    for key in experiments.columns:
        if key in ['commit_sha', 'timestamp', 'seed']:
            loaded_vars[key] = experiments[key].iloc[idx]
        elif key == 'index':
            continue
        else:
            loaded_vars[key] = pickle.loads(experiments[key].iloc[idx])
    return loaded_vars