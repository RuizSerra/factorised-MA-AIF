

# Data Collection

When running experiments using the `simulation.IteratedGame` class, the method `IteratedGame.run()` takes a(n optional) `collect_variables` list of the names of agent attributes to be collected each timestep. 
This can include any agent attributes, for example `agent.q_u`, or `agent.gamma`, or `agent.dynamic_precision`.
The `IteratedGame` class will collect these variables for each agent at each timestep, and store them in a dictionary. The dictionary is returned by the `IteratedGame.run()` method. 
Generally here we specify variables (attributes) that change every time step.

The function `simulation.simulate()` is a wrapper that allows one to run `IteratedGame.run()` multiple times with different seeds and collect the results in a list of dictionaries. 
If a database path is provided, the results will be stored in the database. 
The results are stored in two differnent tables:

### Metadata Table

This table contains the metadata for each run, including the seed, the parameters of the run, and the initial state of the agents.
- `commit_sha` (string) the commit hash of the code used to run the experiment (short version, i.e. 8 characters)
- `timestamp` (string) in `YYYYMMDD-HHMMSS` format
- `description`: (string; optional) experiment description string, provided by the user
- `agent_kwargs`: (pickled dictionary) initial agent configuration parameters
- `game_transitions`: (pickled tuple) specified game transitions, including game label, payoff matrix, and duration of the game

Example of the `metadata` table:

![Metadata table](db-metadata-table.png)

### Timeseries Table

This table contains the timeseries data for each agent at each timestep, pickled (I am going to hell for this)

Example of the `timeseries` table (not showing all columns):

![Timeseries table](db-timeseries-table.png)


## Examples

To select experiments by `description`, first we need to retrieve the matching `commit_sha` and `timestamp` from the `metadata` table. Then we can use these values to filter the `experiments` table. Example:

```python
timestamp_query = '20240920-12'
sql_query = f"SELECT * FROM metadata WHERE timestamp LIKE '%{timestamp_query}%' AND description LIKE '%2x2 no interoception%'"
metadata = data_utils.retrieve_timeseries_matching(sql_query=sql_query, db_path=RESULTS_DB_PATH)

commit_sha = metadata.iloc[0]['commit_sha']
timestamp = metadata.iloc[0]['timestamp']

metadata  # <--- this will show the metadata for the selected experiments

sql_query = f"SELECT * FROM timeseries WHERE timestamp LIKE '%{timestamp}%' AND commit_sha LIKE '%{commit_sha}%'"
experiments = data_utils.retrieve_timeseries_matching(sql_query=sql_query, db_path=RESULTS_DB_PATH)

experiments  # <--- this will show the timeseries data for the selected experiments
```

## Interface

The `notebooks/data_utils.py` module provides functions to interact with the database.