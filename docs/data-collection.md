

# Data Collection

When running experiments using the `IteratedGame` class in `simulation.py`, the variables that will be collected each timestep are specified in the `collect_variables` list argument passed to `IteratedGame.run()`. This can include any agent attributes, for example `agent.q_u`, or `agent.gamma`, or `agent.dynamic_precision`.

The `IteratedGame` class will collect these variables for each agent at each timestep, and store them in a dictionary. The dictionary is returned by the `IteratedGame.run()` method.

The function `simulate()` in `simulation.py` is a wrapper that allows one to run `IteratedGame.run()` multiple times with different seeds and collect the results in a list of dictionaries. If a database path is provided, the results will be stored in the database. The results are stored in two differnent tables:

### Metadata Table

This table contains the metadata for each run, including the seed, the parameters of the run, and the final state of the agents.
- `commit_sha` (string) the commit hash of the code used to run the experiment (short version, i.e. 8 characters)
- `timestamp` (string) in `YYYYMMDD-HHMMSS` format
- `description`: (optional) experiment description string, provided by the user
- `agent_kwargs`: initial agent configurations (a pickled dictionary)
- `game_transitions`: specified game transitions, including game label, payoff matrix, and duration of the game (a pickled tuple)

Example of the `metadata` table:

![Metadata table](db-metadata-table.png)

### Timeseries Table

This table contains the timeseries data for each agent at each timestep, pickled (I am going to hell for this)

Example of the `timeseries` table (not showing all columns):

![Timeseries table](db-timeseries-table.png)


