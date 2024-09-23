
# factorised-MA-AIF

Factorised Active Inference Agents for Multi-Agent Interactions

[Jaime Ruiz-Serra](https://github.com/RuizSerra), [Patrick Sweeney](https://github.com/patricesweeney), [Michael Harré](https://github.com/M-Harre) (2024)

## Start here

1. [`notebooks/`](./notebooks/)
    - `01 INFG 2x2.ipynb`: Jupyter notebook with simulation and plotting of iterated normal-form games (2-player, 2-action)
    - `01 INFG 3x2.ipynb`: Jupyter notebook with simulation and plotting of iterated normal-form games (3-player, 2-action)
    - `99 FAQ.ipynb`: Additional notebook with some explanations and examples
2. [`docs/schematic-diagram.png`](./docs/schematic-diagram.png): (Ontological) diagram of the factorised active inference agent (in time)
3. [`docs/computational-graph.png`](./docs/computational-graph.png): Computational graph of the factorised active inference agent (what goes on under the hood)

## Codebase structure

- [`agent.py`](./agent.py): Factorised active inference agent
- [`games.py`](./games.py): Normal-form game payoff matrices
- [`notebooks/`](./notebooks/): Jupyter notebooks and supporting code
    - `simulation.py`: Simulation functions, to run experiments using the agents and games
    - `plotting.py`: Plotting functions, to plot the results of the simulations
    - `database_utils.py`: Database utilities, to store and retrieve simulation results
    - `timeseries_utils.py`: Time series utilities, to get metrics and apply transforms to time series data
    - `analysis.py`: Functions for information-theoretic analysis
- [`docs/`](./docs/): Documentation and figures
