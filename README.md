
# factorised-MA-AIF

Factorised Active Inference Agents for Strategic Multi-Agent Interactions

[Jaime Ruiz-Serra](https://github.com/RuizSerra), [Patrick Sweeney](https://github.com/patricesweeney), [Michael Harré](https://github.com/M-Harre) (AAMAS 2025) [[arXiv:2411.07362](https://arxiv.org/abs/2411.07362)]

> Ruiz-Serra, J., Sweeney, P., and Harré, M., 2025. Factorised Active Inference Agents for Strategic Multi-Agent Interactions, AAMAS 2025, Detroit, USA, May 19-23, 2025. In _Proceedings of the 24th International Conference on Autonomous Agents and Multi-Agent Systems_. Association for Computing Machinery (ACM).

## Start here

1. [`notebooks/`](https://github.com/RuizSerra/factorised-MA-AIF-notebooks) (separate repository with Jupyter notebooks)
    - `01 INFG 2x2.ipynb`: Jupyter notebook with simulation and plotting of iterated normal-form games (2-player, 2-action)
    - `01 INFG 3x2.ipynb`: Jupyter notebook with simulation and plotting of iterated normal-form games (3-player, 2-action)
    - `99 FAQ.ipynb`: Additional notebook with some explanations and examples
2. [Poster](./docs/AAMAS-2025-poster.pdf) and [presentation](./docs/AAMAS-2025-presentation.pdf) for AAMAS 2025, summarising the main results of the paper
3. [`docs/schematic-diagram.png`](./docs/schematic-diagram.png): (Ontological) diagram of the factorised active inference agent (in time)
4. [`docs/computational-graph.png`](./docs/computational-graph.png): Computational graph of the factorised active inference agent (what goes on under the hood)

## Codebase structure

- [`agent.py`](./agent.py): Factorised active inference agent
- [`games.py`](./games.py): Normal-form game payoff matrices
- [`utils/`](./utils/): Utility modules
    - `simulation.py`: Simulation functions, to run experiments using the agents and games
    - `database.py`: Database utilities, to store and retrieve simulation results
    - `timeseries.py`: Time series utilities, to get metrics and apply transforms to time series data
    - `plotting.py`: Plotting functions, to plot the results of the simulations
    - `analysis.py`: Functions for information-theoretic analysis
- [`scripts/`](./scripts/): Command-line scripts to run simulations and plot results
    - `INFG-simulate.py`: Run multiple simulations and store the results in a database
    - `INFG-plot.py`: Plot the results of multiple simulations
- [`docs/`](./docs/): Documentation and figures

## Scripts

```bash
# 1. Set simulation (expriment) configuration
# [for now, we edit `scripts/INFG-simulate.py` to change the simulation parameters]

# 2. Run simulations
nohup python -OO INFG-simulate.py --num-repeats=20 --db-path=my-experiments.db &

# 3. Plot all results
python INFG-plot-batch.py --db-path=my-experiments.db --timestamp=20240928
```