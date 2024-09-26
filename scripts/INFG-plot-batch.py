'''
Retrieve simulation data from database, plot, and store plots to disk.

Author: Jaime Ruiz Serra
Date:   2024-09-23
'''

import argparse
import os
import concurrent.futures
import logging

import sys
sys.path.append('../')

import utils.database
import utils.plotting

utils.plotting.DPI = 120
utils.plotting.SHOW_LEGEND = False
utils.plotting.ONLY_LEFT_Y_LABEL = True
utils.plotting.TIGHT_LAYOUT = True

import importlib  
plotsie = importlib.import_module("INFG-plot")

def generate_plots(timestamp, args):
    args.timestamp = timestamp
    logging.info(f'\n📊 Plotting data for timestamp {timestamp}')
    plotsie.main(args)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--db-path', type=str, default='experiment-results.db')
    argparser.add_argument('--timestamp', type=str, default='2024')
    argparser.add_argument('--figures-dir', type=str, default=os.path.join(os.environ['HOME'], 'Desktop/MAAIF-figures'))
    argparser.add_argument('--t-min', type=int, default=None)
    argparser.add_argument('--t-max', type=int, default=None)
    argparser.add_argument('--n-clusters', type=int, default=6)

    args = argparser.parse_args()

    metadata = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=(
            'SELECT * FROM metadata '
            f'WHERE timestamp LIKE "%{args.timestamp}%" '
        )
    )

    # Parallel execution
    logging.info(f'\n🚀 Running {len(metadata)} tasks in parallel')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_plots, timestamp, args)
            for timestamp in metadata['timestamp'].values
        ]
        
        # Optionally wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensures any exceptions are raised
