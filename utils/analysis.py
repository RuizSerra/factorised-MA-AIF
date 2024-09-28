'''
Functions for information-theoretic analysis with JIDT

Author: Patrick Sweeney
Date:   Sep 2024
'''

import os
# from collections import defaultdict
# from itertools import combinations
import pandas as pd
import numpy as np
from jpype import startJVM, getDefaultJVMPath, isJVMStarted

import os
from jpype import startJVM, isJVMStarted, getDefaultJVMPath
import matplotlib.pyplot as plt


# Plot setting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'  # Use the Computer Modern Roman font
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math text
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.labelsize'] = 16  # Axis labels
plt.rcParams['axes.titlesize'] = 20  # Title
plt.rcParams['xtick.labelsize'] = 14 # X tick labels
plt.rcParams['ytick.labelsize'] = 14 # Y tick labels

def initialize_jvm(jidt_dir=None, max_memory="4024M"):
    # Check if the JIDT directory is provided
    if jidt_dir is None:
        raise ValueError("Please provide a path to the JIDT installation directory, e.g. '/Users/alice/Downloads/infodynamics-dist-1.6.1/'")

    # Construct the path to the JAR file
    jar_location = os.path.join(jidt_dir, 'infodynamics.jar')

    # Check if the JAR file exists
    if not os.path.isfile(jar_location):
        raise FileNotFoundError(f"The JAR file 'infodynamics.jar' was not found in the directory: {jidt_dir}")

    # Start JVM if it's not already running
    if not isJVMStarted():
        startJVM(
            getDefaultJVMPath(),
            "-ea",
            f"-Xmx{max_memory}",  # Set max memory
            f"-Djava.class.path={jar_location}",  # Set class path to the JAR
            convertStrings=True
        )


def extract_history(variables_history):
    """
    Extract agent data from 'variables_history' and convert it to a DataFrame.
    If 'variables_history' is a list, stack data from each element, and add a column to track which element 
    (or repeat) each data point came from.
    
    Args:
        variables_history (list or dict): Dictionary or list of dictionaries containing action history, 
                                          typically in the form of a nested array.
    
    Returns:
        pd.DataFrame: DataFrame with 'timestep', 'agent_0_action', 'agent_1_action', ..., 'agent_n_action',
                      and 'repeat' to track which element of the list the data originated from (if it's a list).
    """
    # If variables_history is a dictionary, handle as the original function does
    if isinstance(variables_history, dict):
        # Extract 'action' history as a NumPy array
        u_history = np.array(variables_history['u'])
        
        # Determine number of agents from the length of the innermost vectors
        num_agents = u_history.shape[1]
        
        # Generate timestep numbers
        timesteps = np.arange(1, u_history.shape[0] + 1)
        
        # Create a dictionary for the data
        data = {'Timestep': timesteps}
        
        # Add each agent's data to the dictionary
        for agent_id in range(num_agents):
            data[f'agent_{agent_id}_action'] = u_history[:, agent_id]
        
        # Convert the dictionary into a DataFrame
        data = pd.DataFrame(data)
        
        return data
    
    # If variables_history is a list, stack the data with a 'repeat' column
    elif isinstance(variables_history, list):
        all_data = []  # List to hold DataFrames from each repeat
        
        for idx, history in enumerate(variables_history):
            # Extract 'action' history for each dictionary in the list
            u_history = np.array(history['u'])
            
            # Determine number of agents from the length of the innermost vectors
            num_agents = u_history.shape[1]
            
            # Generate timestep numbers
            timesteps = np.arange(1, u_history.shape[0] + 1)
            
            # Create a dictionary for the data
            data = {'Timestep': timesteps}
            
            # Add each agent's data to the dictionary
            for agent_id in range(num_agents):
                data[f'agent_{agent_id}_action'] = u_history[:, agent_id]
            
            # Add the 'repeat' column to signify the list element
            data['Repeat'] = idx + 1
            
            # Convert to DataFrame and append to list
            all_data.append(pd.DataFrame(data))
        
        # Concatenate all the DataFrames from the repeats
        final_data = pd.concat(all_data, ignore_index=True)
        
        return final_data
    
    else:
        raise ValueError("variables_history should be either a dictionary or a list of dictionaries.")


# ======================================================================================================================================================
# NEW EXTRACT HISTORY
# ======================================================================================================================================================
import numpy as np
import pandas as pd
from torch import Tensor

def extract_history_new(variables_history):
    # If variables_history is a dictionary, handle it as before
    if isinstance(variables_history, dict):
        # Extract 'u' history as a NumPy array
        u_history = np.array(variables_history['u'])
        
        # Determine number of agents from the second dimension of 'u'
        num_agents = u_history.shape[1]
        
        # Generate timestep numbers
        timesteps = np.arange(1, u_history.shape[0] + 1)
        
        # Create a dictionary for the DataFrame
        data = {'Timestep': timesteps}
        
        # Add each agent's actions to the data dictionary
        for agent_id in range(num_agents):
            data[f'agent_{agent_id}_action'] = u_history[:, agent_id]
        
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    # If variables_history is a list, handle the new list-of-tuples format
    elif isinstance(variables_history, list):
        all_data = []  # List to collect DataFrames from each repeat
        
        for repeat_idx, repeat in enumerate(variables_history):
            # Each repeat is a tuple: (int, dict)
            if isinstance(repeat, tuple) and len(repeat) >= 2:
                repeat_id, metrics_dict = repeat[0], repeat[1]
            else:
                raise ValueError("Each item in the list should be a tuple with at least two elements.")
            
            # Extract 'u' history; ensure it's a NumPy array
            u_history = np.array(metrics_dict['u'])
            
            # Determine number of agents
            num_agents = u_history.shape[1]
            
            # Generate timestep numbers
            timesteps = np.arange(1, u_history.shape[0] + 1)
            
            # Create a dictionary for the DataFrame
            data = {'Timestep': timesteps}
            
            # Add each agent's actions to the data dictionary
            for agent_id in range(num_agents):
                data[f'agent_{agent_id}_action'] = u_history[:, agent_id]
            
            # Add a 'Repeat' column to track the source repeat
            data['Repeat'] = repeat_id  # You can also use 'repeat_idx + 1' if preferred
            
            # Convert the dictionary into a DataFrame and append to the list
            df_repeat = pd.DataFrame(data)
            all_data.append(df_repeat)
        
        # Concatenate all DataFrames from the repeats into a single DataFrame
        final_df = pd.concat(all_data, ignore_index=True)
        
        return final_df
    
    else:
        raise ValueError("variables_history should be either a dictionary or a list of tuples containing dictionaries.")



# ======================================================================================================================================================
# NEWEST EXTRACT HISTORY
# ======================================================================================================================================================

import numpy as np
import pandas as pd
from torch import Tensor

def extract_history_newest(variables_history, variable):
    """
    Extract a specified metric from 'variables_history' and convert it to a DataFrame.
    Handles both original dictionary format and the new list-of-tuples format.
    If the metric contains tensors, extracts the first element of each tensor and maps them to agent action columns.

    Args:
        variables_history (list or dict): 
            - Original format: A dictionary containing metrics.
            - New format: A list of tuples, each containing an integer and a dictionary with metrics.
        variable (str): The key of the metric to extract (e.g., 'u', 'gamma', etc.).

    Returns:
        pd.DataFrame: 
            A DataFrame with 'Timestep', agent action columns (e.g., 'agent_0_action', 'agent_1_action', ...),
            and 'Repeat' to indicate the source repeat (if 'variables_history' is a list).
    """
    # Helper function to process metric data
    def process_metric_data(metric_data):
        """
        Processes the metric data by extracting the first element if it's a Tensor.
        Converts the processed data into a NumPy array.

        Args:
            metric_data (list, Tensor, or np.ndarray): The raw metric data.

        Returns:
            np.ndarray: The processed metric data as a NumPy array.
        """
        if isinstance(metric_data, list):
            # Check if it's a list of lists containing Tensors
            if len(metric_data) > 0 and isinstance(metric_data[0], list) and isinstance(metric_data[0][0], Tensor):
                # Extract the first element from each Tensor
                try:
                    processed_data = [
                        [tensor[0].item() if tensor.numel() > 0 else np.nan for tensor in inner_list]
                        for inner_list in metric_data
                    ]
                except AttributeError:
                    raise TypeError("Encountered an object that is not a Tensor within the nested lists.")
                return np.array(processed_data)
            else:
                # Assume it's a list of lists of integers or floats
                return np.array(metric_data)
        
        elif isinstance(metric_data, Tensor):
            # If the tensor has 3 dimensions, extract the first element of the last dimension
            if metric_data.ndim == 3:
                if metric_data.shape[2] < 1:
                    raise ValueError("The tensor does not have enough elements in the last dimension.")
                return metric_data[:, :, 0].numpy()
            elif metric_data.ndim == 2:
                # For 2D tensors, extract the first element of each tensor if possible
                return metric_data.numpy()
            else:
                raise ValueError(f"Unsupported Tensor shape: {metric_data.shape}")
        
        elif isinstance(metric_data, np.ndarray):
            if metric_data.ndim == 3 and metric_data.shape[2] == 2:
                # Extract the first element of the last dimension
                return metric_data[:, :, 0]
            else:
                return metric_data
        else:
            raise TypeError(f"Unsupported type for metric data: {type(metric_data)}")

    # Helper function to map extracted actions to agent columns
    def map_to_agent_columns(metric_array):
        """
        Maps the extracted metric array to agent action columns.

        Args:
            metric_array (np.ndarray): Array of shape (timesteps, agents) or (timesteps,).

        Returns:
            dict: Dictionary mapping column names to their data.
        """
        data = {'Timestep': np.arange(1, metric_array.shape[0] + 1)}
        
        if metric_array.ndim == 2:
            num_agents = metric_array.shape[1]
            for agent_id in range(num_agents):
                data[f'agent_{agent_id}_action'] = metric_array[:, agent_id]
        elif metric_array.ndim == 1:
            # Single agent or scalar metric
            data['agent_0_action'] = metric_array
        else:
            # Unsupported dimensionality for agent actions
            raise ValueError(f"Unsupported metric array shape: {metric_array.shape}")
        
        return data

    # If variables_history is a dictionary, handle it as before
    if isinstance(variables_history, dict):
        # Check if the specified variable exists
        if variable not in variables_history:
            raise KeyError(f"Variable '{variable}' not found in the dictionary.")

        # Extract and process the specified metric
        metric_data = variables_history[variable]
        metric_array = process_metric_data(metric_data)

        # Map to agent columns
        data = map_to_agent_columns(metric_array)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    # If variables_history is a list, handle the new list-of-tuples format
    elif isinstance(variables_history, list):
        all_data = []  # List to collect DataFrames from each repeat

        for repeat_idx, repeat in enumerate(variables_history):
            # Each repeat is a tuple: (int, dict)
            if isinstance(repeat, tuple) and len(repeat) >= 2:
                repeat_id, metrics_dict = repeat[0], repeat[1]
            else:
                raise ValueError(f"Each item in the list should be a tuple with at least two elements. Problem at index {repeat_idx}.")

            # Check if the specified variable exists in the current metrics_dict
            if variable not in metrics_dict:
                raise KeyError(f"Variable '{variable}' not found in the dictionary at repeat index {repeat_idx}.")

            # Extract and process the specified metric
            metric_data = metrics_dict[variable]
            metric_array = process_metric_data(metric_data)

            # Map to agent columns
            data = map_to_agent_columns(metric_array)

            # Add a 'Repeat' column to track the source repeat
            data['Repeat'] = repeat_id  # Or use 'repeat_idx + 1' if preferred

            # Convert the dictionary into a DataFrame and append to the list
            df_repeat = pd.DataFrame(data)
            all_data.append(df_repeat)

        # Concatenate all DataFrames from the repeats into a single DataFrame
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    else:
        raise ValueError("variables_history should be either a dictionary or a list of tuples containing dictionaries.")

# ======================================================================================================================================================
# ENTROPY
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import startJVM, getDefaultJVMPath, isJVMStarted, JPackage
from matplotlib.ticker import MaxNLocator
import os


def calculate_base(actions):
    """
    Calculates the base for entropy as the number of unique distinct actions.

    Parameters:
    - actions: pandas Series containing action data.

    Returns:
    - Integer representing the base.
    """
    unique_actions = np.unique(actions.dropna().astype(int))  # Ensure no NaN values
    base = len(unique_actions)
    return base


def entropy_discrete(data, variable, base):
    """
    Calculate discrete entropy for a single variable.

    Parameters:
    - data: pandas DataFrame containing the data.
    - variable: column name for which to calculate entropy.
    - base: base for the discrete variables.

    Returns:
    - DataFrame with Entropy (Bits) per timestep.
    """
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[variable])
    train = data_clean[variable].values.astype(int)

    # Initialize the JVM if it's not already running
    initialize_jvm()

    # Set up the entropy calculator
    calcClass = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    calc = calcClass(base)

    calc.initialise()
    calc.addObservations(train)

    # Calculate entropy results
    local_result_nats = calc.computeLocal(train)

    # Convert nats to bits
    local_result_bits = np.array(local_result_nats) / np.log(2)

    # Return DataFrame
    return pd.DataFrame({'Entropy_Bits': local_result_bits})


def calculate_entropy_discrete(data, action_suffix='_action'):
    """
    Calculate entropy for each agent's actions. Handles both single runs and multiple repeats.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: suffix used to identify action columns.

    Returns:
    - If 'Repeat' is not in data:
        - DataFrame with columns ['Entropy_Bits', 'Agent', 'Timestep']
      If 'Repeat' is in data:
        - DataFrame with columns ['Agent', 'Timestep', 'Mean_Entropy', 'Std_Error']
    """
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate entropy for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            for agent_col in action_columns:
                # Extract agent number (assuming format like 'agent_1_action')
                agent_number = agent_col.split('_')[1]

                # Calculate the base for this agent's actions
                base = calculate_base(group[agent_col])

                # Calculate local entropy
                local_entropy_df = entropy_discrete(group, agent_col, base)

                # Append agent number and timestep values
                local_entropy_df['Agent'] = agent_number
                local_entropy_df['Timestep'] = group['Timestep'].values[:len(local_entropy_df)]
                local_entropy_df['Repeat'] = repeat
                result_df_list.append(local_entropy_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Calculate mean and standard error across repeats
        aggregated_df = combined_df.groupby(['Agent', 'Timestep']).agg(
            Mean_Entropy=('Entropy_Bits', 'mean'),
            Std_Error=('Entropy_Bits', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate entropy normally
        for agent_col in action_columns:
            # Extract agent number
            agent_number = agent_col.split('_')[1]

            # Calculate the base for this agent's actions
            base = calculate_base(data[agent_col])

            # Calculate local entropy
            local_entropy_df = entropy_discrete(data, agent_col, base)

            # Append agent number and timestep values
            local_entropy_df['Agent'] = agent_number
            local_entropy_df['Timestep'] = data['Timestep'].values[:len(local_entropy_df)]
            result_df_list.append(local_entropy_df)

        # Combine all results into a single DataFrame
        return pd.concat(result_df_list, ignore_index=True)


def plot_entropy_heatmap(entropy_measures_df, output_path=None):
    """
    Plot a heatmap or line graphs for the entropy values based on the DataFrame structure.

    Parameters:
    - entropy_measures_df: pandas DataFrame containing entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    if {'Mean_Entropy', 'Std_Error'}.issubset(entropy_measures_df.columns):
        # Data has 'Repeat' column: Plot average entropy with standard error as separate subplots
        plot_entropy_linegraphs_subplots(entropy_measures_df, output_path)
    else:
        # Single run: Plot heatmap
        plot_entropy_heatmap_single(entropy_measures_df, output_path)


def plot_entropy_heatmap_single(entropy_measures_df, output_path=None):
    """
    Plot a heatmap for the entropy values (single run).

    Parameters:
    - entropy_measures_df: pandas DataFrame containing entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    # Create a pivot table for heatmap plotting
    pivot_table = entropy_measures_df.pivot_table(
        index='Agent',
        columns='Timestep',
        values='Entropy_Bits',
        aggfunc='mean'
    )

    # Set up colormap and gridspec for plotting
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # Ratio of heatmap and colorbar

    # Heatmap
    ax = plt.subplot(gs[0])
    sns.heatmap(pivot_table, cmap=cmap, cbar=False, ax=ax, linewidths=0)

    # Set axis labels and title using the updated font settings
    ax.set_xlabel('Time', labelpad=10)
    ax.set_ylabel('Agent', labelpad=10)
    ax.set_title('Marginal Entropy', pad=15)

    # Ticks and labels
    ax.set_yticks(np.arange(len(pivot_table.index)) + 0.5)
    ax.set_yticklabels(pivot_table.index)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()])

    # Colorbar
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Entropy (Bits)', size=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_entropy_linegraphs_subplots(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=2
):
    """
    Plot separate line graphs for each agent showing average entropy with standard error shading,
    with optional Savitzky-Golay smoothing for smoother lines and shading.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Agent', 'Timestep', 'Mean_Entropy', 'Std_Error']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    agents = aggregated_df['Agent'].unique()
    num_agents = len(agents)

    # Define figure size: width proportional to number of agents, height fixed
    width_per_agent = 6
    height = 6
    total_width = width_per_agent * num_agents
    sns.set(style="whitegrid")

    # Create a subplot for each agent
    fig, axes = plt.subplots(
        1, num_agents, 
        figsize=(width_per_agent * num_agents, height), 
        sharey=True
    )

    # If only one agent, axes is not a list
    if num_agents == 1:
        axes = [axes]

    # Define a color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent].sort_values('Timestep')

        # Original timesteps and data
        x = agent_data['Timestep'].values
        y = agent_data['Mean_Entropy'].values
        y_err = agent_data['Std_Error'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if savgol_window % 2 == 0 or savgol_window <= 0:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent {agent}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean entropy with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent {agent}',
            color=color_dict[agent],
            linewidth=1  # Thinner lines
        )

        # Shade the standard error with adjusted transparency
        ax.fill_between(
            x,
            y_smooth - y_err_smooth,
            y_smooth + y_err_smooth,
            color=color_dict[agent],
            alpha=0.2  # Increased alpha for better visibility
        )

        ax.set_title(f'Agent {agent}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('Entropy (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Create a single legend for all subplots
    handles = [plt.Line2D([0], [0], color=color_dict[agent], lw=2) for agent in agents]
    labels = [f'Agent {agent}' for agent in agents]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agents', title_fontsize=18)

    plt.suptitle('Average Marginal Entropy with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def entropy(data, action_suffix='_action'):
    """
    Main function to calculate and plot entropy heatmap or line graphs based on the presence of 'Repeat' column.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: suffix used to identify action columns.
    """

    # Plot setting
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'cmr10'  # Use the Computer Modern Roman font
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math text
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.labelsize'] = 16  # Axis labels
    plt.rcParams['axes.titlesize'] = 20  # Title
    plt.rcParams['xtick.labelsize'] = 14 # X tick labels
    plt.rcParams['ytick.labelsize'] = 14 # Y tick labels

    # Calculate entropy
    entropy_measures_df = calculate_entropy_discrete(data, action_suffix)

    # Plot heatmap or line graphs based on data structure
    plot_entropy_heatmap(entropy_measures_df)


# ======================================================================================================================================================
# JOINT ENTROPY 
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from jpype import JPackage


def create_joint_alphabet(data, action_columns):
    """
    Combines actions of all agents at each timestep into a unique identifier for joint entropy.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_columns: List of column names representing agents' actions.

    Returns:
    - NumPy array of mapped joint alphabet values.
    """
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=action_columns)
    combined_actions = data_clean[action_columns].values.astype(int)

    # Create a unique identifier for each combination of actions
    joint_alphabet = np.apply_along_axis(lambda x: int(''.join(map(str, x))), 1, combined_actions)

    # Map the unique identifiers to a range [0, n_unique_values - 1]
    _, joint_alphabet_mapped = np.unique(joint_alphabet, return_inverse=True)

    return joint_alphabet_mapped


def calculate_base(joint_alphabet_mapped):
    """
    Calculate the base for entropy calculation as the number of unique states in the joint alphabet.

    Parameters:
    - joint_alphabet_mapped: NumPy array of mapped joint alphabet values.

    Returns:
    - Integer representing the base.
    """
    return len(np.unique(joint_alphabet_mapped))


def entropy_discrete(data, variable, base):
    """
    Calculates the discrete entropy for a given variable with the specified base.

    Parameters:
    - data: pandas DataFrame containing the data.
    - variable: Column name for which to calculate entropy.
    - base: Base for the discrete variables.

    Returns:
    - DataFrame with Entropy (Bits) per timestep.
    """
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[variable])
    train = data_clean[variable].values.astype(int)  # Ensure discrete data by casting to integers

    # Set up the entropy calculator (JVM already initialized elsewhere)
    calcClass = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    calc = calcClass(base)

    calc.initialise()  # Initialize the calculator
    calc.addObservations(train)  # Add discrete observations

    # Compute entropy
    local_result_nats = calc.computeLocal(train)

    # Convert to bits
    local_result_bits = np.array(local_result_nats) / np.log(2)

    # Create result DataFrame
    result_df = pd.DataFrame({
        'Entropy_Bits': local_result_bits
    })

    return result_df


def calculate_joint_entropy(data, action_suffix='_action'):
    """
    Calculates joint entropy over all agents' actions and returns a DataFrame with entropy values per timestep.
    Handles both single runs and multiple repeats.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: Suffix used to identify action columns.

    Returns:
    - If 'Repeat' is not in data:
        - DataFrame with columns ['Entropy_Bits', 'Timestep']
      If 'Repeat' is in data:
        - DataFrame with columns ['Timestep', 'Mean_Entropy', 'Std_Error']
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate joint entropy for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Create joint alphabet
            joint_alphabet_mapped = create_joint_alphabet(group, action_columns)

            # Calculate base
            base = calculate_base(joint_alphabet_mapped)

            # Create a temporary DataFrame to pass to entropy_discrete
            temp_df = pd.DataFrame({'Joint_Alphabet': joint_alphabet_mapped})

            # Calculate entropy over the joint alphabet (this is now a single variable)
            joint_entropy_df = entropy_discrete(temp_df, 'Joint_Alphabet', base)

            # Add timestep information to the result
            joint_entropy_df['Timestep'] = group['Timestep'].values[:len(joint_entropy_df)]
            joint_entropy_df['Repeat'] = repeat
            result_df_list.append(joint_entropy_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Calculate mean and standard error across repeats
        aggregated_df = combined_df.groupby('Timestep').agg(
            Mean_Entropy=('Entropy_Bits', 'mean'),
            Std_Error=('Entropy_Bits', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate joint entropy normally
        # Create joint alphabet
        joint_alphabet_mapped = create_joint_alphabet(data, action_columns)

        # Calculate base
        base = calculate_base(joint_alphabet_mapped)

        # Create a temporary DataFrame to pass to entropy_discrete
        temp_df = pd.DataFrame({'Joint_Alphabet': joint_alphabet_mapped})

        # Calculate entropy over the joint alphabet (this is now a single variable)
        joint_entropy_df = entropy_discrete(temp_df, 'Joint_Alphabet', base)

        # Add timestep information to the result
        joint_entropy_df['Timestep'] = data['Timestep'].values[:len(joint_entropy_df)]

        return joint_entropy_df


def plot_joint_entropy(joint_entropy_df, output_path=None):
    """
    Plots a heatmap or line graph for the joint entropy values based on the DataFrame structure.

    Parameters:
    - joint_entropy_df: pandas DataFrame containing joint entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    if {'Mean_Entropy', 'Std_Error'}.issubset(joint_entropy_df.columns):
        # Data has multiple repeats: Plot average joint entropy with standard error as a line graph
        plot_joint_entropy_linegraph(joint_entropy_df, output_path)
    else:
        # Single run: Plot heatmap
        plot_joint_entropy_heatmap_single(joint_entropy_df, output_path)


def plot_joint_entropy_heatmap_single(joint_entropy_df, output_path=None):
    """
    Plots a heatmap of joint entropy values over timesteps (single run).

    Parameters:
    - joint_entropy_df: pandas DataFrame containing joint entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    # Create a pivot table for heatmap plotting
    pivot_table = joint_entropy_df.pivot_table(
        index='Timestep',
        values='Entropy_Bits',
        aggfunc='mean'
    ).T  # Transpose to have Timesteps on x-axis

    # Define a custom colormap
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Create a gridspec layout to control the colorbar and heatmap separately
    fig = plt.figure(figsize=(10, 8), dpi=300)  # Wide and not very tall
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # 20:1 ratio between heatmap and colorbar

    # Create the heatmap in the first grid cell
    ax = plt.subplot(gs[0])
    sns.heatmap(pivot_table, cmap=cmap, cbar=False, ax=ax, linewidths=0, cbar_kws={"shrink": 0.5})

    # Set axis labels and title with the appropriate sizes
    ax.set_xlabel('Time', fontsize=12, labelpad=5)
    ax.set_ylabel('Joint Entropy', fontsize=12, labelpad=5)  # Keep axis label as 'Joint Entropy'
    ax.set_title('Joint Entropy Over Time', fontsize=14, pad=10)

    # Set the y-tick label to 'Joint Entropy' without changing the axis label
    ax.set_yticks([0.5])  # Only one y-tick for "Joint Entropy"
    ax.set_yticklabels(['Joint Entropy'], fontsize=10)

    # Ensure that all labels are shown on x-axis, and ticks are centered in the middle of each cell
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=10)

    # Create the colorbar in the second grid cell
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Entropy (Bits)', size=12)

    # Adjust layout for a clean look
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in high-resolution formats (PDF and PNG)
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_joint_entropy_linegraph(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plots a line graph of average joint entropy with standard error shading, 
    with optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Timestep', 'Mean_Entropy', 'Std_Error']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Timestep', 'Mean_Entropy', 'Std_Error'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by 'Timestep'
    aggregated_df = aggregated_df.sort_values('Timestep')

    # Extract data
    x = aggregated_df['Timestep'].values
    y = aggregated_df['Mean_Entropy'].values
    y_err = aggregated_df['Std_Error'].values

    # Apply Savitzky-Golay filter if enabled
    if use_savgol:
        # Validate Savitzky-Golay parameters
        if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
            warnings.warn(
                f"Savitzky-Golay window size must be a positive odd integer. "
                f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_window > len(x):
            warnings.warn(
                f"Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_polyorder >= savgol_window:
            warnings.warn(
                f"Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        else:
            try:
                # Apply Savitzky-Golay filter
                y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
            except Exception as e:
                warnings.warn(
                    f"Savitzky-Golay filtering failed with error: {e}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
    else:
        # No smoothing; use original data
        y_smooth = y
        y_err_smooth = y_err

    # Initialize the plot
    plt.figure(figsize=(5, 3), dpi=300)
    sns.set(style="whitegrid")

    # Define color palette
    color = sns.color_palette("viridis", 1)[0]

    # Plot mean entropy line
    plt.plot(
        x,
        y_smooth,
        label='Mean Joint Entropy',
        color=color,
        linewidth=1  # Thinner line
    )

    # Plot standard error shading
    plt.fill_between(
        x,
        y_smooth - y_err_smooth,
        y_smooth + y_err_smooth,
        color=color,
        alpha=0.2,  # Increased alpha for better visibility
        label='Standard Error'
    )

    # Set labels and title
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Joint Entropy (Bits)', fontsize=7)
    plt.title('Average Joint Entropy Over Time with Standard Error', fontsize=8)

    # Customize ticks
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Add legend
    plt.legend(fontsize=6)

    # Adjust layout for a clean look
    plt.tight_layout()

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def joint_entropy(data, action_suffix='_action'):
    """
    Main function to calculate and plot joint entropy heatmap or line graph based on the presence of 'Repeat' column.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: Suffix used to identify action columns.
    """
    # Calculate joint entropy
    joint_entropy_df = calculate_joint_entropy(data, action_suffix)

    # Plot heatmap or line graph based on data structure
    plot_joint_entropy(joint_entropy_df)


# ======================================================================================================================================================
# CONDITIONAL MUTUAL INFORMATION 
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import JPackage, JArray, JInt
from matplotlib.ticker import MaxNLocator

def initialize_cmi_calculator():
    """
    Initialize the Conditional Mutual Information calculator from JIDT.
    """
    calcClass = JPackage("infodynamics.measures.discrete").ConditionalMutualInformationCalculatorDiscrete
    return calcClass(2, 2, 2)  # Base-2 for binary actions

def calculate_local_cmi_for_pair(data, agent_col_i, agent_col_j, other_vars, calc):
    """
    Calculate the local Conditional Mutual Information (CMI) between two agents, conditioned on all others.
    """
    # Convert the data into JArray (required by JIDT)
    var_i = JArray(JInt)(data[agent_col_i].values.astype(int).tolist())
    var_j = JArray(JInt)(data[agent_col_j].values.astype(int).tolist())
    
    # Flatten and convert other variables into JArray
    if other_vars.size > 0:
        cond = JArray(JInt)(other_vars.flatten().astype(int).tolist())
    else:
        # If no conditions are present, pass an empty array to the condition
        cond = JArray(JInt)([])

    # Initialize the calculator again for each pair to avoid possible carryover issues
    calc.initialise()  
    
    # Add observations and compute local CMI
    calc.addObservations(var_i, var_j, cond)
    local_cmi = calc.computeLocal(var_i, var_j, cond)

    return local_cmi

def calculate_local_conditional_mutual_information(data, action_suffix):
    """
    Calculate local Conditional Mutual Information (CMI) for all pairs of agents.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    num_agents = len(action_columns)

    # Initialize the JIDT CMI calculator
    calc = initialize_cmi_calculator()

    # Prepare result storage for local CMI
    local_cmi_dict = {}

    # Loop over each pair of agents and calculate local CMI
    for i, agent_col_i in enumerate(action_columns):
        for j, agent_col_j in enumerate(action_columns):
            if i != j:
                # Prepare other variables as the condition
                other_vars = np.vstack([data[col].values.astype(int) for k, col in enumerate(action_columns) if k != i and k != j]).T

                # Calculate local CMI between agent i and agent j
                local_cmi = calculate_local_cmi_for_pair(data, agent_col_i, agent_col_j, other_vars, calc)

                # Store the result
                pair_key = f"{i} ; {j}"
                local_cmi_dict[pair_key] = local_cmi

    # Convert the result dictionary to a DataFrame
    local_cmi_df = pd.DataFrame(local_cmi_dict)
    # local_cmi_df['Timestep'] = data.index

    # Preserve the 'Timestep' column from the original data
    local_cmi_df['Timestep'] = data['Timestep'].values

    return local_cmi_df

def calculate_cmi_with_repeats(data, action_suffix):
    """
    Calculates CMI for agent pairs and averages over repeats.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate local CMI for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Reset the Timestep to start from 1 for each repeat
            group = group.copy()  # Avoid modifying the original data
            group['Timestep'] = np.arange(1, len(group) + 1)

            # Calculate CMI for this repeat
            local_cmi_df = calculate_local_conditional_mutual_information(group, action_suffix)
            local_cmi_df['Repeat'] = repeat
            result_df_list.append(local_cmi_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Pair', value_name='Local CMI')

        # Ensure 'Local CMI' column is numeric after melting
        melted_df['Local CMI'] = pd.to_numeric(melted_df['Local CMI'], errors='coerce')

        # Drop any rows where 'Local CMI' is NaN
        melted_df = melted_df.dropna(subset=['Local CMI'])


        # Now we group by 'Pair' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Pair', 'Timestep']).agg(
            mean_cmi=('Local CMI', 'mean'),
            sem_cmi=('Local CMI', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate CMI normally
        return calculate_local_conditional_mutual_information(data, action_suffix)

def plot_local_conditional_mutual_information_heatmap(local_cmi_df):
    """
    Plot a heatmap of the local Conditional Mutual Information (CMI) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_cmi_melted = local_cmi_df.melt(id_vars='Timestep', var_name='Pair', value_name='Local CMI')

    # Create a pivot table for heatmap input
    heatmap_data = local_cmi_melted.pivot_table(index='Pair', columns='Timestep', values='Local CMI')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title with updated fontsize and padding
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent Pairs', fontsize=20, labelpad=10)
    ax.set_title('Conditional Mutual Information', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local CMI (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_cmi_linegraphs(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot CMI line graphs for all agent pairs with standard error shading, 
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Pair', 'Timestep', 'mean_cmi', 'sem_cmi']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Pair', 'Timestep', 'mean_cmi', 'sem_cmi'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agent pairs
    agent_pairs = aggregated_df['Pair'].unique()
    num_pairs = len(agent_pairs)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_pairs // max_per_row) + (1 if num_pairs % max_per_row != 0 else 0)

    # Set figure size based on number of pairs and rows
    width_per_pair = 6
    height_per_row = 6
    total_width = min(num_pairs, max_per_row) * width_per_pair
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_pairs), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair].sort_values('Timestep')

        # Extract data
        x = pair_data['Timestep'].values
        y = pair_data['mean_cmi'].values
        y_err = pair_data['sem_cmi'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent Pair {pair}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean CMI with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent Pair {pair}',
            color=color_dict[pair],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_cmi' in pair_data.columns:
            if pair_data['sem_cmi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[pair],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CMI (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agent_pairs):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[pair], lw=2) for pair in agent_pairs]
    labels = [f'Agent Pair {pair}' for pair in agent_pairs]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agent Pairs', title_fontsize=18)

    plt.suptitle('Conditional Mutual Information with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()


def conditional_mutual_information(data, action_suffix='_action'):
    """
    Main function to calculate local CMI, plot the line graphs (for repeats), and display the results.
    """
    # Calculate CMI, either with or without repeats
    aggregated_df = calculate_cmi_with_repeats(data, action_suffix)
    

    # Check if mean and sem_cmi exist for plotting
    if 'mean_cmi' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_cmi_linegraphs(aggregated_df)
    else:
        # Otherwise, use the original heatmap plot
        plot_local_conditional_mutual_information_heatmap(aggregated_df)
   

# ======================================================================================================================================================
# CONDITIONAL TRANSFER ENTROPY
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import JPackage, JArray, JInt
from matplotlib.ticker import MaxNLocator

def initialize_cte_calculator(num_condition_vars):
    """
    Initialize the Conditional Transfer Entropy calculator from JIDT.
    """
    calcClass = JPackage("infodynamics.measures.discrete").ConditionalTransferEntropyCalculatorDiscrete
    return calcClass(2, 1, num_condition_vars)  # Base-2 for source, destination, and conditionals

def calculate_local_cte_for_pair(states, i, j, other_vars_indices, calc):
    """
    Calculate local Conditional Transfer Entropy (CTE) for a specific pair of agents.
    """
    othersAbsolute = JArray(JInt)(other_vars_indices)

    # Calculate the local conditional transfer entropy TE(source -> dest | othersAbsolute)
    local_cte = calc.computeLocal(states, i, j, othersAbsolute)

    return local_cte

def calculate_local_conditional_transfer_entropy(data, action_suffix):
    """
    Calculate local Conditional Transfer Entropy (CTE) for all pairs of agents.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    num_agents = len(action_columns)

    # Prepare result storage for local CTE
    local_cte_dict = {}

    # Convert the dataframe to a 2D numpy array (time steps as rows, variables as columns)
    states = data[action_columns].values.astype(int)

    # Loop over each pair of agents and calculate local CTE
    for i, agent_col_i in enumerate(action_columns):
        for j, agent_col_j in enumerate(action_columns):
            if i != j:
                # Condition on all other variables
                other_vars_indices = [k for k in range(len(action_columns)) if k != i and k != j]

                # Initialize the CTE calculator
                calc = initialize_cte_calculator(len(other_vars_indices))
                calc.initialise()

                # Calculate local CTE between agent i and agent j
                local_cte = calculate_local_cte_for_pair(states, i, j, other_vars_indices, calc)

                # Store the result in the dictionary
                pair_key = f"{i} → {j}"
                local_cte_dict[pair_key] = local_cte

    # Convert the result dictionary to a DataFrame
    local_cte_df = pd.DataFrame(local_cte_dict)

    # Preserve the 'Timestep' column from the original data
    local_cte_df['Timestep'] = data['Timestep'].values

    return local_cte_df

def calculate_cte_with_repeats(data, action_suffix):
    """
    Calculates CTE for agent pairs and averages over repeats.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate local CTE for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Calculate CTE for this repeat
            local_cte_df = calculate_local_conditional_transfer_entropy(group, action_suffix)
            local_cte_df['Repeat'] = repeat
            result_df_list.append(local_cte_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Pair', value_name='Local CTE')

        # Ensure 'Local CTE' column is numeric after melting
        melted_df['Local CTE'] = pd.to_numeric(melted_df['Local CTE'], errors='coerce')

        # Drop any rows where 'Local CTE' is NaN
        melted_df = melted_df.dropna(subset=['Local CTE'])

        # Now we group by 'Pair' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Pair', 'Timestep']).agg(
            mean_cte=('Local CTE', 'mean'),
            sem_cte=('Local CTE', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate CTE normally
        return calculate_local_conditional_transfer_entropy(data, action_suffix)
    
def plot_local_conditional_transfer_entropy_heatmap(local_cte_df):
    """
    Plot a heatmap of the local Conditional Transfer Entropy (CTE) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_cte_melted = local_cte_df.melt(id_vars='Timestep', var_name='Pair', value_name='Local CTE')

    # Create a pivot table for heatmap input
    heatmap_data = local_cte_melted.pivot_table(index='Pair', columns='Timestep', values='Local CTE')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title with updated fontsize and padding
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent Pairs', fontsize=20, labelpad=10)
    ax.set_title('Conditional Transfer Entropy', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local CTE (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_cte_linegraphs(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot CTE line graphs for all agent pairs with standard error shading,
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Pair', 'Timestep', 'mean_cte', 'sem_cte']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Pair', 'Timestep', 'mean_cte', 'sem_cte'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agent pairs
    agent_pairs = aggregated_df['Pair'].unique()
    num_pairs = len(agent_pairs)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_pairs // max_per_row) + (1 if num_pairs % max_per_row != 0 else 0)

    # Set figure size based on number of pairs and rows
    width_per_pair = 6
    height_per_row = 6
    total_width = min(num_pairs, max_per_row) * width_per_pair
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_pairs), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair].sort_values('Timestep')

        # Extract data
        x = pair_data['Timestep'].values
        y = pair_data['mean_cte'].values
        y_err = pair_data['sem_cte'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent Pair {pair}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean CTE with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent Pair {pair}',
            color=color_dict[pair],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_cte' in pair_data.columns:
            if pair_data['sem_cte'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[pair],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CTE (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agent_pairs):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[pair], lw=2) for pair in agent_pairs]
    labels = [f'Agent Pair {pair}' for pair in agent_pairs]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agent Pairs', title_fontsize=18)

    plt.suptitle('Conditional Transfer Entropy with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def conditional_transfer_entropy(data, action_suffix='_action'):
    """
    Main function to calculate local CTE, plot the heatmap, or line graphs if repeats are present.

    Parameters:
    - data: pandas DataFrame containing relevant data for CTE calculation.
    - action_suffix: Suffix to identify action-related columns in the DataFrame.
    """
    # Calculate CTE, either with or without repeats
    aggregated_df = calculate_cte_with_repeats(data, action_suffix)

    # Check if mean and sem_cte exist for plotting
    if 'mean_cte' in aggregated_df.columns and 'sem_cte' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_cte_linegraphs(aggregated_df)
    else:
        # Otherwise, use the original heatmap plot
        plot_local_conditional_transfer_entropy_heatmap(aggregated_df)




# ======================================================================================================================================================
# PREDICTIVE INFORMATION
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import JPackage
from matplotlib.ticker import MaxNLocator

def initialize_pi_calculator(base, past_window):
    """
    Initialize the Predictive Information calculator from JIDT.
    """
    calcClass = JPackage("infodynamics.measures.discrete").PredictiveInformationCalculatorDiscrete
    return calcClass(base, past_window)

def calculate_local_pi_for_agent(actions, calc):
    """
    Calculate local Predictive Information (PI) for a specific agent.
    """
    # Add observations
    calc.addObservations(actions)
    
    # Calculate the local predictive information
    local_pi = calc.computeLocal(actions)
    
    return local_pi

def calculate_local_predictive_information(data, action_suffix, past_window=1, future_window=1):
    """
    Calculate local Predictive Information (PI) for all agents.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]

    # Prepare result storage for local PI
    local_pi_dict = {}

    # Loop over each agent and calculate local PI
    for agent_col in action_columns:
        agent_number = agent_col.split('_')[1]  # Extract agent number

        # Calculate the base for this agent's actions
        base = len(np.unique(data[agent_col].dropna().values))

        # Initialize the PI calculator
        calc = initialize_pi_calculator(base, past_window)
        calc.initialise()

        # Calculate local PI for this agent
        local_pi = calculate_local_pi_for_agent(data[agent_col].values.astype(int), calc)

        # Store the result
        local_pi_dict[f'Agent {agent_number}'] = local_pi

    # Convert the result dictionary to a DataFrame
    local_pi_df = pd.DataFrame(local_pi_dict)
    local_pi_df['Timestep'] = data['Timestep'].values

    return local_pi_df

def calculate_pi_with_repeats(data, action_suffix, past_window=1, future_window=1):
    """
    Calculate Predictive Information (PI) for agent actions, averaging over repeats if present.
    """
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate PI for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Calculate PI for this repeat
            local_pi_df = calculate_local_predictive_information(group, action_suffix, past_window, future_window)
            local_pi_df['Repeat'] = repeat
            result_df_list.append(local_pi_df)

            # Print the number of timesteps in each repeat
            num_timesteps = len(group)
    

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Agent', value_name='Local PI')

        # Ensure 'Local PI' column is numeric after melting
        melted_df['Local PI'] = pd.to_numeric(melted_df['Local PI'], errors='coerce')

        # Drop any rows where 'Local PI' is NaN
        melted_df = melted_df.dropna(subset=['Local PI'])



        # Group by 'Agent' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Agent', 'Timestep']).agg(
            mean_pi=('Local PI', 'mean'),
            sem_pi=('Local PI', 'sem')
        ).reset_index()

 

        return aggregated_df
    else:
        # Single run: Calculate PI normally
        return calculate_local_predictive_information(data, action_suffix, past_window, future_window)

def plot_pi_heatmap(local_pi_df):
    """
    Plot a heatmap of the local Predictive Information (PI) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_pi_melted = local_pi_df.melt(id_vars='Timestep', var_name='Agent', value_name='Local PI')

    # Create a pivot table for heatmap input
    heatmap_data = local_pi_melted.pivot_table(index='Agent', columns='Timestep', values='Local PI')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent', fontsize=20, labelpad=10)
    ax.set_title('Predictive Information', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local PI (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_pi_linegraphs(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot Predictive Information (PI) line graphs for all agents with standard error shading,
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Agent', 'Timestep', 'mean_pi', 'sem_pi']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Agent', 'Timestep', 'mean_pi', 'sem_pi'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agents
    agents = aggregated_df['Agent'].unique()
    num_agents = len(agents)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_agents // max_per_row) + (1 if num_agents % max_per_row != 0 else 0)

    # Set figure size based on number of agents and rows
    width_per_agent = 6
    height_per_row = 6
    total_width = min(num_agents, max_per_row) * width_per_agent
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_agents), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_agents > 1 else [axes]  # Single row with multiple agents or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent].sort_values('Timestep')

        # Extract data
        x = agent_data['Timestep'].values
        y = agent_data['mean_pi'].values
        y_err = agent_data['sem_pi'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent {agent}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean PI with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent {agent}',
            color=color_dict[agent],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_pi' in agent_data.columns:
            if agent_data['sem_pi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[agent],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent {agent}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('Predictive Information (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agents):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[agent], lw=2) for agent in agents]
    labels = [f'Agent {agent}' for agent in agents]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agents', title_fontsize=18)

    plt.suptitle('Predictive Information with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def predictive_information(data, action_suffix='_action', past_window=1, future_window=1):
    """
    Main function to calculate local Predictive Information, plot the heatmap, or line graphs if repeats are present.
    """
    # Calculate Predictive Information, either with or without repeats
    aggregated_df = calculate_pi_with_repeats(data, action_suffix, past_window, future_window)

    # Check if mean and sem_pi exist for plotting
    if 'mean_pi' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_pi_linegraphs(aggregated_df)
    else:
        # Otherwise, use the original heatmap plot
        plot_pi_heatmap(aggregated_df)

# ======================================================================================================================================================
# CONTINUOUS ENTROPY USING KOZACHENKO-LEONENKO ESTIMATOR
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import JPackage
from matplotlib.ticker import MaxNLocator
import warnings
from scipy.signal import savgol_filter


import numpy as np
import pandas as pd
from jpype import JPackage

def entropy_continuous(data, variable):
    """
    Calculate continuous entropy for a single variable using the Kozachenko-Leonenko estimator.

    Parameters:
    - data: pandas DataFrame containing the data.
    - variable: column name for which to calculate entropy.

    Returns:
    - DataFrame with Entropy (Bits) per timestep.
    """
    # Clean the data by removing infinities and NaNs
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[variable])
    observations = data_clean[variable].values.astype(float)  # Ensure it's a 1D array

    # Set up the entropy calculator
    calcClass = JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
    calc = calcClass()

    # Initialize the calculator with the number of dimensions
    calc.initialise(1)  # Univariate entropy

    # Set the observations (1D array since it's univariate)
    calc.setObservations(observations)

    # Compute local entropy in nats
    local_result_nats = calc.computeLocalOfPreviousObservations()

    # Convert nats to bits
    local_result_bits = np.array(local_result_nats) / np.log(2)

    # Return as DataFrame
    return pd.DataFrame({'Entropy_Bits': local_result_bits})


def calculate_entropy_continuous(data, action_suffix='_action'):
    """
    Calculate continuous entropy for each agent's actions. Handles both single runs and multiple repeats.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: suffix used to identify action columns.

    Returns:
    - If 'Repeat' is in data:
        - DataFrame with columns ['Agent', 'Timestep', 'Mean_Entropy', 'Std_Error']
      Otherwise:
        - DataFrame with columns ['Entropy_Bits', 'Agent', 'Timestep']
    """
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate entropy for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            for agent_col in action_columns:
                # Extract agent number (assuming format like 'agent_1_action')
                try:
                    agent_number = agent_col.split('_')[1]
                except IndexError:
                    warnings.warn(
                        f"Agent column '{agent_col}' does not follow the expected format 'agent_X_action'. Skipping."
                    )
                    continue

                # Calculate continuous entropy
                local_entropy_df = entropy_continuous(group, agent_col)

                # Append agent number and timestep values
                local_entropy_df['Agent'] = agent_number
                local_entropy_df['Timestep'] = group['Timestep'].values[:len(local_entropy_df)]
                local_entropy_df['Repeat'] = repeat
                result_df_list.append(local_entropy_df)

        if not result_df_list:
            warnings.warn("No valid entropy calculations were performed due to incorrect column naming.")
            return pd.DataFrame()

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Calculate mean and standard error across repeats
        aggregated_df = combined_df.groupby(['Agent', 'Timestep']).agg(
            Mean_Entropy=('Entropy_Bits', 'mean'),
            Std_Error=('Entropy_Bits', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate entropy normally
        for agent_col in action_columns:
            # Extract agent number
            try:
                agent_number = agent_col.split('_')[1]
            except IndexError:
                warnings.warn(
                    f"Agent column '{agent_col}' does not follow the expected format 'agent_X_action'. Skipping."
                )
                continue

            # Calculate continuous entropy
            local_entropy_df = entropy_continuous(data, agent_col)

            # Append agent number and timestep values
            local_entropy_df['Agent'] = agent_number
            local_entropy_df['Timestep'] = data['Timestep'].values[:len(local_entropy_df)]
            result_df_list.append(local_entropy_df)

        if not result_df_list:
            warnings.warn("No valid entropy calculations were performed due to incorrect column naming.")
            return pd.DataFrame()

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        return combined_df


def plot_entropy_heatmap(entropy_measures_df):
    """
    Plot a heatmap or line graphs for the entropy values based on the DataFrame structure.

    Parameters:
    - entropy_measures_df: pandas DataFrame containing entropy measures.
    """
    if {'Mean_Entropy', 'Std_Error'}.issubset(entropy_measures_df.columns):
        # Data has 'Repeat' column: Plot average entropy with standard error as separate subplots
        plot_entropy_linegraphs_subplots(entropy_measures_df)
    else:
        # Single run: Plot heatmap
        plot_entropy_heatmap_single(entropy_measures_df)


def plot_entropy_heatmap_single(entropy_measures_df):
    """
    Plot a heatmap for the entropy values (single run).

    Parameters:
    - entropy_measures_df: pandas DataFrame containing entropy measures.
    """
    if entropy_measures_df.empty:
        warnings.warn("No data available to plot the heatmap.")
        return

    # Create a pivot table for heatmap plotting
    pivot_table = entropy_measures_df.pivot_table(
        index='Agent',
        columns='Timestep',
        values='Entropy_Bits',
        aggfunc='mean'
    )

    if pivot_table.empty:
        warnings.warn("Pivot table is empty. Check your data.")
        return

    # Set up colormap and gridspec for plotting
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # Ratio of heatmap and colorbar

    # Heatmap
    ax = plt.subplot(gs[0])
    sns.heatmap(pivot_table, cmap=cmap, cbar=False, ax=ax, linewidths=0)

    # Set axis labels and title
    ax.set_xlabel('Time', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent', fontsize=20, labelpad=10)
    ax.set_title('Marginal Entropy', fontsize=24, pad=15)

    # Ticks and labels
    ax.set_yticks(np.arange(len(pivot_table.index)) + 0.5)
    ax.set_yticklabels(pivot_table.index, fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Entropy (Bits)', size=18)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show plot
    plt.show()


def plot_entropy_linegraphs_subplots(
    aggregated_df, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=2
):
    """
    Plot separate line graphs for each agent showing average entropy with standard error shading,
    with optional Savitzky-Golay smoothing for smoother lines and shading.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Agent', 'Timestep', 'Mean_Entropy', 'Std_Error']
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    agents = aggregated_df['Agent'].unique()
    num_agents = len(agents)

    if num_agents == 0:
        warnings.warn("No agents found in the data to plot.")
        return

    # Define figure size: width proportional to number of agents, height fixed
    width_per_agent = 6
    height = 6
    total_width = width_per_agent * num_agents
    sns.set(style="whitegrid")

    # Create a subplot for each agent
    fig, axes = plt.subplots(
        1, num_agents, 
        figsize=(width_per_agent * num_agents, height), 
        sharey=True
    )

    # If only one agent, axes is not a list
    if num_agents == 1:
        axes = [axes]

    # Define a color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent].sort_values('Timestep')

        if agent_data.empty:
            warnings.warn(f"No data available for Agent {agent}. Skipping plot.")
            continue

        # Original timesteps and data
        x = agent_data['Timestep'].values
        y = agent_data['Mean_Entropy'].values
        y_err = agent_data['Std_Error'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if savgol_window % 2 == 0 or savgol_window <= 0:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent {agent}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean entropy with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent {agent}',
            color=color_dict[agent],
            linewidth=1  # Thinner lines
        )

        # Shade the standard error with adjusted transparency
        ax.fill_between(
            x,
            y_smooth - y_err_smooth,
            y_smooth + y_err_smooth,
            color=color_dict[agent],
            alpha=0.2  # Increased alpha for better visibility
        )

        ax.set_title(f'Agent {agent}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('Entropy (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Create a single legend for all subplots
    handles = [plt.Line2D([0], [0], color=color_dict[agent], lw=2) for agent in agents]
    labels = [f'Agent {agent}' for agent in agents]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agents', title_fontsize=18)

    plt.suptitle('Average Marginal Entropy with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Show plot
    plt.show()


def entropy_cont(data, action_suffix='_action'):
    """
    Main function to calculate and plot entropy heatmap or line graphs based on the presence of 'Repeat' column.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: suffix used to identify action columns.
    """

    # Calculate continuous entropy
    entropy_measures_df = calculate_entropy_continuous(data, action_suffix)

    if entropy_measures_df.empty:
        warnings.warn("Entropy calculation returned an empty DataFrame. No plots will be generated.")
        return

    # Plot heatmap or line graphs based on data structure
    plot_entropy_heatmap(entropy_measures_df)



# ======================================================================================================================================================
# JOINT ENTROPY USING KOZACHENKO-LEONENKO ESTIMATOR
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from jpype import JPackage, startJVM, isJVMStarted, getDefaultJVMPath
import warnings
from scipy.signal import savgol_filter

import numpy as np



def entropy_continuous_multivariate(data, action_columns):
    import jpype
    import jpype.imports
    from jpype.types import JArray, JDouble
    import numpy as np

    """
    Calculate joint continuous entropy using the Kozachenko-Leonenko estimator for continuous variables.
    This version uses JPype to call the Java class.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_columns: List of column names representing the continuous variables for entropy calculation.

    Returns:
    - NumPy array with Entropy (Bits) per observation.
    """
    # Clean the data by removing infinities and NaNs
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=action_columns)

    # Ensure combined_actions is a 2D array (n_observations x n_variables)
    combined_actions = data_clean[action_columns].values.astype(float)

    # Import the required Java class
    from infodynamics.measures.continuous.kozachenko import EntropyCalculatorMultiVariateKozachenko

    # Initialize the calculator
    calc = EntropyCalculatorMultiVariateKozachenko()
    num_dimensions = combined_actions.shape[1]
    calc.initialise(num_dimensions)

    # Convert combined_actions to a JArray of JDouble (2D array)
    combined_actions_jarray = JArray(JArray(JDouble))(len(combined_actions))
    for i, row in enumerate(combined_actions):
        combined_actions_jarray[i] = JArray(JDouble)(row)

    # Supply the observations
    calc.setObservations(combined_actions_jarray)

    # Compute local entropy in nats
    local_result_nats = calc.computeLocalOfPreviousObservations()

    # Convert nats to bits
    local_result_bits = np.array(local_result_nats) / np.log(2)

    return local_result_bits


def calculate_joint_entropy_continuous(data, action_suffix='_action'):

    """
    Calculates joint continuous entropy over all agents' actions or variables.
    Handles both single runs and multiple repeats.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: Suffix used to identify action columns.

    Returns:
    - If 'Repeat' is not in data:
        - DataFrame with columns ['Entropy_Bits', 'Timestep']
      If 'Repeat' is in data:
        - DataFrame with columns ['Timestep', 'Mean_Entropy', 'Std_Error']
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate joint entropy for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Ensure the group is sorted by Timestep
            group_sorted = group.sort_values('Timestep')

            # Calculate entropy for the continuous variables
            entropy_bits = entropy_continuous_multivariate(group_sorted, action_columns)

            # Create a DataFrame for the entropy values
            joint_entropy_df = pd.DataFrame({
                'Entropy_Bits': entropy_bits,
                'Timestep': group_sorted['Timestep'].values[:len(entropy_bits)],
                'Repeat': repeat
            })
            result_df_list.append(joint_entropy_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Calculate mean and standard error across repeats for each timestep
        aggregated_df = combined_df.groupby('Timestep').agg(
            Mean_Entropy=('Entropy_Bits', 'mean'),
            Std_Error=('Entropy_Bits', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate joint entropy normally
        # Ensure the data is sorted by Timestep
        data_sorted = data.sort_values('Timestep')

        # Calculate entropy for the continuous variables
        entropy_bits = entropy_continuous_multivariate(data_sorted, action_columns)

        # Create a DataFrame for the entropy values
        joint_entropy_df = pd.DataFrame({
            'Entropy_Bits': entropy_bits,
            'Timestep': data_sorted['Timestep'].values[:len(entropy_bits)]
        })

        return joint_entropy_df

def plot_joint_entropy(joint_entropy_df, output_path=None):
    """
    Plots a heatmap or line graph for the joint entropy values based on the DataFrame structure.

    Parameters:
    - joint_entropy_df: pandas DataFrame containing joint entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    if {'Mean_Entropy', 'Std_Error'}.issubset(joint_entropy_df.columns):
        # Data has multiple repeats: Plot average joint entropy with standard error as a line graph
        plot_joint_entropy_linegraph(joint_entropy_df, output_path)
    else:
        # Single run: Plot heatmap
        plot_joint_entropy_heatmap_single(joint_entropy_df, output_path)

def plot_joint_entropy_heatmap_single(joint_entropy_df, output_path=None):
    """
    Plots a heatmap of joint entropy values over timesteps (single run).

    Parameters:
    - joint_entropy_df: pandas DataFrame containing joint entropy measures.
    - output_path: Optional; path to save the plot (without extension).
    """
    # Create a pivot table for heatmap plotting
    pivot_table = joint_entropy_df.pivot_table(
        index='Timestep',
        values='Entropy_Bits',
        aggfunc='mean'
    ).T  # Transpose to have Timesteps on x-axis

    # Define a custom colormap
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Create a gridspec layout to control the colorbar and heatmap separately
    fig = plt.figure(figsize=(10, 2), dpi=300)  # Wide and not very tall
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # 20:1 ratio between heatmap and colorbar

    # Create the heatmap in the first grid cell
    ax = plt.subplot(gs[0])
    sns.heatmap(pivot_table, cmap=cmap, cbar=False, ax=ax, linewidths=0, cbar_kws={"shrink": 0.5})

    # Set axis labels and title with the appropriate sizes
    ax.set_xlabel('Time', fontsize=12, labelpad=5)
    ax.set_ylabel('Joint Entropy', fontsize=12, labelpad=5)  # Keep axis label as 'Joint Entropy'
    ax.set_title('Joint Entropy Over Time', fontsize=14, pad=10)

    # Set the y-tick label to 'Joint Entropy' without changing the axis label
    ax.set_yticks([0.5])  # Only one y-tick for "Joint Entropy"
    ax.set_yticklabels(['Joint Entropy'], fontsize=10)

    # Ensure that all labels are shown on x-axis, and ticks are centered in the middle of each cell
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=10)

    # Create the colorbar in the second grid cell
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Entropy (Bits)', size=12)

    # Adjust layout for a clean look
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in high-resolution formats (PDF and PNG)
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_joint_entropy_linegraph(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plots a line graph of average joint entropy with standard error shading, 
    with optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Timestep', 'Mean_Entropy', 'Std_Error']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Timestep', 'Mean_Entropy', 'Std_Error'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Sort DataFrame by 'Timestep'
    aggregated_df = aggregated_df.sort_values('Timestep')

    # Extract data
    x = aggregated_df['Timestep'].values
    y = aggregated_df['Mean_Entropy'].values
    y_err = aggregated_df['Std_Error'].values

    # Apply Savitzky-Golay filter if enabled
    if use_savgol:
        # Validate Savitzky-Golay parameters
        if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
            warnings.warn(
                f"Savitzky-Golay window size must be a positive odd integer. "
                f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_window > len(x):
            warnings.warn(
                f"Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_polyorder >= savgol_window:
            warnings.warn(
                f"Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        else:
            try:
                # Apply Savitzky-Golay filter
                y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
            except Exception as e:
                warnings.warn(
                    f"Savitzky-Golay filtering failed with error: {e}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
    else:
        # No smoothing; use original data
        y_smooth = y
        y_err_smooth = y_err

    # Initialize the plot
    plt.figure(figsize=(5, 3), dpi=300)
    sns.set(style="whitegrid")

    # Define color palette
    color = sns.color_palette("viridis", 1)[0]

    # Plot mean entropy line
    plt.plot(
        x,
        y_smooth,
        label='Mean Joint Entropy',
        color=color,
        linewidth=1  # Thinner line
    )

    # Plot standard error shading
    plt.fill_between(
        x,
        y_smooth - y_err_smooth,
        y_smooth + y_err_smooth,
        color=color,
        alpha=0.2,  # Increased alpha for better visibility
        label='Standard Error'
    )

    # Set labels and title
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Joint Entropy (Bits)', fontsize=7)
    plt.title('Average Joint Entropy Over Time with Standard Error', fontsize=8)

    # Customize ticks
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Add legend
    plt.legend(fontsize=6)

    # Adjust layout for a clean look
    plt.tight_layout()

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def joint_entropy_cont(data, action_suffix='_action'):
    """
    Main function to calculate and plot joint entropy heatmap or line graph based on the presence of 'Repeat' column.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: Suffix used to identify action columns.
    """
    # Calculate joint entropy
    joint_entropy_df = calculate_joint_entropy_continuous(data, action_suffix)

    # Plot heatmap or line graph based on data structure
    plot_joint_entropy(joint_entropy_df)



# ======================================================================================================================================================
# CONDITIONAL MUTUAL INFORMATION  (CONTINUOUS WITH KSG)
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from jpype import JPackage, JArray, JInt
from matplotlib.ticker import MaxNLocator

def initialize_cmi_calculator_cont():
    """
    Initialize the Conditional Mutual Information calculator from JIDT.
    """
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov1
    return calcClass()  # Kraskov estimator for continuous variables

def calculate_local_cmi_for_pair_cont(data, agent_col_i, agent_col_j, other_vars, calc):
    # Convert the source and destination to native Python lists (for 1D arrays)
    source = data[agent_col_i].values.astype(float).tolist()
    destination = data[agent_col_j].values.astype(float).tolist()

    # Check if other_vars contains multiple conditionals or just one
    if other_vars.ndim == 1:
        # Single conditional, pass as a 1D array (double[])
        conditional = other_vars.astype(float).tolist()
        calc.initialise(1, 1, 1)  # Initialize with single condition
        calc.setObservations(source, destination, conditional)  # Pass 1D arrays
    elif other_vars.ndim == 2:
        # Multiple conditionals, pass as 2D array (double[][])
        num_conditionals = other_vars.shape[1]  # Number of conditionals
        conditionals_2d = other_vars.astype(float).tolist()  # 2D list
        # Reshape source and destination to 2D arrays (double[][])
        source_2d = [[val] for val in source]
        destination_2d = [[val] for val in destination]
        calc.initialise(1, 1, num_conditionals)  # Initialize with multiple conditionals
        calc.setObservations(source_2d, destination_2d, conditionals_2d)  # Pass 2D arrays
    else:
        raise ValueError("Invalid dimensionality for conditionals.")

    # Compute and return local CMI using the correct method
    local_cmi = calc.computeLocalOfPreviousObservations()  # This returns an array of local CMI values

    return local_cmi

def calculate_local_conditional_mutual_information_cont(data, action_suffix):
    """
    Calculate local Conditional Mutual Information (CMI) for all pairs of agents.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    num_agents = len(action_columns)

    # Initialize the JIDT CMI calculator
    calc = initialize_cmi_calculator_cont()

    # Prepare result storage for local CMI
    local_cmi_dict = {}

    # Loop over each pair of agents and calculate local CMI
    for i, agent_col_i in enumerate(action_columns):
        for j, agent_col_j in enumerate(action_columns):
            if i != j:
                # Prepare other variables as the condition
                other_vars = data[[col for k, col in enumerate(action_columns) if k != i and k != j]].values.astype(float)

                # Debugging information
                # print(f"Calculating CMI for {agent_col_i} and {agent_col_j} with other_vars shape: {other_vars.shape}")

                # Calculate local CMI between agent i and agent j
                local_cmi = calculate_local_cmi_for_pair_cont(data, agent_col_i, agent_col_j, other_vars, calc)

                # Store the result (assuming local_cmi is a list/array over timesteps)
                pair_key = f"{i} ; {j}"
                local_cmi_dict[pair_key] = local_cmi  # local_cmi must be an array or list, not scalar

    # Convert the result dictionary to a DataFrame
    local_cmi_df = pd.DataFrame(local_cmi_dict)
    # Preserve the 'Timestep' column from the original data
    local_cmi_df['Timestep'] = data['Timestep'].values

    return local_cmi_df

def calculate_cmi_with_repeats_cont(data, action_suffix):
    """
    Calculates CMI for agent pairs and averages over repeats.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate local CMI for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Reset the Timestep to start from 1 for each repeat
            group = group.copy()  # Avoid modifying the original data
            group['Timestep'] = np.arange(1, len(group) + 1)

            # Calculate CMI for this repeat
            local_cmi_df = calculate_local_conditional_mutual_information_cont(group, action_suffix)
            local_cmi_df['Repeat'] = repeat
            result_df_list.append(local_cmi_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Pair', value_name='Local CMI')

        # Ensure 'Local CMI' column is numeric after melting
        melted_df['Local CMI'] = pd.to_numeric(melted_df['Local CMI'], errors='coerce')

        # Drop any rows where 'Local CMI' is NaN
        melted_df = melted_df.dropna(subset=['Local CMI'])

        # Now we group by 'Pair' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Pair', 'Timestep']).agg(
            mean_cmi=('Local CMI', 'mean'),
            sem_cmi=('Local CMI', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate CMI normally
        return calculate_local_conditional_mutual_information_cont(data, action_suffix)

def plot_local_conditional_mutual_information_heatmap_cont(local_cmi_df):
    """
    Plot a heatmap of the local Conditional Mutual Information (CMI) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_cmi_melted = local_cmi_df.melt(id_vars='Timestep', var_name='Pair', value_name='Local CMI')

    # Create a pivot table for heatmap input
    heatmap_data = local_cmi_melted.pivot_table(index='Pair', columns='Timestep', values='Local CMI')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title with updated fontsize and padding
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent Pairs', fontsize=20, labelpad=10)
    ax.set_title('Conditional Mutual Information', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local CMI (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_cmi_linegraphs_cont(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot CMI line graphs for all agent pairs with standard error shading, 
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Pair', 'Timestep', 'mean_cmi', 'sem_cmi']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Pair', 'Timestep', 'mean_cmi', 'sem_cmi'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agent pairs
    agent_pairs = aggregated_df['Pair'].unique()
    num_pairs = len(agent_pairs)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_pairs // max_per_row) + (1 if num_pairs % max_per_row != 0 else 0)

    # Set figure size based on number of pairs and rows
    width_per_pair = 6
    height_per_row = 6
    total_width = min(num_pairs, max_per_row) * width_per_pair
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_pairs), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair].sort_values('Timestep')

        # Extract data
        x = pair_data['Timestep'].values
        y = pair_data['mean_cmi'].values
        y_err = pair_data['sem_cmi'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent Pair {pair}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean CMI with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent Pair {pair}',
            color=color_dict[pair],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_cmi' in pair_data.columns:
            if pair_data['sem_cmi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[pair],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CMI (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agent_pairs):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[pair], lw=2) for pair in agent_pairs]
    labels = [f'Agent Pair {pair}' for pair in agent_pairs]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agent Pairs', title_fontsize=18)

    plt.suptitle('Conditional Mutual Information with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()


def conditional_mutual_information_cont(data, action_suffix='_action'):
    """
    Main function to calculate local CMI, plot the line graphs (for repeats), and display the results.
    """
    # Calculate CMI, either with or without repeats
    aggregated_df = calculate_cmi_with_repeats_cont(data, action_suffix)
    # print(aggregated_df)
    
    # Check if mean and sem_cmi exist for plotting
    if 'mean_cmi' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_cmi_linegraphs_cont(aggregated_df, use_savgol=True)
    else:
        # Otherwise, use the original heatmap plot
        plot_local_conditional_mutual_information_heatmap_cont(aggregated_df)



# ======================================================================================================================================================
# CONDITIONAL TRANSFER ENTROPY (CONTINUOUS)
# ======================================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jpype import JPackage, JArray
from matplotlib.ticker import MaxNLocator

def initialize_cte_calculator_cont():
    """
    Initialize the Conditional Transfer Entropy calculator for continuous variables (KSG).
    """
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov
    return calcClass()

def calculate_local_cte_for_pair_cont(data, agent_col_i, agent_col_j, other_vars, calc):
    # Convert the source and destination into 1D arrays (double[])
    source = data[agent_col_i].values.astype(float).tolist()
    destination = data[agent_col_j].values.astype(float).tolist()

    # Check if other_vars contains multiple conditionals or just one
    if other_vars.ndim == 1:
        # Single conditional, pass as 1D array
        conditional = other_vars.astype(float).tolist()  # Convert to 1D list
        calc.initialise(1, 1, 1)  # Initialize with single condition
        calc.setObservations(source, destination, conditional)  # Use 1D arrays for conditionals
    elif other_vars.ndim == 2:
        # Multiple conditionals, pass as 2D array (double[][])
        num_conditionals = other_vars.shape[1]  # Number of conditionals
        conditionals_2d = other_vars.astype(float).tolist()  # Convert to 2D list

        # Ensure that source and destination are 1D arrays (double[])
        calc.initialise(1, 1, num_conditionals)  # Initialize with multiple conditionals
        calc.setObservations(source, destination, conditionals_2d)  # Use 2D arrays for conditionals
    else:
        raise ValueError("Invalid dimensionality for conditionals.")

    # Compute and return local CTE using the correct method
    local_cte = calc.computeLocalOfPreviousObservations()  # This returns an array of local CTE values

    return local_cte

def calculate_local_conditional_transfer_entropy_cont(data, action_suffix):
    """
    Calculate local Conditional Transfer Entropy (CTE) for all pairs of agents.
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    

    # Initialize the JIDT CTE calculator
    calc = initialize_cte_calculator_cont()

    # calc.setProperty("k_HISTORY", "10")
    # calc.setProperty("k_TAU", "10")
    # calc.setProperty("l_HISTORY", "10")
    # calc.setProperty("l_TAU", "10")
    # calc.setProperty("DELAY", "10")
    # calc.setProperty("COND_EMBED_LENGTHS", "10")
    # calc.setProperty("COND_TAUS", "10")
    # calc.setProperty("COND_DELAYS", "10")

    # Prepare result storage for local CTE
    local_cte_dict = {}

    # Loop over each pair of agents and calculate local CTE
    for i, agent_col_i in enumerate(action_columns):
        for j, agent_col_j in enumerate(action_columns):
            if i != j:
                # Prepare other variables as the condition (excluding i and j)
                other_vars = data[[col for k, col in enumerate(action_columns) if k != i and k != j]].values.astype(float)

                # Calculate local CTE between agent i and agent j
                local_cte = calculate_local_cte_for_pair_cont(data, agent_col_i, agent_col_j, other_vars, calc)

                # Store the result (assuming local_cte is a list/array over timesteps)
                pair_key = f"{i} → {j}"
                local_cte_dict[pair_key] = local_cte

    # Convert the result dictionary to a DataFrame
    local_cte_df = pd.DataFrame(local_cte_dict)
    # Preserve the 'Timestep' column from the original data
    local_cte_df['Timestep'] = data['Timestep'].values

    return local_cte_df

def calculate_cte_with_repeats_cont(data, action_suffix):
    """
    Calculates CTE for agent pairs and averages over repeats (continuous case).
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate local CTE for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Reset the Timestep to start from 1 for each repeat
            group = group.copy()  # Avoid modifying the original data
            group['Timestep'] = np.arange(1, len(group) + 1)

            # Calculate CTE for this repeat
            local_cte_df = calculate_local_conditional_transfer_entropy_cont(group, action_suffix)
            local_cte_df['Repeat'] = repeat
            result_df_list.append(local_cte_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Pair', value_name='Local CTE')

        # Ensure 'Local CTE' column is numeric after melting
        melted_df['Local CTE'] = pd.to_numeric(melted_df['Local CTE'], errors='coerce')

        # Drop any rows where 'Local CTE' is NaN
        melted_df = melted_df.dropna(subset=['Local CTE'])

        # Now we group by 'Pair' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Pair', 'Timestep']).agg(
            mean_cte=('Local CTE', 'mean'),
            sem_cte=('Local CTE', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate CTE normally
        return calculate_local_conditional_transfer_entropy_cont(data, action_suffix)

# (The plotting functions remain the same as in the discrete version)
    
def plot_local_conditional_transfer_entropy_heatmap(local_cte_df):
    """
    Plot a heatmap of the local Conditional Transfer Entropy (CTE) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_cte_melted = local_cte_df.melt(id_vars='Timestep', var_name='Pair', value_name='Local CTE')

    # Create a pivot table for heatmap input
    heatmap_data = local_cte_melted.pivot_table(index='Pair', columns='Timestep', values='Local CTE')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title with updated fontsize and padding
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent Pairs', fontsize=20, labelpad=10)
    ax.set_title('Conditional Transfer Entropy', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local CTE (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_cte_linegraphs(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot CTE line graphs for all agent pairs with standard error shading,
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Pair', 'Timestep', 'mean_cte', 'sem_cte']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Pair', 'Timestep', 'mean_cte', 'sem_cte'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agent pairs
    agent_pairs = aggregated_df['Pair'].unique()
    num_pairs = len(agent_pairs)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_pairs // max_per_row) + (1 if num_pairs % max_per_row != 0 else 0)

    # Set figure size based on number of pairs and rows
    width_per_pair = 6
    height_per_row = 6
    total_width = min(num_pairs, max_per_row) * width_per_pair
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_pairs), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair].sort_values('Timestep')

        # Extract data
        x = pair_data['Timestep'].values
        y = pair_data['mean_cte'].values
        y_err = pair_data['sem_cte'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent Pair {pair}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent Pair {pair}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean CTE with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent Pair {pair}',
            color=color_dict[pair],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_cte' in pair_data.columns:
            if pair_data['sem_cte'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[pair],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CTE (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agent_pairs):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[pair], lw=2) for pair in agent_pairs]
    labels = [f'Agent Pair {pair}' for pair in agent_pairs]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agent Pairs', title_fontsize=18)

    plt.suptitle('Conditional Transfer Entropy with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def conditional_transfer_entropy_cont(data, action_suffix='_action'):
    """
    Main function to calculate local CTE, plot the heatmap, or line graphs if repeats are present.

    Parameters:
    - data: pandas DataFrame containing relevant data for CTE calculation.
    - action_suffix: Suffix to identify action-related columns in the DataFrame.
    """
    # Calculate CTE, either with or without repeats
    aggregated_df = calculate_cte_with_repeats_cont(data, action_suffix)

    # Check if mean and sem_cte exist for plotting
    if 'mean_cte' in aggregated_df.columns and 'sem_cte' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_cte_linegraphs(aggregated_df)
    else:
        # Otherwise, use the original heatmap plot
        plot_local_conditional_transfer_entropy_heatmap(aggregated_df)



# ======================================================================================================================================================
# PREDICTIVE INFORMATION (CONTINUOUS WITH KSG)
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jpype import JPackage
from matplotlib.ticker import MaxNLocator

def initialize_pi_calculator_cont():
    """
    Initialize the Predictive Information calculator for continuous variables (KSG).
    """
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
    return calcClass()

def calculate_local_pi_for_agent_cont(data, agent_col, calc):
    """
    Calculate local Predictive Information (PI) for a specific agent (continuous case).
    """
    # Convert the agent's data into a native Python list (1D array)
    actions = data[agent_col].values.astype(float).tolist()
    
    # Start adding observations
    calc.startAddObservations()
    
    # Add observations to the calculator
    calc.addObservations(actions)
    
    # Finalize adding observations
    calc.finaliseAddObservations()
    
    # Compute and return local PI using the correct method
    local_pi = calc.computeLocalOfPreviousObservations()  # This returns an array of local PI values
    
    return local_pi

def calculate_local_predictive_information_cont(data, action_suffix, past_window=1, future_window=1):
    """
    Calculate local Predictive Information (PI) for all agents (continuous case).
    """
    # Find all agent action columns
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]

    # Initialize the PI calculator
    calc = initialize_pi_calculator_cont()
    calc.initialise(past_window)  # Set past window length (you can adjust this as needed)

    # Prepare result storage for local PI
    local_pi_dict = {}

    # Loop over each agent and calculate local PI
    for agent_col in action_columns:
        agent_number = agent_col.split('_')[1]  # Extract agent number

        # Calculate local PI for this agent
        local_pi = calculate_local_pi_for_agent_cont(data, agent_col, calc)

        # Store the result
        local_pi_dict[f'Agent {agent_number}'] = local_pi

    # Convert the result dictionary to a DataFrame
    local_pi_df = pd.DataFrame(local_pi_dict)
    local_pi_df['Timestep'] = data['Timestep'].values

    return local_pi_df

def calculate_pi_with_repeats_cont(data, action_suffix, past_window=1, future_window=1):
    """
    Calculate Predictive Information (PI) for agent actions, averaging over repeats if present (continuous case).
    """
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate PI for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Calculate PI for this repeat
            local_pi_df = calculate_local_predictive_information_cont(group, action_suffix, past_window, future_window)
            local_pi_df['Repeat'] = repeat
            result_df_list.append(local_pi_df)

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Agent', value_name='Local PI')

        # Ensure 'Local PI' column is numeric after melting
        melted_df['Local PI'] = pd.to_numeric(melted_df['Local PI'], errors='coerce')

        # Drop any rows where 'Local PI' is NaN
        melted_df = melted_df.dropna(subset=['Local PI'])

        # Group by 'Agent' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Agent', 'Timestep']).agg(
            mean_pi=('Local PI', 'mean'),
            sem_pi=('Local PI', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate PI normally
        return calculate_local_predictive_information_cont(data, action_suffix, past_window, future_window)


def plot_pi_heatmap(local_pi_df):
    """
    Plot a heatmap of the local Predictive Information (PI) over time.
    """
    # Melt the DataFrame for Seaborn heatmap
    local_pi_melted = local_pi_df.melt(id_vars='Timestep', var_name='Agent', value_name='Local PI')

    # Create a pivot table for heatmap input
    heatmap_data = local_pi_melted.pivot_table(index='Agent', columns='Timestep', values='Local PI')

    # Set up the plot
    cmap = sns.color_palette("viridis", as_cmap=True)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

    # Heatmap plot
    ax = plt.subplot(gs[0])
    sns.heatmap(heatmap_data, cmap=cmap, cbar=False, ax=ax, linewidths=0)
    
    # Set axis labels and title
    ax.set_xlabel('Timestep', fontsize=20, labelpad=10)
    ax.set_ylabel('Agent', fontsize=20, labelpad=10)
    ax.set_title('Predictive Information', fontsize=24, pad=15)

    # Y-axis: Ensure agent pairs are displayed correctly
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels([label for label in heatmap_data.index], fontsize=16)

    # X-axis: Timestep ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=16)

    # Colorbar setup
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Local PI (Bits)', size=20)

    # Final layout adjustments and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings

def plot_pi_linegraphs(
    aggregated_df, 
    output_path=None, 
    use_savgol=True,
    savgol_window=21,
    savgol_polyorder=3
):
    """
    Plot Predictive Information (PI) line graphs for all agents with standard error shading,
    incorporating optional Savitzky-Golay smoothing.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Agent', 'Timestep', 'mean_pi', 'sem_pi']
    - output_path: Optional; path to save the plot (without extension).
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    # Validate input DataFrame
    required_columns = {'Agent', 'Timestep', 'mean_pi', 'sem_pi'}
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Extract the agents
    agents = aggregated_df['Agent'].unique()
    num_agents = len(agents)

    # Define maximum number of subplots per row
    max_per_row = 3
    num_rows = (num_agents // max_per_row) + (1 if num_agents % max_per_row != 0 else 0)

    # Set figure size based on number of agents and rows
    width_per_agent = 6
    height_per_row = 6
    total_width = min(num_agents, max_per_row) * width_per_agent
    total_height = num_rows * height_per_row
    sns.set(style="whitegrid")

    # Create subplots for each agent, distributed across multiple rows if needed
    fig, axes = plt.subplots(
        num_rows, 
        min(max_per_row, num_agents), 
        figsize=(total_width, total_height), 
        sharey=True
    )

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_agents > 1 else [axes]  # Single row with multiple agents or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent].sort_values('Timestep')

        # Extract data
        x = agent_data['Timestep'].values
        y = agent_data['mean_pi'].values
        y_err = agent_data['sem_pi'].values

        if use_savgol:
            # Validate Savitzky-Golay parameters
            if not isinstance(savgol_window, int) or savgol_window <= 0 or savgol_window % 2 == 0:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size must be a positive odd integer. "
                    f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_window > len(x):
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            elif savgol_polyorder >= savgol_window:
                warnings.warn(
                    f"Agent {agent}: Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                    f"Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
            else:
                try:
                    # Apply Savitzky-Golay filter
                    y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                    y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
                except Exception as e:
                    warnings.warn(
                        f"Agent {agent}: Savitzky-Golay filtering failed with error: {e}. "
                        f"Skipping Savitzky-Golay filtering."
                    )
                    y_smooth = y
                    y_err_smooth = y_err
        else:
            # No smoothing; use original data
            y_smooth = y
            y_err_smooth = y_err

        # Plot the mean PI with a thinner line
        ax.plot(
            x,
            y_smooth,
            label=f'Agent {agent}',
            color=color_dict[agent],
            linewidth=1  # Thinner lines
        )

        # Plot the standard error shading
        if 'sem_pi' in agent_data.columns:
            if agent_data['sem_pi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(
                    x,
                    y_smooth - y_err_smooth,
                    y_smooth + y_err_smooth,
                    color=color_dict[agent],
                    alpha=0.2,  # Increased alpha for better visibility
                    label='Standard Error'
                )

        # Set axis labels and title
        ax.set_title(f'Agent {agent}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('Predictive Information (Bits)', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend().set_visible(False)  # Hide individual legends

    # Remove empty subplots if there are any
    for ax in axes[len(agents):]:
        fig.delaxes(ax)

    # Create a single legend
    handles = [plt.Line2D([0], [0], color=color_dict[agent], lw=2) for agent in agents]
    labels = [f'Agent {agent}' for agent in agents]
    fig.legend(handles, labels, loc='upper right', fontsize=16, title='Agents', title_fontsize=18)

    plt.suptitle('Predictive Information with Standard Error', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def predictive_information_cont(data, action_suffix='_action', past_window=1, future_window=1):
    """
    Main function to calculate local Predictive Information, plot the heatmap, or line graphs if repeats are present (continuous case).
    """
    # Calculate Predictive Information, either with or without repeats
    aggregated_df = calculate_pi_with_repeats_cont(data, action_suffix, past_window, future_window)

    # Check if mean and sem_pi exist for plotting
    if 'mean_pi' in aggregated_df.columns:
        # If repeats are present, plot line graphs with standard error shading
        plot_pi_linegraphs(aggregated_df)
    else:
        # Otherwise, use the original heatmap plot
        plot_pi_heatmap(aggregated_df)


# ======================================================================================================================================================
# TOTAL CORRELATION CALCULATION: SUM OF MARGINAL ENTROPIES MINUS JOINT ENTROPY
# ======================================================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import warnings
from scipy.signal import savgol_filter

# Import your existing entropy functions
# Ensure that these functions are defined in your project as per your initial code
# from your_entropy_module import entropy_continuous, entropy_continuous_multivariate

# ======================================================================================================================================================
# TOTAL CORRELATION FUNCTIONS
# ======================================================================================================================================================

def calculate_total_correlation(data, action_suffix='_action'):
    """
    Calculate Total Correlation as the sum of marginal entropies minus joint entropy.
    Handles both single runs and multiple repeats.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: suffix used to identify action columns.

    Returns:
    - If 'Repeat' is in data:
        - DataFrame with columns ['Timestep', 'Mean_Total_Correlation', 'Std_Error']
      Otherwise:
        - DataFrame with columns ['Timestep', 'Total_Correlation']
    """
    action_columns = [col for col in data.columns if col.endswith(action_suffix)]
    result_df_list = []

    if 'Repeat' in data.columns:
        # Multiple runs: Calculate total correlation for each repeat
        grouped = data.groupby('Repeat')
        for repeat, group in grouped:
            # Ensure the group is sorted by Timestep
            group_sorted = group.sort_values('Timestep')

            # Calculate marginal entropies for each agent
            marginal_entropies = {}
            for agent_col in action_columns:
                try:
                    agent_number = agent_col.split('_')[1]
                except IndexError:
                    warnings.warn(
                        f"Agent column '{agent_col}' does not follow the expected format 'agent_X_action'. Skipping."
                    )
                    continue

                entropy_df = entropy_continuous(group_sorted, agent_col)
                if entropy_df is not None and not entropy_df.empty:
                    marginal_entropies[agent_number] = entropy_df['Entropy_Bits'].values
                else:
                    warnings.warn(
                        f"Entropy calculation returned empty for Agent {agent_number} in Repeat {repeat}."
                    )
                    marginal_entropies[agent_number] = np.zeros(len(group_sorted))

            if not marginal_entropies:
                warnings.warn(f"No valid marginal entropies calculated for Repeat {repeat}. Skipping.")
                continue

            # Sum of marginal entropies per timestep
            sum_marginals = np.sum(np.column_stack(list(marginal_entropies.values())), axis=1)

            # Calculate joint entropy
            joint_entropy = entropy_continuous_multivariate(group_sorted, action_columns)
            if joint_entropy is None:
                warnings.warn(f"Joint entropy calculation returned None for Repeat {repeat}. Skipping.")
                continue

            # Ensure sum_marginals and joint_entropy have the same length
            min_length = min(len(sum_marginals), len(joint_entropy))
            sum_marginals = sum_marginals[:min_length]
            joint_entropy = joint_entropy[:min_length]

            # Calculate Total Correlation
            total_correlation = sum_marginals - joint_entropy

            # Create a DataFrame for the total correlation values
            total_corr_df = pd.DataFrame({
                'Total_Correlation': total_correlation,
                'Timestep': group_sorted['Timestep'].values[:min_length],
                'Repeat': repeat
            })
            result_df_list.append(total_corr_df)

        if not result_df_list:
            warnings.warn("No valid total correlation calculations were performed due to incorrect column naming or empty data.")
            return pd.DataFrame()

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Calculate mean and standard error across repeats for each timestep
        aggregated_df = combined_df.groupby('Timestep').agg(
            Mean_Total_Correlation=('Total_Correlation', 'mean'),
            Std_Error=('Total_Correlation', 'sem')
        ).reset_index()

        return aggregated_df
    else:
        # Single run: Calculate total correlation normally
        # Sort by Timestep
        data_sorted = data.sort_values('Timestep')

        # Calculate marginal entropies for each agent
        marginal_entropies = {}
        for agent_col in action_columns:
            try:
                agent_number = agent_col.split('_')[1]
            except IndexError:
                warnings.warn(
                    f"Agent column '{agent_col}' does not follow the expected format 'agent_X_action'. Skipping."
                )
                continue

            entropy_df = entropy_continuous(data_sorted, agent_col)
            if entropy_df is not None and not entropy_df.empty:
                marginal_entropies[agent_number] = entropy_df['Entropy_Bits'].values
            else:
                warnings.warn(
                    f"Entropy calculation returned empty for Agent {agent_number}."
                )
                marginal_entropies[agent_number] = np.zeros(len(data_sorted))

        if not marginal_entropies:
            warnings.warn("No valid marginal entropies calculated.")
            return pd.DataFrame()

        # Sum of marginal entropies per timestep
        sum_marginals = np.sum(np.column_stack(list(marginal_entropies.values())), axis=1)

        # Calculate joint entropy
        joint_entropy = entropy_continuous_multivariate(data_sorted, action_columns)
        if joint_entropy is None:
            warnings.warn("Joint entropy calculation returned None.")
            return pd.DataFrame()

        # Ensure sum_marginals and joint_entropy have the same length
        min_length = min(len(sum_marginals), len(joint_entropy))
        sum_marginals = sum_marginals[:min_length]
        joint_entropy = joint_entropy[:min_length]

        # Calculate Total Correlation
        total_correlation = sum_marginals - joint_entropy

        # Create a DataFrame for the total correlation values
        total_corr_df = pd.DataFrame({
            'Total_Correlation': total_correlation,
            'Timestep': data_sorted['Timestep'].values[:min_length]
        })

        return total_corr_df


def plot_total_correlation(total_corr_df):
    """
    Plots Total Correlation based on the DataFrame structure.

    Parameters:
    - total_corr_df: pandas DataFrame containing total correlation measures.
      - If 'Mean_Total_Correlation' and 'Std_Error' are present: plots average with standard error.
      - Else: plots heatmap.
    """
    if total_corr_df.empty:
        warnings.warn("Total correlation DataFrame is empty. No plots will be generated.")
        return

    if {'Mean_Total_Correlation', 'Std_Error'}.issubset(total_corr_df.columns):
        # Data has multiple repeats: Plot average total correlation with standard error
        plot_total_correlation_linegraph(total_corr_df)
    else:
        # Single run: Plot heatmap
        plot_total_correlation_heatmap(total_corr_df)


def plot_total_correlation_heatmap(total_corr_df):
    """
    Plot a heatmap for the total correlation values (single run).

    Parameters:
    - total_corr_df: pandas DataFrame containing total correlation measures.
    """
    if total_corr_df.empty:
        warnings.warn("No data available to plot the heatmap.")
        return

    # Create a pivot table for heatmap plotting
    # Since Total Correlation is a single value per timestep, represent it as a heatmap with one row
    pivot_table = total_corr_df.pivot_table(
        index=['Total_Correlation'],
        columns='Timestep',
        values='Total_Correlation',
        aggfunc='mean'
    )

    if pivot_table.empty:
        warnings.warn("Pivot table is empty. Check your data.")
        return

    # Define a custom colormap
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Create a gridspec layout to control the colorbar and heatmap separately
    fig = plt.figure(figsize=(12, 2), dpi=300)  # Wide and not very tall
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # 20:1 ratio between heatmap and colorbar

    # Heatmap
    ax = plt.subplot(gs[0])
    sns.heatmap(pivot_table, cmap=cmap, cbar=False, ax=ax, linewidths=0)

    # Set axis labels and title
    ax.set_xlabel('Time', fontsize=14, labelpad=10)
    ax.set_ylabel('Total Correlation', fontsize=14, labelpad=10)
    ax.set_title('Total Correlation Heatmap', fontsize=16, pad=15)

    # Ticks and labels
    ax.set_yticks([])  # Hide y-ticks since it's a single row
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()], fontsize=12)

    # Colorbar
    cbar_ax = plt.subplot(gs[1])
    norm = plt.Normalize(vmin=pivot_table.values.min(), vmax=pivot_table.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Total Correlation (Bits)', size=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show plot
    plt.show()


def plot_total_correlation_linegraph(aggregated_df, use_savgol=True, savgol_window=21, savgol_polyorder=3):
    """
    Plots a line graph of average total correlation with standard error shading,
    with optional Savitzky-Golay smoothing for smoother lines.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Timestep', 'Mean_Total_Correlation', 'Std_Error']
    - use_savgol: Boolean; if True, apply Savitzky-Golay filter to smooth the data.
    - savgol_window: Window length for Savitzky-Golay filter (must be a positive odd integer and <= number of data points).
    - savgol_polyorder: Polynomial order for Savitzky-Golay filter (must be less than savgol_window).
    """
    if aggregated_df.empty:
        warnings.warn("Aggregated DataFrame is empty. No plots will be generated.")
        return

    # Sort DataFrame by 'Timestep'
    aggregated_df = aggregated_df.sort_values('Timestep')

    # Extract data
    x = aggregated_df['Timestep'].values
    y = aggregated_df['Mean_Total_Correlation'].values
    y_err = aggregated_df['Std_Error'].values

    # Apply Savitzky-Golay filter if enabled
    if use_savgol:
        # Validate Savitzky-Golay parameters
        if savgol_window % 2 == 0 or savgol_window <= 0:
            warnings.warn(
                f"Savitzky-Golay window size must be a positive odd integer. "
                f"Received window={savgol_window}. Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_window > len(x):
            warnings.warn(
                f"Savitzky-Golay window size ({savgol_window}) is larger than the number of data points ({len(x)}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        elif savgol_polyorder >= savgol_window:
            warnings.warn(
                f"Savitzky-Golay polyorder ({savgol_polyorder}) must be less than window size ({savgol_window}). "
                f"Skipping Savitzky-Golay filtering."
            )
            y_smooth = y
            y_err_smooth = y_err
        else:
            try:
                # Apply Savitzky-Golay filter
                y_smooth = savgol_filter(y, window_length=savgol_window, polyorder=savgol_polyorder)
                y_err_smooth = savgol_filter(y_err, window_length=savgol_window, polyorder=savgol_polyorder)
            except Exception as e:
                warnings.warn(
                    f"Savitzky-Golay filtering failed with error: {e}. Skipping Savitzky-Golay filtering."
                )
                y_smooth = y
                y_err_smooth = y_err
    else:
        # No smoothing; use original data
        y_smooth = y
        y_err_smooth = y_err

    # Initialize the plot
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(style="whitegrid")

    # Define color palette
    color = sns.color_palette("viridis", 1)[0]

    # Plot mean total correlation line
    plt.plot(
        x,
        y_smooth,
        label='Mean Total Correlation',
        color=color,
        linewidth=2
    )

    # Plot standard error shading
    plt.fill_between(
        x,
        y_smooth - y_err_smooth,
        y_smooth + y_err_smooth,
        color=color,
        alpha=0.3,
        label='Standard Error'
    )

    # Set labels and title
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Total Correlation (Bits)', fontsize=16)
    plt.title('Average Total Correlation Over Time with Standard Error', fontsize=18)

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add legend
    plt.legend(fontsize=14)

    # Adjust layout for a clean look
    plt.tight_layout()

    # Show plot
    plt.show()


def total_correlation_cont(data, action_suffix='_action'):
    """
    Main function to calculate and plot total correlation heatmap or line graph based on the presence of 'Repeat' column.

    Parameters:
    - data: pandas DataFrame containing the data.
    - action_suffix: Suffix used to identify action columns.
    """
    # Calculate total correlation
    total_corr_df = calculate_total_correlation(data, action_suffix)

    if total_corr_df.empty:
        warnings.warn("Total correlation calculation returned an empty DataFrame. No plots will be generated.")
        return

    # Plot heatmap or line graph based on data structure
    plot_total_correlation(total_corr_df)
