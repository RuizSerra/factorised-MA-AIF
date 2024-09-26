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

import numpy as np
import pandas as pd

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

    # Save the figure
    if output_path is not None:
        plt.savefig(f'{output_path}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()


def plot_entropy_linegraphs_subplots(aggregated_df, output_path=None):
    """
    Plot separate line graphs for each agent showing average entropy with standard error shading.

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Agent', 'Timestep', 'Mean_Entropy', 'Std_Error']
    - output_path: Optional; path to save the plot (without extension).
    """
    agents = aggregated_df['Agent'].unique()
    num_agents = len(agents)

    # Define figure size: width proportional to number of agents, height fixed
    # For example, width=6 inches per agent, height=6 inches
    width_per_agent = 6
    height = 6
    total_width = width_per_agent * num_agents
    plt.figure(figsize=(total_width, height), dpi=300)
    sns.set(style="whitegrid")

    # Create a subplot for each agent
    fig, axes = plt.subplots(1, num_agents, figsize=(width_per_agent * num_agents, height), sharey=True)

    # If only one agent, axes is not a list
    if num_agents == 1:
        axes = [axes]

    # Define a color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent]
        ax.plot(
            agent_data['Timestep'],
            agent_data['Mean_Entropy'],
            label=f'Agent {agent}',
            color=color_dict[agent],
            linewidth=2
        )
        ax.fill_between(
            agent_data['Timestep'],
            agent_data['Mean_Entropy'] - agent_data['Std_Error'],
            agent_data['Mean_Entropy'] + agent_data['Std_Error'],
            color=color_dict[agent],
            alpha=0.3
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


def plot_joint_entropy_linegraph(aggregated_df, output_path=None):
    """
    Plots a line graph of average joint entropy with standard error shading (multiple repeats).

    Parameters:
    - aggregated_df: pandas DataFrame containing ['Timestep', 'Mean_Entropy', 'Std_Error']
    - output_path: Optional; path to save the plot (without extension).
    """
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set(style="whitegrid")

    # Define color palette
    color = sns.color_palette("viridis", 1)[0]

    # Plot mean entropy line
    plt.plot(
        aggregated_df['Timestep'],
        aggregated_df['Mean_Entropy'],
        label='Mean Joint Entropy',
        color=color,
        linewidth=2
    )

    # Plot standard error shading
    plt.fill_between(
        aggregated_df['Timestep'],
        aggregated_df['Mean_Entropy'] - aggregated_df['Std_Error'],
        aggregated_df['Mean_Entropy'] + aggregated_df['Std_Error'],
        color=color,
        alpha=0.3,
        label='Standard Error'
    )

    # Set labels and title
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Joint Entropy (Bits)', fontsize=14)
    plt.title('Average Joint Entropy Over Time with Standard Error', fontsize=16)

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(fontsize=12)

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

def plot_cmi_linegraphs(aggregated_df):
    """
    Plot CMI line graphs for all agent pairs with standard error shading (for repeats).
    Maximum 3 subplots per row.
    """
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
    plt.figure(figsize=(total_width, total_height), dpi=300)
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(num_rows, min(max_per_row, num_pairs), figsize=(total_width, total_height), sharey=True)

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair]
        

        # Plot the mean CMI over time for this agent pair
        ax.plot(pair_data['Timestep'], pair_data['mean_cmi'], label=f'Agent Pair {pair}', color=color_dict[pair], linewidth=2)
        
        # Plot the standard error shading if sem_cmi is available
        if 'sem_cmi' in pair_data.columns:
            if pair_data['sem_cmi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(pair_data['Timestep'], 
                                pair_data['mean_cmi'] - pair_data['sem_cmi'], 
                                pair_data['mean_cmi'] + pair_data['sem_cmi'], 
                                color=color_dict[pair], alpha=0.3)

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CMI (Bits)', fontsize=16)
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

def plot_cte_linegraphs(aggregated_df):
    """
    Plot CTE line graphs for all agent pairs with standard error shading (for repeats).
    """
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
    plt.figure(figsize=(total_width, total_height), dpi=300)
    sns.set(style="whitegrid")

    # Create subplots for each agent pair, distributed across multiple rows if needed
    fig, axes = plt.subplots(num_rows, min(max_per_row, num_pairs), figsize=(total_width, total_height), sharey=True)

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_pairs > 1 else [axes]  # Single row with multiple pairs or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_pairs)
    color_dict = dict(zip(agent_pairs, palette))

    for ax, pair in zip(axes, agent_pairs):
        pair_data = aggregated_df[aggregated_df['Pair'] == pair]
        
        # Plot the mean CTE over time for this agent pair
        ax.plot(pair_data['Timestep'], pair_data['mean_cte'], label=f'Agent Pair {pair}', color=color_dict[pair], linewidth=2)
        
        # Plot the standard error shading if sem_cte is available
        if 'sem_cte' in pair_data.columns:
            if pair_data['sem_cte'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(pair_data['Timestep'], 
                                pair_data['mean_cte'] - pair_data['sem_cte'], 
                                pair_data['mean_cte'] + pair_data['sem_cte'], 
                                color=color_dict[pair], alpha=0.3)

        # Set axis labels and title
        ax.set_title(f'Agent Pair {pair}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('CTE (Bits)', fontsize=16)
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
    plt.show()

def conditional_transfer_entropy(data, action_suffix='_action'):
    """
    Main function to calculate local CTE, plot the heatmap, or line graphs if repeats are present.
    """
    # Calculate CTE, either with or without repeats
    aggregated_df = calculate_cte_with_repeats(data, action_suffix)

    # Check if mean and sem_cte exist for plotting
    if 'mean_cte' in aggregated_df.columns:
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
            print(f"Repeat {repeat}: {num_timesteps} timesteps")

        # Combine all results into a single DataFrame
        combined_df = pd.concat(result_df_list, ignore_index=True)

        # Melt the DataFrame so that each repeat is handled separately
        melted_df = combined_df.melt(id_vars=['Timestep', 'Repeat'], var_name='Agent', value_name='Local PI')

        # Ensure 'Local PI' column is numeric after melting
        melted_df['Local PI'] = pd.to_numeric(melted_df['Local PI'], errors='coerce')

        # Drop any rows where 'Local PI' is NaN
        melted_df = melted_df.dropna(subset=['Local PI'])

        # Check how many repeats exist
        num_repeats = len(melted_df['Repeat'].unique())
        print(f"Unique repeats in the melted data: {num_repeats}")

        # Group by 'Agent' and 'Timestep' ONLY to average over repeats
        aggregated_df = melted_df.groupby(['Agent', 'Timestep']).agg(
            mean_pi=('Local PI', 'mean'),
            sem_pi=('Local PI', 'sem')
        ).reset_index()

        # Debug: Check the structure of the final aggregated_df
        print(f"Aggregated DataFrame (averaged over repeats, {len(aggregated_df)} timesteps):")
        print(aggregated_df.head())

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

def plot_pi_linegraphs(aggregated_df):
    """
    Plot Predictive Information line graphs for all agents with standard error shading (for repeats).
    """
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
    plt.figure(figsize=(total_width, total_height), dpi=300)
    sns.set(style="whitegrid")

    # Create subplots for each agent, distributed across multiple rows if needed
    fig, axes = plt.subplots(num_rows, min(max_per_row, num_agents), figsize=(total_width, total_height), sharey=True)

    # Flatten axes for easier indexing when there's more than one row
    if num_rows == 1:
        axes = axes.flatten() if num_agents > 1 else [axes]  # Single row with multiple agents or single plot
    else:
        axes = axes.flatten()  # Multi-row case

    # Define color palette
    palette = sns.color_palette("viridis", n_colors=num_agents)
    color_dict = dict(zip(agents, palette))

    for ax, agent in zip(axes, agents):
        agent_data = aggregated_df[aggregated_df['Agent'] == agent]

        # Plot the mean Predictive Information over time for this agent
        ax.plot(agent_data['Timestep'], agent_data['mean_pi'], label=f'Agent {agent}', color=color_dict[agent], linewidth=2)

        # Plot the standard error shading if sem_pi is available
        if 'sem_pi' in agent_data.columns:
            if agent_data['sem_pi'].sum() > 0:  # Ensure there's variation before shading
                ax.fill_between(agent_data['Timestep'], 
                                agent_data['mean_pi'] - agent_data['sem_pi'], 
                                agent_data['mean_pi'] + agent_data['sem_pi'], 
                                color=color_dict[agent], alpha=0.3)

        # Set axis labels and title
        ax.set_title(f'Agent {agent}', fontsize=18)
        ax.set_xlabel('Time', fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel('Predictive Information (Bits)', fontsize=16)
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