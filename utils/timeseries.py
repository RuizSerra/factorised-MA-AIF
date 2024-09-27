'''
Functions to compute metrics from data

Author: Jaime Ruiz Serra
Date:   2024-09-19
'''

import numpy as np
from scipy.spatial import ConvexHull

# ==============================================================================
# Time series metrics
# ==============================================================================
def get_t_steady_state(trajectory, steady_state_threshold=0.02):
    '''Compute the time step at which the trajectory reaches the steady state

    Args:
        trajectory (np.array): The trajectory of the system in state space
        steady_state_threshold (float): The threshold for the distance to the final state

    '''
    x_final = trajectory[-1]
    t_steady = None
    for t in range(len(trajectory)-1, 0, -1):
        distance = np.linalg.norm(trajectory[t] - x_final)
        if distance > steady_state_threshold:
            t_steady = t
            break
    return t_steady

def get_distance_to_main_diagonal(point):
    '''Compute the distance from a point to the main diagonal of the unit cube

    The main diagonal of the unit cube is the line that connects the origin (0, 0, 0) to the point (1, 1, 1).

    Args:
        point (np.array): The point in 3D space

    Returns:
        float: The distance from the point to the main diagonal
    '''
    # Define points A and B that define the line
    A = np.array([0, 0, 0])
    B = np.array([1, 1, 1])
    d = B - A  # Direction vector of the line
    AP = point - A  # Compute vector AP (from A to P)
    cross_product = np.cross(AP, d)  # Compute the cross product of AP and the direction vector d
    distance = np.linalg.norm(cross_product) / np.linalg.norm(d)  # Compute the distance from point P to the line
    return distance

def get_t_bifurcation(trajectory, bifurcation_threshold=0.05):
    t_bifurcation = None
    for t in range(len(trajectory)-1, 0, -1):
        distance = get_distance_to_main_diagonal(trajectory[t])
        if distance <= bifurcation_threshold:
            t_bifurcation = t
            return t_bifurcation
        
# ==============================================================================
# Geometry metrics
# ==============================================================================
# Reference vertices
vertices = np.array([
    [0, 0, 0],  # 0C -- pink
    [1, 0, 0],  # 1C -- red
    [1, 1, 0],  # 2C -- orange
    [1, 1, 1]  # 3C -- green
])
hull = ConvexHull(vertices)

def is_outside_convex_hull(point, hull=hull):
    # Using the hull equations (inequalities) to check if the point is inside or outside
    return np.any(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] > 0)

# def process_trajectory(trajectory_raw):

#     num_agents = len(trajectory_raw[0])
#     trajectory_raw = trajectory_raw.copy()

#     trajectory = []
#     for point in trajectory_raw:
#         if point[0] < point[1]:  # Reflect about plane x=y
#             # point = np.stack([point[1], point[0], point[2]])
#             point[[0, 1]] = point[[1, 0]].copy()
#         if num_agents == 3:
#             if point[1] < point[2]:  # Reflect about plane y=z
#                 # point = np.stack([point[0], point[2], point[1]])
#                 point[[1, 2]] = point[[2, 1]].copy()
#             if point[0] < point[1]:  # Reflect about plane x=y
#                 # point = np.stack([point[1], point[0], point[2]])
#                 point[[0, 1]] = point[[1, 0]].copy()
        
#         # CHECKS
#         # if is_outside_convex_hull(hull, point):
#         #     color = 'red'
#         # else:
#         #     color = 'green'
#         # ax.scatter(*point, c=color, zorder=999)
        
#         trajectory.append(point)

#     return trajectory


# Compute all possible permuatations of the agents -----------------------------
from itertools import permutations

def permute_agents(
        trajectory, 
        t_check=None,
        perms=None,
    ):
    '''Permute the agents in the trajectory to match the reference vertices of the unit cube.

    Applied recursively until the appropriate permutation is reached.
    
    Args:
        trajectory (np.array): The trajectory of the system in state space
        t_check (int): The time step to check for the closest vertex (automatically computed if None)
        perms (list): The list of permutations of the agents (automatically generated if None)

    Returns:
        trajectory (np.array): The permuted trajectory
    '''

    num_players = trajectory.shape[-1]
    trajectory = trajectory.copy()  # Avoid modifying the original trajectory
    
    if perms is None:
        # Full list of possible permutations of the agents (initially)
        perms = list(permutations(range(num_players)))
    
    # Reference points: vertices of the unit (D-dimensional) cube where N-1 agents cooperate
    if num_players == 2:
        ref_points = np.array([
            [1., 0.],
            [0., 1.],
        ])
    elif num_players == 3:
        ref_points = np.array([
            [1., 1., 0.],
            [1., 0., 1.],
            [0., 1., 1.],
        ])

    if t_check is None:
        min_distance_to_each_ref_point = [
            # Out of the whole trajectory, what is the minimum distance to ref_point?
            np.linalg.norm(trajectory - ref_point, axis=1).min()
            # Compute for each ref_point
            for ref_point in ref_points
        ]
        # The closest vertex is the one with the minimum distance
        closest_vertex_idx = np.argmin(min_distance_to_each_ref_point)
    else:
        # For the time step t_check, compute the closest vertex
        closest_vertex_idx = np.linalg.norm(
            ref_points - trajectory[t_check], 
            axis=1
        ).argmin()

    # If the closest vertex is not the first one, permute the agents
    if closest_vertex_idx != 0:
        last_perm = perms.pop()
        trajectory[:, list(perms[0])] = trajectory[:, list(last_perm)].copy()
        print(f'Permuting agents: {last_perm}')
        return permute_agents(
            trajectory, 
            t_check=t_check,
            perms=perms
        )
    else:
        return trajectory