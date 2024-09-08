'''
Game payoff matrix definitions

All values must be (converted to) float

TODO: naming convention for n-players, m-actions, log, exp
TODO: general linting
'''

import torch

# Add a small CONSTANT to avoid log(0) issues
CONSTANT = 1
NEG_CONSTANT = -5
EPSILON = 1e-09
EPSILON = 1

# ==============================================================================
# 2 Player, 2 action games
# ==============================================================================

# Prisoner's dilemma (row player payoffs)
prisoners_dilemma_2player = torch.tensor(
    [[ 3.0,  1.0],      # R S
     [ 4.0,  2.0]]      # T P
) 

# Harmony game (row player payoffs)
harmony_2player = torch.tensor(     
    [[ 4.0,  3.0],      # R S
     [ 2.0,  1.0]]      # T P
)

# Stag Hunt game (row player payoffs)
stag_hunt_2player = torch.tensor(
    [[4.0, 1.0],      # R S
     [3.0, 2.0]]      # T P
)

# Chicken game (row player payoffs)
chicken_2player = torch.tensor(
    [[2.0, 3.0],      # Dare, Chicken
     [4.0, 1.0]]      # Chicken, Dare
)

matching_pennies_2player = torch.tensor(    # Matching pennies (row player payoffs)
    [[ 1.0, -1.0],      # H T               # NOTE: to get column player payoffs, 
     [-1.0,  1.0]]      # T H               #       G_c = -G_r
)                                           #       e.g., to load, game_matrix=[(-1)**i * matching_pennies_2player for i in range(num_players)][i]



#Fisherman's Dilemma (Bowles)
alpha = 1.0  # Temptation surplus: Set this to any desired value
u = 2.0  # Fallback position / reservation option / BATNA: Set this to any desired value
fishers_dilemma_2player = torch.tensor([
    [1.0, 0.0],         # Jay fishes 6 hours (Row 1)
    [1.0 + alpha, u]    # Jay fishes 8 hours (Row 2)
])
fishers_dilemma_2player += EPSILON

# Firm Survival Game (Bowles)
p1 = 0.5  # Probability firm succeeds if the employer invests and the worker does not work:  Set this to any desired value, where 1 > p1 > p2 > 0
p2 = 0.2  # # Probability firm succeeds if the employer does not invest and the worker does work: Set this to any desired value, where p1 > p2 > 0
firm_survival_game_2player = torch.tensor([
    [1.0, p2],         # Worker works (Row 1): Employer invests, Employer does not invest
    [p1, 0.0]       # Worker does not work (Row 2): Employer invests, Employer does not invest
])
firm_survival_game_2player  += EPSILON

# Invisible Hand Game (Bowles)
invisible_hand_game_2player_row = torch.tensor([
    [2, 4],  # Player 1 chooses Corn
    [5, 3]   # Player 1 chooses Tomatoes
])

invisible_hand_game_2player_column = torch.tensor([
    [4, 3],  # Player 2 chooses Corn
    [5, 2]   # Player 2 chooses Tomatoes
])


# Hawk-Dove Game (Bowles)
# Define the values of v and c
v = 10  # Example value for the resource
c = 2  # Example cost of conflict

# Compute the payoffs
a = (v - c) / 2  # Payoff for Hawk vs. Hawk
b = v            # Payoff for Hawk vs. Dove
c_payoff = 0     # Payoff for Dove vs. Hawk
d = v / 2        # Payoff for Dove vs. Dove

# Player 1's (row player's) payoff matrix
hawk_dove_2player = torch.tensor([
    [a, b],  # Hawk vs. [Hawk, Dove]
    [c_payoff, d]  # Dove vs. [Hawk, Dove]
])


hawk_dove_2player += EPSILON

# ==============================================================================
# 2 Player, 3 action games
# ==============================================================================

climbing_game_2player = torch.tensor([
    [   11, -30,   0],    # Climbing game, adjusted for positive values
    [  -30,    7,   6],
    [   0,    0,   5]
], dtype=torch.float32) + 31   # <--- Make non-negative

rockpaperscissors_2player = torch.tensor(
    [[2., 1., 3.],
     [3., 2., 1.],
     [1., 3., 2.]]
)

test_game_2player = torch.tensor(
    [[1., 1., 1.], 
     [1., 1., 1.],
     [10., 1., 1.]]
)


import torch
#Hawk Dove Bourgeois (Bowles)
v = 4  # Example value for the resource
c = 2  # Example cost of conflict

# Compute the payoffs
hawk_hawk = (v - c) / 2              # Hawk vs. Hawk
hawk_dove = v                        # Hawk vs. Dove
hawk_bourgeois = v / 2 + (v - c) / 4 # Hawk vs. Bourgeois

dove_hawk = 0                        # Dove vs. Hawk
dove_dove = v / 2                    # Dove vs. Dove
dove_bourgeois = v / 4               # Dove vs. Bourgeois

bourgeois_hawk = (v - c) / 4         # Bourgeois vs. Hawk
bourgeois_dove = v / 2 + v / 4       # Bourgeois vs. Dove
bourgeois_bourgeois = v / 2          # Bourgeois vs. Bourgeois


hawk_dove_bourgeois_2player = torch.tensor([
    [hawk_hawk, hawk_dove, hawk_bourgeois],       # Hawk vs. [Hawk, Dove, Bourgeois]
    [dove_hawk, dove_dove, dove_bourgeois],       # Dove vs. [Hawk, Dove, Bourgeois]
    [bourgeois_hawk, bourgeois_dove, bourgeois_bourgeois] # Bourgeois vs. [Hawk, Dove, Bourgeois]
])

hawk_dove_bourgeois_2player += EPSILON

# ==============================================================================
# 2 Player, 5 action games
# ==============================================================================

division_game_2player = torch.tensor([
    [[100, 0], [80, 20], [60, 40], [40, 60], [20, 80], [0, 100]],  # Player 1 claims 100
    [[80, 20], [60, 40], [40, 60], [20, 80], [0, 100], [100, 0]],  # Player 1 claims 80
    [[60, 40], [40, 60], [20, 80], [0, 100], [100, 0], [80, 20]],  # Player 1 claims 60
    [[40, 60], [20, 80], [0, 100], [100, 0], [80, 20], [60, 40]],  # Player 1 claims 40
    [[20, 80], [0, 100], [100, 0], [80, 20], [60, 40], [40, 60]],  # Player 1 claims 20
    [[0, 100], [100, 0], [80, 20], [60, 40], [40, 60], [20, 80]],  # Player 1 claims 0
])



# ==============================================================================
# 3 Player, 2 action games
# ==============================================================================

prisoners_dilemma_3player = torch.tensor([
    [[ 2., 1.],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (4 / 4)
     [ 1., 1.]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (4 / 3)
    
    [[ 4., 4.],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (2 / 1)
     [ 4., 3.]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 1)
])

chicken_3player = torch.tensor([
    [[ 2.,  2.],  # Player 1: C, Player 2: C, Player 3: (C / D) -> (0 / 0)
     [ 2.,  2.]], # Player 1: C, Player 2: D, Player 3: (C / D) -> (0 / 0)
    
    [[ 4.,  1.],  # Player 1: D, Player 2: C, Player 3: (C / D) -> (-1 / 1)
     [ 1.,  1.]], # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 1)
])

harmony_3player = torch.tensor([     
    [[ 4., 3.],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (0 / 1)
     [ 3., 3.]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (0 / 0)
    
    [[ 2., 2.],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (0 / 0)
     [ 2., 1.]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 0)
])

stag_hunt_3player = torch.tensor([     
    [[ 4., 1.],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (0 / 1)
     [ 1., 1.]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (0 / 0)
    
    [[ 2., 2.],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (0 / 0)
     [ 2., 2.]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 0)
])

# ==============================================================================
# 3 Player, 3 action games
# ==============================================================================

# Player 1 payoffs
climbing_game_3player = torch.tensor([
    [[11, -30, 0], 
     [7, 6, 5], 
     [0, 0, 5]],  

    [[-30, 7, 6], 
     [7, 6, 5], 
     [0, 0, 5]],

    [[0, 0, 5], 
     [7, 6, 5], 
     [0, 0, 5]]
]) + 31

rockpaperscissors_3player = torch.tensor([
    [[ 0,  1,  1],
     [-1,  0, -1],
     [-1,  1,  0]],

    [[ 1, -1,  1],
     [ 1,  0, -1],
     [-1, -1,  0]],

    [[ 1,  1, -1],
     [ 1, -1,  0],
     [ 0, -1,  1]]]) + 2

# ==============================================================================
# 4 Player, 2 action games
# ==============================================================================

prisoners_dilemma_4player = torch.tensor([
    # u_i = d --------------------------------------------
    [[[2., 1.], [1., 1.]],   # Player 1: D, Player 2: D, Player 3: (D / C), Player 4: (D / C) -> (2 / 1) / (1 / 1)
     [[1., 1.], [1., 1.]]],  # Player 1: D, Player 2: C, Player 3: (D / C), Player 4: (D / C) -> (1 / 1) / (1 / 1)
    
    # u_i = c --------------------------------------------
    [[[4., 4.], [4., 3.]],   # Player 1: C, Player 2: D, Player 3: (D / C), Player 4: (D / C) -> (4 / 4) / (4 / 3)
     [[4., 3.], [4., 3.]]]   # Player 1: C, Player 2: C, Player 3: (D / C), Player 4: (D / C) -> (4 / 3) / (4 / 3)
])


