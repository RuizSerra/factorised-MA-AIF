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

# ==============================================================================
# 3 Player, 2 action games
# ==============================================================================

prisoners_dilemma_3player = torch.tensor([
    
    # u_i = d ------------
    [[2., 1.],   # Player 1: D, Player 2: D, Player 3: (D / C) -> (2 / 1)
     [1., 1.]],  # Player 1: D, Player 2: C, Player 3: (D / C) -> (1 / 1)
    
    # u_i = c ------------
    [[4., 4.],   # Player 1: C, Player 2: D, Player 3: (D / C) -> (4 / 4)
     [4., 3.]]   # Player 1: C, Player 2: C, Player 3: (D / C) -> (4 / 3)
])

# Chicken game: payoffs for Player 1
# Index order: [Player 1's action, Player 2's action, Player 3's action]
chicken_3player = torch.tensor([
    
    # u_i = d ------------
    [[-1.,  1.],  # Player 1: D, Player 2: D, Player 3: (D / C) -> (-1 / 1)
     [ 1.,  1.]], # Player 1: D, Player 2: C, Player 3: (D / C) -> (1 / 1)

    # u_i = c ------------ 
    [[ 0.,  0.],  # Player 1: C, Player 2: D, Player 3: (D / C) -> (0 / 0)
     [ 0.,  0.]]  # Player 1: C, Player 2: C, Player 3: (D / C) -> (0 / 0)
])

harmony_3player = torch.tensor([
    [[1., 0.],  # Player 1: D, Player 2: D, Player 3: (D / C) -> (1 / 0)
     [0., 0.]], # Player 1: D, Player 2: C, Player 3: (D / C) -> (0 / 0)
     
    [[0., 0.],  # Player 1: C, Player 2: D, Player 3: (D / C) -> (0 / 0)
     [0., 1.]]  # Player 1: C, Player 2: C, Player 3: (D / C) -> (0 / 1)
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


