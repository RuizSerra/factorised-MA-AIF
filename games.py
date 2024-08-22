'''
Game payoff matrix definitions

All values must be (converted to) float

TODO: naming convention for n-players, m-actions, log, exp
TODO: general linting
'''

import torch


# Add a small constant to avoid log(0) issues
constant = 1
neg_constant = -5

# ========================================
# 2 Player, 2 action games
# =======================================


prisoners_dilemma_2player = torch.tensor(     # Prisoner's dilemma (row player payoffs)
    [[ 3.0,  1.0],      # R S
     [ 4.0,  2.0]]      # T P
) 


harmony_2player = torch.tensor(     # Harmony game (row player payoffs)
    [[ 4.0,  3.0],      # R S
     [ 2.0,  1.0]]      # T P
)

stag_hunt_2player = torch.tensor(     # Stag Hunt game (row player payoffs)
    [[4.0, 1.0],      # R S
     [3.0, 2.0]]      # T P
)

chicken_2player = torch.tensor(     # Chicken game (row player payoffs)
    [[2.0, 3.0],      # Dare, Chicken
     [4.0, 1.0]]      # Chicken, Dare
)

# Compute the logarithm of each game matrix
prisoners_dilemma_2player_log = torch.log(prisoners_dilemma_2player + constant) 
harmony_2player_log = torch.log(harmony_2player + constant) 
stag_hunt_2player_log = torch.log(stag_hunt_2player + constant) 
chicken_2player_log = torch.log(chicken_2player + constant)

prisoners_dilemma_2player_exp = torch.exp(prisoners_dilemma_2player + neg_constant)
harmony_2player_exp = torch.exp(harmony_2player + neg_constant)
stag_hunt_2player_exp = torch.exp(stag_hunt_2player + neg_constant)
chicken_2player_exp = torch.exp(chicken_2player + neg_constant)


# ========================================
# 2 Player, 3 action games
# =======================================

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

climbing_game_2player_log = torch.log(climbing_game_2player + constant)
rockpaperscissors_2player_log = torch.log(rockpaperscissors_2player + constant)
test_game_2player_log = torch.log(test_game_2player + constant)

# ========================================
# 3 Player, 2 action games
# =======================================

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

# ========================================
# 3 Player, 3 action games
# =======================================


climbing_game_3player = torch.tensor([
    [[11, -30, 0], [7, 6, 5], [0, 0, 5]],   # Player 1 payoffs
    [[-30, 7, 6], [7, 6, 5], [0, 0, 5]],
    [[0, 0, 5], [7, 6, 5], [0, 0, 5]]
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



# Compute the logarithm of each game matrix without epsilon
prisoners_dilemma_3player_log = torch.log(prisoners_dilemma_3player + constant)
climbing_game_3player_log = torch.log(climbing_game_3player + constant)
rockpaperscissors_3player_log = torch.log(rockpaperscissors_3player + constant)