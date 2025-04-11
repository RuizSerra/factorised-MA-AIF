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
EPSILON = torch.finfo(torch.float32).eps
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



# Attachment game (Buono et al)

# Define the parameters for the attachment game
v = 1.  # Value (comfort) if the child goes to the mother and the mother attends
s = 1.  # Stress if the child goes to the mother and is ignored
c = 0.5  # Cost for the mother attending to the child

# Compute the payoffs
# Format is: (Child's payoff, Mother's payoff)

go_attend = (v, v - c)          # Child goes to the mother, mother attends
go_ignore = (-s, -s)            # Child goes to the mother, mother ignores
dont_go_attend = (0, -c)        # Child doesn't go to the mother, mother attends
dont_go_ignore = (0, 0)         # Child doesn't go to the mother, mother ignores

# Create the payoff matrix for the 2x2 attachment game
# Child's actions: [Go, Don't Go]
# Mother's actions: [Attend, Ignore]

attachment_game_2player_child = torch.tensor([
    [go_attend[0], go_ignore[0]],       # Child's payoffs when [Attend, Ignore]
    [dont_go_attend[0], dont_go_ignore[0]]  # Child's payoffs when [Attend, Ignore]
])

attachment_game_2player_mother = torch.tensor([
    [go_attend[1], go_ignore[1]],       # Mother's payoffs when [Go, Don't Go]
    [dont_go_attend[1], dont_go_ignore[1]]  # Mother's payoffs when [Go, Don't Go]
])

# Optional: Add small epsilon to avoid zero values
EPSILON = 1e-6
attachment_game_2player_child += EPSILON
attachment_game_2player_mother += EPSILON



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
# cf. Harré (2018; Fig. 4) "Multi-agent Economics and the Emergence of Critical Markets"
T, R, P, S = 4., 3.1, 2., 1.  # R > (T+S)/2  http://gamut.stanford.edu/gamut.pdf
prisoners_dilemma_3player = torch.tensor([
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (4 / 4)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (4 / 3)
    
    [[ T,  T],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (2 / 1)
     [ T,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 1)
])

T, R, S, P = 4., 3., 2., 1.  
chicken_3player = torch.tensor([
    [[ R,  S],  # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / S)
     [ S,  S]], # Player 1: C, Player 2: D, Player 3: (C / D) -> (S / S)
    
    [[ T,  P],  # Player 1: D, Player 2: C, Player 3: (C / D) -> (T / P)
     [ P,  P]], # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])

R, T, S, P = 4., 2., 2., 1.
harmony_3player = torch.tensor([     
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (0 / 1)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (0 / 0)
    
    [[ T,  T],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (0 / 0)
     [ T,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (1 / 0)
])

R, T, P, S = 4., 3., 2., 1.
stag_hunt_3player_green = stag_hunt_3player_M2 = torch.tensor([  # M2: 2 players required to hunt a stag
    [[ R,  R],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / R)
     [ R,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (R / S)

    [[ T,  P],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (P / P)
     [ P,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])

stag_hunt_3player_red = stag_hunt_3player_M3 = torch.tensor([  # M3: 3 players required to hunt a stag
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / S)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (S / S)

    [[ T,  P],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (P / P)
     [ P,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])

stag_hunt_3player_penalty = torch.tensor([  # M2: 2 players required to hunt a stag
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / R)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (R / S)

    [[ P,  P],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (P / P)
     [ P,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])

R, P, T, S = 4., 3., 2., 1.
coordination_3player = torch.tensor([
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / S)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (S / S)

    [[ T,  P],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (T / P)
     [ P,  P]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])

T, S, R, P = 4., 3., 2., 1.
leader_3player = torch.tensor([
    [[ R,  S],   # Player 1: C, Player 2: C, Player 3: (C / D) -> (R / S)
     [ S,  S]],  # Player 1: C, Player 2: D, Player 3: (C / D) -> (S / S)

    [[ T,  P],   # Player 1: D, Player 2: C, Player 3: (C / D) -> (T / P)
     [ P,  T]],  # Player 1: D, Player 2: D, Player 3: (C / D) -> (P / P)
])


# # ALTERNATIVE STAG HUNT
# # Adapted from N-player Stag Hunt (Luo et al., 2021)
# a = 4.  # Payoff for hunting a stag
# r = 2.  # Payoff for hunting a rabbit
# M = 2   # Number of players required to hunt a stag
# stag_hunt_3player = torch.zeros(2, 2, 2)
# for joint_action in torch.cartesian_prod(*(torch.arange(2), ) * 3):
#     if joint_action[0] == 1:  # Ego defects (hunts a rabbit)
#         stag_hunt_3player[joint_action] = r
#     elif joint_action.sum() <= (3 - M):  # Ego cooperates, and there are enough stag hunters (incl. ego)
#         stag_hunt_3player[joint_action] = a
#     else:  # Ego cooperates, but not enough stag hunters
#         stag_hunt_3player[joint_action] = 1.

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
# 3 Player, 4 action games
# ==============================================================================

T, R, P, S = 4., 3.1, 2., 1.  # R > (T+S)/2  http://gamut.stanford.edu/gamut.pdf
prisoners_dilemma_3x4 = torch.tensor([
    [   # u_k = CC
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = CD
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = DC
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
    [   # u_k = DD
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
]).swapaxes(0, 1)

T, R, S, P = 4., 3., 2., 1.
chicken_3x4 = torch.tensor([
    [   # u_k = CC
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = CD
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = DC
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
    [   # u_k = DD
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
]).swapaxes(0, 1)

R, T, P, S = 4., 3., 2., 1.
stag_hunt_3x4 = torch.tensor([
    [   # u_k = CC
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = CD
        [R+R, R+R, S+R, S+R],   # u_i = CC
        [R+T, R+T, S+T, S+T],   # u_i = CD
        [T+R, T+R, P+R, P+R],   # u_i = DC
        [T+T, T+T, P+T, P+T],   # u_i = DD
    ],
    [   # u_k = DC
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
    [   # u_k = DD
        [R+S, R+S, S+S, S+S],   # u_i = CC
        [R+P, R+P, S+P, S+P],   # u_i = CD
        [T+S, T+S, P+S, P+S],   # u_i = DC
        [T+P, T+P, P+P, P+P],   # u_i = DD
    ],
]).swapaxes(0, 1)


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


