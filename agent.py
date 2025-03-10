'''
Factorised Active Inference Agent

Authors: Jaime Ruiz Serra, Patrick Sweeney, Mike Harré
Date: 2024-07
'''

import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from typing import Union

# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# PYTORCH_ENABLE_MPS_FALLBACK = 1 
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# torch.set_default_device(device)
torch.set_printoptions(precision=2)

EPSILON = torch.finfo().eps
TOLERANCE = 1e-5

class Agent:

    def __init__(
            self, 
            id=None, 
            game_matrix=None, 
            # Perception
            interoception:bool=False,
            inference_num_iterations:int=10, 
            inference_num_samples:int=100,
            inference_learning_rate:float=1e-2,
            # Planning/action selection
            policy_length:int=1,
            compute_novelty:bool=False,
            deterministic_actions:bool=False,
            # Precision
            dynamic_precision:bool=True,  
            beta_0:float=10.,
            beta_1:float=10.,
            # Learning
            A_prior:Union[torch.Tensor, float]=99,  # identity
            B_prior:Union[torch.Tensor, float]=0,   # uniform
            D_prior:Union[torch.Tensor, None]=None,
            E_prior:Union[torch.Tensor, None]=None,
            A_learning:bool=True,
            B_learning:bool=True,
            B_learning_rate:Union[float, None]=0.05,
            learn_every_t_steps:int=24,
            learning_offset:int=6,
            decay:float=0.5,
            A_BMR:Union[str, None]='identity',
            B_BMR:Union[str, None]='identity',
        ):
        '''Initialise an agent with the following parameters
        
        Args:
            - id (int): The identifier of the agent
            - game_matrix (torch.Tensor): The game matrix (payoffs) of the game
            - interoception (bool): Whether the agent can perceive its own hidden state
            - inference_num_iterations (int): The number of variational optimisation iterations
            - inference_num_samples (int): The number of variational optimisation samples
            - inference_learning_rate (float): The learning rate for variational optimisation
            - compute_novelty (bool): Whether to compute novelty
            - deterministic_actions (bool): Whether the agent takes deterministic actions
            - dynamic_precision (bool): Whether the agent updates the precision
            - beta_0 (float): The beta_0 hyperparameter
            - beta_1 (float): The beta_1 hyperparameter
            - decay (float): The decay rate
            - A_prior (Union[torch.Tensor, float]): The prior for the observation model. If float, the strength of softmax (0 -> uniform, inf -> identity)
            - B_prior (Union[torch.Tensor, float]): The prior for the transition model. If float, the strength of softmax (0 -> uniform, inf -> identity)
            - A_learning (bool): Whether the agent learns the observation model
            - B_learning (bool): Whether the agent learns the transition model
            - B_learning_rate (Union[float, NoneType]): The learning rate for the transition model
            - learn_every_t_steps (int): The length of the interval for learning
            - A_BMR (Union[str, NoneType]): The Bayesian Model Reduction method for the observation model. One of ['identity', 'uniform', None]
            - B_BMR (Union[str, NoneType]): The Bayesian Model Reduction method for the transition model. One of ['identity', 'uniform', None]
            - E_prior (Union[torch.Tensor, NoneType]): The prior for the habits
            - theta_prior (Union[torch.Tensor, NoneType]): The prior for the initial state
        '''

        self.id = id
        
        # Generative model hyperparameters -------------------------------------
        self.game_matrix = game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
        self.num_actions = num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        self.num_agents = num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model parameters ------------------------------------------
        if isinstance(A_prior, torch.Tensor):
            self.A_params = A_prior
        elif isinstance(A_prior, (int, float)):
            self.A_params = torch.stack([
                torch.softmax(A_prior * torch.eye(num_actions) + EPSILON, dim=-1)
                for _ in range(num_agents)])  # Identity observation model prior
        else:
            raise ValueError(f"Invalid A_prior: {A_prior}")
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)

        if isinstance(B_prior, torch.Tensor):
            self.B_params = B_prior
        elif isinstance(B_prior, (int, float)):
            self.B_params = torch.stack([
                torch.softmax(B_prior * torch.eye(num_actions) + EPSILON, dim=-1)
                for _ in range(num_actions) 
                for _ in range(num_agents)
            ]).reshape(num_agents, num_actions, num_actions, num_actions)  # shape: (funk): factor, action (u), next state, kurrent state
        else:
            raise ValueError(f"Invalid B_prior: {B_prior}")
        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)

        self.set_log_C(game_matrix)  # Log preference over observations (payoffs)
        self.theta = [torch.ones(num_actions) for _ in range(num_agents)] if D_prior is None else D_prior  # Dirichlet state prior
        self.D = self.q_s = torch.stack([Dirichlet(theta).mean for theta in self.theta])  # Categorical state prior (D)
        self.E = torch.ones(num_actions**policy_length) / num_actions if E_prior is None else E_prior  # Habits 

        # Learning parameters --------------------------------------------------
        # Precision
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.beta_0 = beta_0  # beta in Gamma distribution (stays fixed)
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean)
        self.gamma = self.beta_1 / self.beta_0 # Current precision
        # Perception
        self.interoception = interoception  # Ego can perceive its own hidden state
        self.inference_num_iterations = inference_num_iterations
        self.inference_num_samples = inference_num_samples
        self.inference_learning_rate = inference_learning_rate
        # Planning
        self.compute_novelty = compute_novelty
        self.deterministic_actions = deterministic_actions
        self.policy_length = policy_length  # Length of the policy (number of actions to consider)
        # Learning
        self.learn_every_t_steps = learn_every_t_steps  # Learn every t steps
        self.learning_offset = learning_offset  # Random offset for learning
        self.current_random_offset_learning = random.randint(-self.learning_offset, self.learning_offset)  # Random offset for learning
        self.A_learning = A_learning  # Learn the observation model
        self.B_learning = B_learning  # Learn the transition model
        self.B_learning_rate = B_learning_rate
        self.A_BMR = A_BMR
        self.B_BMR = B_BMR
        self.decay = decay  # Forgetting rate for learning

        # Store blanket states history for learning ----------------------------
        self.o_history = []  # Observations (opponent actions) history
        self.q_s_history = []  # Beliefs (opponent policies) history
        self.u_history = []  # Actions (ego) history
        
        # Agents values (change at each time step) -----------------------------
        self.u = torch.multinomial(self.E, 1).item()  # Starting action randomly sampled according to habits

        self.VFE = [None] * num_agents  # Variational Free Energy for each agent (state factor)
        self.accuracy = [None] * num_agents
        self.complexity = [None] * num_agents
        self.energy = [None] * num_agents
        self.entropy = [None] * num_agents
        # q(s_j|u_i) Posterior predictive state distribution (for each factor) conditional on action u_i
        self.q_s_u = torch.empty((num_agents, num_actions, num_actions))  # shape (n_agents, n_actions, n_actions)

        self.EFE = torch.zeros(num_actions)  # Expected Free Energy (for each possible action)
        self.EFE_terms = torch.zeros((num_actions**policy_length, policy_length, 5))  # Store ambiguity, risk, salience, pragmatic value, novelty for the current time step
        self.q_u = torch.zeros(num_actions)  # q(u_i) Policy of ego (self) agent
        
        self.expected_EFE = None  # Expected EFE (under current policy q(u)): <G> = E_q(u)[ G[u] ]

    def set_log_C(self, game_matrix):
        '''Set the log preference over observations (payoffs)
        
        Args:
            game_matrix (torch.Tensor): The game matrix (payoffs) of the game, of shape (n_actions, ) * n_agents
        '''
        # Normalise log_C into a proper probability distribution
        # This is not strictly necessary, but it ensures the EFE values are positive
        flattened_C = torch.softmax(game_matrix.to(torch.float).flatten(), dim=0)
        C = flattened_C.view_as(game_matrix)
        self.log_C = torch.log(C + EPSILON)
        assert torch.isclose(torch.exp(self.log_C).sum(), torch.tensor(1.0), atol=TOLERANCE), (
            "The sum of exponentiated values of log C is not 1."
        )

    # ==========================================================================
    # Summaries
    # ==========================================================================

    def __repr__(self):
        return (f"Agent(id={self.id!r}, beta_1={self.beta_1}, decay={self.decay}, "
                f"gamma={self.gamma:.3f}, game_matrix_shape={self.game_matrix.shape}, "
                f"action={self.u}, VFE={self.VFE}, expected_EFE={self.expected_EFE})")

    def __str__(self):
        # Function to format a tensor for readable output
        def format_tensor(tensor):
            # Recursively formats the tensor depending on its number of dimensions
            if tensor.dim() == 1:
                # If 1D, format the elements directly
                return '\t'.join([f'{item:.2f}' for item in tensor])
            else:
                # If multi-dimensional, apply the formatting to each slice along the first dimension
                return '\n'.join([format_tensor(tensor_slice) for tensor_slice in tensor])

        # Format the game matrix and opponent model parameters
        formatted_game_matrix = format_tensor(self.log_C)

        # Format the state priors as percentages, with labels for each state factor
        formatted_state_priors = [
            f"State Factor {i + 1} Estimate: {', '.join([f'{prob * 100:.2f}%' for prob in state])}"
            for i, state in enumerate(self.q_s)
        ]

        total_vfe = sum(vfe for vfe in self.VFE if vfe is not None)
        total_accuracy = sum(acc for acc in self.accuracy if acc is not None)
        total_complexity = sum(comp for comp in self.complexity if comp is not None)
        total_energy = sum(en for en in self.energy if en is not None)
        total_entropy = sum(ent for ent in self.entropy if ent is not None)

        # Format the VFE values to 2 decimal places
        formatted_VFE = [f'{vfe:.2f}' if vfe is not None else 'None' for vfe in self.VFE]

        # Format the expected EFE values to 2 decimal places
        formatted_expected_EFE = [f'{efe:.2f}' if efe is not None else 'None' for efe in self.expected_EFE]

        # Format the EFE values per action to 2 decimal places
        formatted_EFE = [f'{efe:.2f}' for efe in self.EFE]

        # Format the additional per action metrics (ambiguity, risk, salience, pragmatic value)
        formatted_ambiguity = [f'{amb:.2f}' for amb in self.ambiguity]
        formatted_risk = [f'{risk:.2f}' for risk in self.risk]
        formatted_salience = [f'{sal:.2f}' for sal in self.salience]
        formatted_pragmatic_value = [f'{pv:.2f}' for pv in self.pragmatic_value]



        summary = (f"Agent ID: {self.id}\n"
                f"Gamma: {self.gamma:.3f}\n"
                f"Gamma Hyperparameters: beta=1.0, alpha={self.beta_1}\n"
                f"Opponent Learning Decay: {self.decay}\n"
                f"Current Action: {self.u}\n"
                f"Log C (Payoffs):\n{formatted_game_matrix}\n"
                f"{', '.join(formatted_state_priors)}\n"  # Join the state estimates on a single line
                f"Habits E: {', '.join([f'{prob * 100:.2f}%' for prob in self.E])}\n"

                f"Total Variational Free Energy (VFE): {total_vfe:.2f}, Per Factor: {', '.join(formatted_VFE)}\n"
                f"Accuracy: {total_accuracy:.2f}\n"
                f"Complexity: {total_complexity:.2f}\n"
                f"Energy: {total_energy:.2f}\n"
                f"Entropy: {total_entropy:.2f}\n"

                f"Expected EFE: {', '.join(formatted_expected_EFE)} (Per Action: {', '.join(formatted_EFE)})\n"
                f"Ambiguity: {', '.join(formatted_ambiguity)}\n"
                f"Risk: {', '.join(formatted_risk)}\n"
                f"Salience: {', '.join(formatted_salience)}\n"
                f"Pragmatic Value: {', '.join(formatted_pragmatic_value)}")

        return summary

    # ==========================================================================
    # Perception 
    # ==========================================================================

    def infer_state(self, o):
        '''
        Infer the hidden state of each agent (factor_idx) (i.e. the probability 
        distribution over actions of each agent) given the observation `o` 
        (i.e. the action taken by each agent).

        Employs self.A (observation model) and self.B (transition model),
        and updates the variational parameters self.theta for each agent
        through a Monte Carlo approximation of the variational free energy.

        Args:
            o (torch.Tensor): Observation tensor of shape (n_agents, n_actions), 
                i.e. one-hot encoding of actions for each agent
        
        Returns:
            s (torch.Tensor): Hidden state tensor of shape (n_agents, n_actions)
        '''
        if self.interoception:
            '''
            If proprioception is enabled, ego can perceive the true hidden state for the ego factor, 
            i.e. q(s_i) = q(u_i)
            and will have to infer the hidden states of the other agents ("theory of mind")
            '''
            # Ego factor
            factor_idx = 0
            self.q_s[factor_idx] = self.q_u.clone().detach()  # Ego factor is the previous timestep's policy
            self.theta[factor_idx] = torch.zeros_like(self.theta[factor_idx])  # PLACEHOLDER
            self.VFE[factor_idx] = 0  # PLACEHOLDER
            self.entropy[factor_idx] = 0  # PLACEHOLDER
            self.energy[factor_idx] = 0  # PLACEHOLDER
            self.accuracy[factor_idx] = 0  # PLACEHOLDER
            self.complexity[factor_idx] = 0  # PLACEHOLDER
            # Alter factors
            factors = range(1, len(self.q_s))
        else:
            '''
            Otherwise, ego has to infer all hidden states including its own, 
            i.e. "introspection" (towards self) and "theory of mind" (towards others)
            '''
            factors = range(len(self.q_s))
        
        # Iterate over (remaining) factors
        for factor_idx in factors:
            s_prev = self.q_s[factor_idx].clone().detach()  # State t-1
            assert torch.allclose(s_prev.sum(), torch.tensor(1.0)), "s_prev tensor does not sum to 1."
            log_prior = torch.log(self.B[factor_idx, self.u] @ s_prev + EPSILON)  # New prior is old posterior
            log_likelihood = torch.log(self.A[factor_idx].T @ o[factor_idx] + EPSILON)  # Likelihood of hidden states for a given observation

            variational_params = self.theta[factor_idx].clone().detach().requires_grad_(True)  # Variational Dirichlet distribution for each factor (agent)
            optimizer = torch.optim.Adam([variational_params], lr=self.inference_learning_rate)

            for _ in range(self.inference_num_iterations):
                optimizer.zero_grad()

                s_samples = Dirichlet(variational_params).rsample((self.inference_num_samples,))  # Variational dist samples
                log_s = torch.log(s_samples + EPSILON)  # Log variational dist samples
                vfe_samples = torch.sum(s_samples * (log_s - log_likelihood - log_prior), dim=-1) 
                VFE = vfe_samples.mean()

                VFE.backward()
                optimizer.step()
                variational_params.data.clamp_(min=1e-3)

            # Results of variational inference: variational posterior, variational parameters, VFE
            self.q_s[factor_idx] = Dirichlet(variational_params).mean.detach()  # Variational posterior
            self.theta[factor_idx] = variational_params.detach()  # Store variational parameters (for next timestep)
            self.VFE[factor_idx] = VFE.detach()
            
            # Compute additional metrics (for validation and plotting)
            self.entropy[factor_idx] = entropy = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
            self.energy[factor_idx] = energy = -torch.sum(s_samples * (log_prior + log_likelihood), dim=-1).mean().detach()
            self.accuracy[factor_idx] = accuracy = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
            self.complexity[factor_idx] = complexity = -torch.sum(s_samples * (log_prior - log_s), dim=-1).mean().detach()
            assert torch.allclose(VFE, energy - entropy, atol=TOLERANCE), "VFE != energy + entropy"
            assert torch.allclose(VFE, complexity - accuracy, atol=TOLERANCE), "VFE != complexity - accuracy"

        # Data collection (for learning and plotting)
        self.q_s_history.append(self.q_s)
        self.o_history.append(o)
        
        return self.q_s

    # ==========================================================================
    # Action 
    # ==========================================================================

    def compute_efe_OLD(self):
        '''
        Compute the Expected Free Energy (EFE) for each possible action
        
        Returns:
            EFE (torch.Tensor): Expected Free Energy for each possible action
        '''
        EFE = torch.zeros(self.num_actions)  # n-action length vector of zeros
        ambiguity = torch.zeros(self.num_actions)
        risk = torch.zeros(self.num_actions)
        salience = torch.zeros(self.num_actions)
        pragmatic_value = torch.zeros(self.num_actions)
        novelty = torch.zeros(self.num_actions)
        ppe_obs = torch.zeros(self.num_actions)
        
        # Posterior predictive state -------------------------------------------
        # (per factor 'f' and per possible action 'u')
        # E_{q(s)}[p(s' | s, u_i_hat)]
        self.q_s_u = torch.einsum(
            'funk,fk->fun',
            self.B,          # (f, u, n, k): factor, u (action), next (state), kurrent (state)
            self.q_s           # (f, k)
        )  # (f, u, n)

        # Posterior predictive observation -------------------------------------
        # (per factor 'f' and per possible action 'u')
        # E_{q(s'|u_i_hat)}[p(o | s)]
        self.q_o_u = torch.einsum(
            'fos,fus->fuo',
            self.A,         # (f, o, s)
            self.q_s_u      # (f, u, s)
        )  # (f, u, o)
        

        #I know #my PPO (INTEROCEPTION)
        # If ego was to take action u_i_hat, the observation o_i would be guaranteed
        # to be o_i = u_i_hat, so replace q(o_i | u_i_hat) = one_hot(u_i_hat) for this action
        # self.q_o_u[0] = torch.stack([
        #     F.one_hot(torch.tensor(u_i_hat), self.num_actions).to(torch.float)
        #     for u_i_hat in range(self.num_actions)
        # ])


        # EFE of each action ---------------------------------------------------
        for u_i_hat in range(self.num_actions):

            # Per-factor terms
            for factor_idx in range(self.num_agents):

                # Expected ambiguity term (per factor) -------------------------
                H = -torch.diag(self.A[factor_idx] @ torch.log(self.A[factor_idx] + EPSILON))  # Conditional (pseudo?) entropy (of the generated emissions matrix)
                assert H.ndimension() == 1, "H is not a 1-dimensional tensor"

                s_pred = self.q_s_u[factor_idx, u_i_hat]  # shape (2, )
                o_pred = self.q_o_u[factor_idx, u_i_hat]
                assert s_pred.ndimension() == 1, "s_pred is not a 1-dimensional tensor"
                assert torch.allclose(s_pred.sum(), torch.tensor(1.0), atol=TOLERANCE), f"s_pred does not sum to 1: {s_pred.sum().item()}"
 
                # ambiguity[u_i_hat] += (H @ s_pred) # Ambiguity (Friston)

                H_update = -torch.sum(self.A[factor_idx] * torch.log(self.A[factor_idx] + EPSILON), dim=1) #Actual entropy, not of diagonal
                ambiguity[u_i_hat] += torch.dot(s_pred, H_update)    # Ambiguity (Pat Update)

                ppe_obs[u_i_hat] += -torch.sum(o_pred * torch.log(o_pred)) 

                # Assert that ambiguity is greater than zero within tolerance
                assert torch.all(ambiguity > -TOLERANCE), f"Ambiguity contains values less than or equal to zero: {ambiguity[ambiguity <= 0]}"
         

 


            # Joint predictive observation posterior ---------------------------
            # q(o_i, o_j, o_k | u_i_hat)
            # Create the einsum subscripts string dynamically for n_agents
            # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
            einsum_str = (
                ','.join([chr(105 + i) for i in range(self.num_agents)]) 
                + '->' 
                + ''.join([chr(105 + i) for i in range(self.num_agents)])
            )

            # Compute joint probability q(o|u)
            q_o_joint_u = torch.einsum(
                einsum_str, 
                *[self.q_o_u[i, u_i_hat] for i in range(self.num_agents)]
            )
            assert q_o_joint_u.shape == (self.num_actions,) * (self.num_agents), (
                f"q_o_joint_u shape {q_o_joint_u.shape} != {(self.num_actions,) * (self.num_agents)}"
            )
            assert torch.allclose(q_o_joint_u.sum(), torch.tensor(1.0)), (
                f"q_o_joint_u sum {q_o_joint_u.sum()} != 1.0"
            )

            for agent_idx in range(self.num_agents):
                # Marginalize over all agents except the current one
                other_agents = tuple(i for i in range(self.num_agents) if i != agent_idx)
                
                # Compute marginal from joint by summing over other agents
                marginal_from_joint = torch.sum(q_o_joint_u, dim=other_agents)
                
                # Check that the marginal computed from the joint matches the original marginal
                assert torch.allclose(marginal_from_joint, self.q_o_u[agent_idx, u_i_hat], atol=TOLERANCE), (
                    f"Marginal of agent {agent_idx} from joint does not match: "
                    f"marginal_from_joint = {marginal_from_joint}, "
                    f"original marginal = {self.q_o_u[agent_idx, u_i_hat]}"
                )


            # Format log_C into a proper probability distribution
            flattened_C = torch.softmax(self.log_C.flatten(), dim=0)
            C = flattened_C.view_as(self.log_C)
            log_C = torch.log(C)
            assert torch.isclose(torch.exp(log_C).sum(), torch.tensor(1.0), atol=TOLERANCE), (
                "The sum of exponentiated values of log C is not 1."
            )

            ###########

            # C = torch.exp(self.log_C)

            ##########

            # Risk term (KL divergence)
            risk[u_i_hat] = torch.tensordot(
                q_o_joint_u,
                torch.log(q_o_joint_u + EPSILON) - log_C,
                dims=self.num_agents
            )

            # Pragmatic value term (Negative cross entropy or Expected Reward)
            pragmatic_value[u_i_hat] = torch.tensordot(
                q_o_joint_u,
                torch.log(C),
                dims=self.num_agents
            )
        
            # Novelty ----------------------------------------------------------
            if self.compute_novelty:
                novelty[u_i_hat] += self.compute_A_novelty(u_i_hat)
                # TODO: B novelty?

        EFE = ambiguity + risk - novelty
        assert not torch.any(torch.isnan(EFE)), f"EFE has NaN: {EFE}"

        salience = ppe_obs - ambiguity

        # Assert that the salience term >= 0
        if not torch.all(salience >= -TOLERANCE):
            invalid_values = salience[salience < 0]
            raise AssertionError(f"Salience term is not >= 0. Invalid values: {invalid_values}")


        EFE1 =  -salience - pragmatic_value
        EFE2 = risk + ambiguity
        

        ########### MISC PRINTS FOR CHECKS (can be deleted)
        
        # # Print EFE1 with rounded values
        # print(f'EFE 1: {[round(val.item(), 2) for val in EFE1]}')
        # print(f'Pragmatic value: {[round(val.item(), 2) for val in pragmatic_value]}')
        # print(f'Salience: {[round(val.item(), 2) for val in salience]}')
        # print()

        # print(f'Post. Pred Entropy: {[round(val.item(), 2) for val in ppe_obs]}')
        # print()

        # # Print EFE2 with rounded values
        # print(f'EFE 2: {[round(val.item(), 2) for val in EFE2]}')
        # print(f'Risk: {[round(val.item(), 2) for val in risk]}')
        # print(f'Ambiguity: {[round(val.item(), 2) for val in ambiguity]}')
        # print()

        # Element-wise percentage difference calculation
        percentage_diff = (torch.abs(EFE1 - EFE2) / ((EFE1 + EFE2) / 2)) * 100

        # Compute the mean percentage difference
        avg_percentage_diff = percentage_diff.mean().item()
        # print(f'Diff: {avg_percentage_diff:.1f}%')

        # Assert that the two tensors are close, and print the average percentage difference if they aren't
        assert torch.allclose(EFE1, EFE2, rtol = TOLERANCE, atol = TOLERANCE), \
            f"""Assertion failed — EFE's don't add up:
            -Salience - Pragmatic Value = {EFE1}
            Risk + Ambiguity = {EFE2}
            Average percentage difference = {avg_percentage_diff:.1f}%"""
        
        # print('================')

        # Data collection ------------------------------------------------------
        self.EFE = EFE
        self.ambiguity = ambiguity
        self.risk = risk
        self.salience = salience
        self.pragmatic_value = pragmatic_value
        self.novelty = novelty
        
        return EFE

    def compute_efe(self, u, q_s_u, A, log_C):
        '''
        Compute the Expected Free Energy (EFE) of a given action
        
        Args:
            u (int): action
            q_s_u (torch.Tensor): variational posterior over states given action is u
            A (torch.Tensor): observation likelihood model
            log_C (torch.Tensor): log preference over observations

        Returns:
            EFE (torch.Tensor): Expected Free Energy for a given action
        '''
        EFE = 0 
        ambiguity = 0 
        risk = 0 
        salience = 0 
        pragmatic_value = 0 
        ppo_entropy = 0
        novelty = 0 
        
        # Predictive observation posterior -------------------------------------
        # (per factor 'f' and per possible action 'u')
        # E_{q(s'|u)}[p(o | s)]
        q_o_u = torch.einsum(
            'fos,fs->fo',
            A,         # (f, o, s)
            q_s_u      # (f, s)
        )              # (f, o)
        
        # If ego was to take action u, the observation o_i would be guaranteed
        # to be o_i = u, so replace q(o_i | u) = one_hot(u) for this action
        q_o_u[0] = F.one_hot(u, self.num_actions).to(torch.float)

        # EFE computation -------------------------------------------------------

        # Per-factor terms
        for factor_idx in range(self.num_agents):

            # Expected ambiguity term (per factor) -------------------------
            s_pred = q_s_u[factor_idx]  # shape (2, )
            assert s_pred.ndimension() == 1, "s_pred is not a 1-dimensional tensor"
            assert torch.allclose(
                    s_pred.sum(), 
                    torch.tensor(1.0), 
                    atol=TOLERANCE), (
                        f"s_pred does not sum to 1: {s_pred.sum().item()}"
                    )
            
            # Ambiguity as per Parr et al. (2022)
            H = -torch.diag(A[factor_idx].T @ torch.log(A[factor_idx] + EPSILON))
            assert H.ndimension() == 1, "H is not a 1-dimensional tensor"
            ambiguity += (H @ s_pred) # Ambiguity is conditional entropy of emissions
            assert torch.all(ambiguity > -TOLERANCE), (
                f"Ambiguity contains values less than or equal to zero: "
                f"{ambiguity[ambiguity <= 0]}"
            )
            # Convert any (-TOLERANCE < ambiguity < 0) values to 0
            ambiguity = torch.clip(ambiguity, min=0.0, max=None)
            
            o_pred = q_o_u[factor_idx]  # shape (2, )
            o_pred = o_pred[o_pred > 0]  # Remove zero values
            ppo_entropy += -torch.sum(o_pred * torch.log(o_pred)) # Entropy of predictive posterior observation

        # Joint predictive observation posterior ---------------------------
        # q(o_i, o_j, o_k | u)
        # Create the einsum subscripts string dynamically for n_agents
        # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
        einsum_str = (
            ','.join([chr(105 + i) for i in range(self.num_agents)]) 
            + '->' 
            + ''.join([chr(105 + i) for i in range(self.num_agents)])
        )
        q_o_joint_u = torch.einsum(
            einsum_str, 
            *[q_o_u[i] for i in range(self.num_agents)]
        )
        # assert q_o_joint_u.shape == (self.num_actions, ) * (self.num_agents), (
        #     f"q_o_joint_u shape {q_o_joint_u.shape} != {(self.num_actions, ) * (self.num_agents)}"
        # )
        assert torch.allclose(q_o_joint_u.sum(), torch.tensor(1.0)), (
            f"q_o_joint_u sum {q_o_joint_u.sum()} != 1.0"
        )

        # Risk term (joint) ------------------------------------------------
        # i.e. KL[q(o|u) || p*(o)]
    
        risk += torch.tensordot(
            q_o_joint_u,
            (torch.log(q_o_joint_u + EPSILON) - log_C),
            dims=self.num_agents
        )

        # Pragmatic value term (Negative cross entropy or Expected Reward)
        pragmatic_value += torch.tensordot(
            q_o_joint_u,
            log_C,
            dims=self.num_agents
        )

        # Novelty ----------------------------------------------------------
        if self.compute_novelty:
            if self.A_learning:
                novelty += self.compute_A_novelty(q_s_u, q_o_u)
            if self.B_learning:
                novelty += self.compute_B_novelty(self.q_s, q_s_u, u) 
        # EFE value checks -----------------------------------------------------
        salience = ppo_entropy - ambiguity
        # assert salience >= -TOLERANCE, f"Salience term is not >= 0: {salience}, {ambiguity}, {ppo_entropy}"
        # if not torch.all(salience >= -TOLERANCE):
        #     invalid_values = salience[salience < 0]
        #     raise AssertionError(f"Salience term is not >= 0. Invalid values: {invalid_values}")
        EFE1 = - pragmatic_value - salience - novelty
        EFE2 = risk + ambiguity - novelty

        # Assert that the two tensors are close, and print the average percentage difference if they aren't
        percentage_diff = (torch.abs(EFE1 - EFE2) / ((EFE1 + EFE2) / 2)) * 100  # Element-wise percentage difference calculation
        avg_percentage_diff = percentage_diff.mean().item()  # Compute the mean percentage difference
        assert torch.allclose(EFE1, EFE2, rtol = TOLERANCE, atol = TOLERANCE), \
            f"""Assertion failed — EFE's don't add up:
            -Salience - Pragmatic Value = {EFE1}
            Risk + Ambiguity = {EFE2}
            Average percentage difference = {avg_percentage_diff:.1f}%"""

        EFE = -pragmatic_value - (salience + novelty)

        return EFE.unsqueeze(0), torch.tensor((ambiguity, risk, salience, pragmatic_value, novelty))
    
    def compute_A_novelty_ALGEBRAIC(self, q_s_u, u):
        '''
        Compute the novelty of the likelihood model A for action u_i
        '''
        novelty = 0
        # Add a small constant to A_params to avoid W blowing up
        A_params = self.A_params.clone().detach() + 0.5  # FIXME: softmax??
        # A_params = self.A.clone().detach()
        # Da Costa et al. (2020; Eq. D.17)
        W = 0.5 * (1/A_params - 1/A_params.sum(dim=1, keepdim=True))
        for factor_idx in range(self.num_agents):
            s_pred = q_s_u[factor_idx]
            novelty += torch.dot(
                self.A[factor_idx] @ s_pred, 
                W[factor_idx] @ s_pred)
        return novelty
    
    def compute_A_novelty(self, q_s_u, q_o_u):

        # A' = A + q_o_u x q_s_u
        A_prime_params = self.A_params + torch.einsum(
            'fs,fo->fos',  # (f, o, s): factor, observation, state
            q_s_u,         # (f, s) 
            q_o_u          # (f, o)
        )
        A_prime = A_prime_params / A_prime_params.sum(dim=1, keepdim=True)
        # TODO: account for BMR?

        # KL divergence D[ A' || A ] for each factor
        kl_div = (
            A_prime * torch.log(A_prime / (self.A + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))

        return kl_div.sum()  # over factors
    
    def compute_B_novelty(self, q_s, q_s_u, u):

        # B'[u] = B[u] + q_s_u x q_s
        outer_product = torch.einsum(
            'fn,fk->fnk',  # (f, n, k): factor, next (state), kurrent (state)
            q_s_u,         # (f, n) 
            q_s            # (f, k)
        )
        B_prime_params = self.B_params[:, u].squeeze() + self.B_learning_rate * outer_product
        B_prime = B_prime_params / B_prime_params.sum(dim=1, keepdim=True)

        # KL divergence D[ B'[u] || B[u] ] for each factor
        kl_div = (
            B_prime * torch.log(B_prime / (self.B[:, u].squeeze() + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))

        return kl_div.sum()  # over factors

    
    def collect_policies(
            self,
            node, 
            q_s,
            policy_EFEs=None,
            policy_EFE_terms=None,
            current_policy=None,
        ):
        '''Function to traverse the tree and collect policies (as tensors)'''
        
        # Root node case
        if node.u is None:
            node.q_s_u = q_s
            current_policy = []
            policy_EFEs = []
            new_policy_EFEs = policy_EFEs
        # Other nodes
        else:
            # Compute q(s|u) and EFE(u) for the current node
            node.q_s_u = torch.einsum(
                'funk,fk->fun',
                self.B,             # (f, u, n, k): factor, u (action), next (state), kurrent (state)
                q_s                 # (f, k)
            )[:, node.u].squeeze()  # (f, u, n) -> (f, n)
        
            node.EFE_u, node.EFE_terms = self.compute_efe(node.u, node.q_s_u, self.A, self.log_C)
            new_policy_EFEs = policy_EFEs + [node.EFE_u]  # EFEs collected top-down
            # EFE terms collected top-down
            if policy_EFE_terms is None:
                policy_EFE_terms = node.EFE_terms.unsqueeze(0)
            else:
                policy_EFE_terms = torch.vstack((policy_EFE_terms, node.EFE_terms.unsqueeze(0)))
        
        # Base case (leaf node)
        if not node.children:
            return [torch.cat(new_policy_EFEs)], policy_EFE_terms, [torch.cat(current_policy)]

        # Recursive case
        EFEs = []
        EFE_terms = []
        policies = []
        for child in node.children:
            new_policy = current_policy + [child.u]  # Policies collected bottom-up
            subtree_EFEs, subtree_EFE_terms, sub_policy = self.collect_policies(
                child, node.q_s_u, new_policy_EFEs, policy_EFE_terms, new_policy)
            EFEs.extend(subtree_EFEs)
            EFE_terms.extend(subtree_EFE_terms)
            policies.extend(sub_policy)

        return torch.vstack(EFEs), torch.vstack(EFE_terms), torch.vstack(policies)

    def select_action(self):
        
        # Build the policy tree and "collect" policies
        root = build_policy_tree(torch.arange(self.num_actions), self.policy_length)
        EFEs, EFE_terms, policies = self.collect_policies(
            root, 
            q_s=self.q_s
        )

        EFE_policies = EFEs.sum(dim=1)
        q_u = torch.softmax(
            torch.log(self.E) - self.gamma * EFE_policies, 
            dim=0
        )

        # EFE = self.compute_efe()
        # q_u = torch.softmax(torch.log(self.E) - self.gamma * EFE, dim=0)
        assert torch.allclose(q_u.sum(), torch.tensor(1.0)), (
            "q_u policy tensor does not sum to 1.",
            f"q_u: {q_u}",
            f"q_u.sum(): {q_u.sum()}",
            f"EFE: {EFE_policies}"
        )
        # Data collection ------------------------------------------------------
        self.EFE = EFE_policies
        self.EFE_terms = EFE_terms.reshape(*EFEs.shape, -1)
        self.q_u = q_u

        # Select action
        policy_idx = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u = policies[policy_idx][0].item()
        # self.u = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u_history.append(self.u)

        if self.dynamic_precision:
            self.update_precision(EFE_policies, q_u)

        return self.u

    def update_precision(self, EFE, q_u):
        # Compute the expected EFE as a scalar value
        self.expected_EFE = torch.dot(q_u, EFE).item()
        
        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.beta_1 / (self.beta_0 - self.expected_EFE)
        
        return self.gamma

    # ==========================================================================
    # Learning
    # ==========================================================================

    def learn(self):

        # Learn every t steps
        if len(self.u_history) == (self.learn_every_t_steps + self.current_random_offset_learning):

            # Convert history to tensors
            self.q_s_history = torch.stack(self.q_s_history)  # Shape: (T, n_agents, n_actions)
            self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
            self.u_history = torch.tensor(self.u_history)  # Shape: (T, )
        
            # Learn
            if self.A_learning:
                self.learn_A()
            if self.B_learning:
                self.learn_B()

            # Reset history
            self.q_s_history = []
            self.o_history = []
            self.u_history = []
            self.current_random_offset_learning = random.randint(-self.learning_offset, self.learning_offset)  # Renew random offset

    def learn_A(self):

        # Perform the row-wise outer product
        outer_products = torch.einsum(  # Compute outer products
            'tfs,tfo->tfos',   # t (time), f (factor), s (state), o (observation)
            self.q_s_history, 
            self.o_history
        )  # Shape: (T, n_agents, n_actions, n_actions)

        # Posterior parameters
        delta_params = outer_products.mean(dim=0)  # Shape: (n_agents, n_actions, n_actions)
        A_posterior_params = self.A_params + delta_params  # Shape: (n_agents, n_actions, n_actions)

        # Bayesian Model Reduction ---------------------------------------------
        if self.A_BMR:

            if self.A_BMR == 'identity':
                A_reduced = torch.stack([
                    torch.eye(self.num_actions)
                    for _ in range(self.A.shape[0])
                ]).view(*self.A.shape)

                MIXTURE = 0.7
                A_reduced = (
                    MIXTURE * self.A_params 
                    + (1 - MIXTURE) * A_reduced
                )
                A_reduced = A_reduced / A_reduced.sum(dim=1, keepdim=True)

                # # Update model parameters if they reduce the free energy
                # self.delta_F = []
                # for factor_idx in range(len(self.q_s)):
                #     delta_F = delta_free_energy(
                #         A_posterior_params[factor_idx].flatten(), 
                #         self.A_params[factor_idx].flatten(), 
                #         A_reduced_params[factor_idx].flatten()
                #     )
                #     self.delta_F.append(delta_F)  # Data collection
                #     if delta_F < 0:
                #         # Reduced model is preferred -> replace full model with reduced model
                #         self.A_params[factor_idx] = A_reduced_params[factor_idx]
                #         # print(f"Factor {factor_idx}: Reduced model is preferred.")
                #         # print(f"Delta F: {delta_F}")
                #         # print(f"Reduced: {A_reduced_params[factor_idx]}")
                #         # print(f"Posterior: {A_posterior_params[factor_idx]}")
                #     else:
                #         # Full model is preferred -> update posterior
                #         self.A_params[factor_idx] = A_posterior_params[factor_idx]
                    # print(delta_F, self.A_params[factor_idx].flatten(), A_posterior_params[factor_idx], A_reduced_params[factor_idx].flatten())
            elif self.A_BMR == 'uniform':
                A_reduced = torch.ones_like(self.A_params)
                raise NotImplementedError("Uniform BMR for A disabled.")
            elif self.A_BMR == 'softmax':
                STRENGTH = self.decay
                A_reduced = F.softmax(
                    STRENGTH * self.A_params,
                    dim=-1
                )
            else:
                raise ValueError(f"Invalid A_BMR: {self.A_BMR}")
            
            A_prior = self.A_params / self.A_params.sum(dim=1, keepdim=True)
            A_posterior = A_posterior_params / A_posterior_params.sum(dim=1, keepdim=True)

            # Da Costa et al. (2020; Eq. 26)
            # log E_{q(A)}[ p_reduced(A) / p(A) ]
            evidence_differences = torch.log(
                torch.einsum(
                    'fos,fos->f',
                    A_posterior,
                    A_reduced / A_prior
                )
            )

            for factor_idx in range(self.num_agents):
                if evidence_differences[factor_idx] > 0:
                    self.A_params[factor_idx] = A_reduced[factor_idx]
                else:
                    self.A_params[factor_idx] = A_posterior_params[factor_idx]

        else:
            self.A_params = A_posterior_params
        
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)  # Shape: (n_agents, n_actions, n_actions)

    def learn_B(self):

        # Shift arrays for prev and next
        s_prev = self.q_s_history[:-1]  # Shape: (T-1, n_agents, n_actions)
        s_next = self.q_s_history[1:]  # Shape: (T-1, n_agents, n_actions)
        
        outer_products = torch.einsum(  # Compute outer products
            'tfn,tfk->tfnk',   # t (time), f (factor), n (next), k (kurrent)
            s_next, 
            s_prev
        )  # Shape: (T-1, n_agents, n_actions, n_actions)
        T = outer_products.shape[0]

        # Update parameters for every transition (s, u, s') in the history
        B_posterior_params = self.B_params.clone()
        LEARNING_RATE = self.B_learning_rate if self.B_learning_rate is not None else 1/T
        for t in range(outer_products.shape[0]):
            # Likelihood parameters update
            delta_params = outer_products[t]  # Shape: (n_agents, n_actions, n_actions)
            u_it = self.u_history[t].item()   # Action u_i at time t
            B_posterior_params[:, u_it] = self.B_params[:, u_it] + LEARNING_RATE * delta_params

        # Bayesian Model Reduction ---------------------------------------------
        if self.B_BMR:
            
            if self.B_BMR == 'identity':
                B_reduced = torch.stack([
                    torch.eye(self.num_actions)
                    for action_idx in range(self.B.shape[1])
                    for factor_idx in range(self.B.shape[0])
                ]).view(*self.B.shape)
                raise NotImplementedError("Identity BMR for B disabled.")
                # mixture = self.decay
                # B_reduced_params = (
                #     mixture * self.B_params 
                #     + (1 - mixture) * B_reduced
                # )

                # # Update model parameters if they reduce the free energy
                # self.delta_F = []
                # for factor_idx in range(len(self.q_s)):
                #     delta_F = delta_free_energy(
                #         B_posterior_params[factor_idx].flatten(), 
                #         self.B_params[factor_idx].flatten(), 
                #         B_reduced_params[factor_idx].flatten()
                #     )
                #     self.delta_F.append(delta_F)  # Data collection
                #     if delta_F < 0:
                #         # Reduced model is preferred -> replace full model with reduced model
                #         self.B_params[factor_idx] = B_reduced_params[factor_idx]
                #     else:
                #         # Full model is preferred -> update posterior
                #         self.B_params[factor_idx] = B_posterior_params[factor_idx]
            elif self.B_BMR == 'uniform':
                B_reduced = torch.ones_like(self.B_params)
                raise NotImplementedError("Uniform BMR for B disabled.")
            elif self.B_BMR == 'softmax':
                STRENGTH = self.decay
                B_reduced = F.softmax(
                    STRENGTH * self.B_params.view(self.num_agents, self.num_actions, -1), 
                    dim=-1
                ).view_as(self.B_params)  # Shape: (n_agents, n_actions, n_actions, n_actions)
                B_prior = self.B_params / self.B_params.sum(dim=2, keepdim=True)
                B_posterior = B_posterior_params / B_posterior_params.sum(dim=2, keepdim=True)

                # Da Costa et al. (2020; Eq. 26)
                # log E_{q(B)}[ p_reduced(B) / p(B) ]
                evidence_differences = torch.log(
                    torch.einsum(
                        'funk,funk->fu',
                        B_posterior,
                        B_reduced / B_prior
                    )
                )

                for factor_idx in range(self.num_agents):
                    for action_idx in range(self.num_actions):
                        if evidence_differences[factor_idx, action_idx] > 0:
                            self.B_params[factor_idx, action_idx] = B_reduced[factor_idx, action_idx]
                            # print('Reduced', factor_idx, action_idx, B_reduced[factor_idx, action_idx])
                        else:
                            self.B_params[factor_idx, action_idx] = B_posterior_params[factor_idx, action_idx]

            else:
                raise ValueError(f"Invalid B_BMR: {self.B_BMR}")

        else:
            self.B_params = B_posterior_params

        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)


# ==============================================================================
# Helper functions
# ==============================================================================

def multivariate_beta(alpha):
    """
    Compute the multivariate Beta function B(alpha).
    Args:
    - alpha (torch.Tensor): 1D tensor of concentration parameters alpha (shape: K)
    Returns:
    - beta_value (torch.Tensor): The computed multivariate Beta function value
    """
    gamma_sum = torch.lgamma(alpha.sum())  # log(Gamma(sum(alpha)))
    gamma_individual = torch.lgamma(alpha).sum()  # sum(log(Gamma(alpha_i)))
    
    return torch.exp(gamma_sum - gamma_individual)

def delta_free_energy(a_posterior, a_prior, a_reduced):
    """
    Compute the change in free energy (ΔF) using Bayesian Model Reduction.
    Args:
    - a_prior (torch.Tensor): 1D tensor of prior concentration parameters alpha (shape: K)
    - a_posterior (torch.Tensor): 1D tensor of posterior concentration parameters alpha (shape: K)
    - a_reduced (torch.Tensor): 1D tensor of reduced concentration parameters alpha (shape: K)
    Returns:
    - delta_F (torch.Tensor): The change in free energy ΔF
    """
    B_a_posterior = multivariate_beta(a_posterior)
    B_a_prior = multivariate_beta(a_prior)
    B_a_reduced = multivariate_beta(a_reduced)
    B_diff = multivariate_beta(a_posterior + a_reduced - a_prior)
    
    delta_F = torch.log(B_a_posterior) + torch.log(B_a_reduced) - torch.log(B_a_prior) - torch.log(B_diff)
    
    return delta_F

# ==============================================================================
# Policy tree
# ==============================================================================

class TreeNode:
    def __init__(self, action=None, depth=0):
        self.u = action  # Tensor or action at this node
        self.EFE_u = torch.tensor(0)
        self.children = []  # List to hold child nodes
        self.depth = depth  # Depth of the node

    def add_child(self, action):
        # Add a child node with the given action (as a tensor)
        child = TreeNode(action, self.depth + 1)
        self.children.append(child)
        return child
    
    def __repr__(self):
        return f"TreeNode(action={self.u}, depth={self.depth}, children={len(self.children)})"
    

def build_policy_tree(action_space, max_depth, node=None):
    '''Recursive function to build the tree'''

    if node is None:
        node = TreeNode()

    # Base case
    if node.depth == max_depth:
        return node

    # Add a child for each action in the action space 
    # (recursively until the max depth is reached)
    for action in action_space:
        child = node.add_child(torch.tensor([action]))
        build_policy_tree(action_space, max_depth, node=child)

    return node

