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
from torch.distributions.kl import kl_divergence
from typing import Union
from types import NoneType

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
            beta_0:float=1.,
            beta_1:float=10.,
            # Learning
            A_prior_type:str='uniform',
            B_prior_type:str='identity',
            E_prior:Union[torch.Tensor, NoneType]=None,
            theta_prior:Union[torch.Tensor, NoneType]=None,
            A_learning:bool=True,
            B_learning:bool=True,
            learn_every_t_steps:int=6,
            learning_offset:int=0,
            decay:float=0.5,
            A_BMR:Union[str, NoneType]='identity',
            B_BMR:Union[str, NoneType]='identity',
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
            - A_prior_type (str): The type of prior for the observation model. One of ['identity', 'uniform']
            - B_prior_type (str): The type of prior for the transition model. One of ['identity', 'uniform']
            - A_learning (bool): Whether the agent learns the observation model
            - B_learning (bool): Whether the agent learns the transition model
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
        self.theta = [torch.ones(num_actions) for _ in range(num_agents)] if theta_prior is None else theta_prior # Dirichlet state prior
        
        if A_prior_type == 'identity':
            self.A_params = torch.stack([torch.eye(num_actions) for _ in range(num_agents)]) + EPSILON  # Identity observation model prior
        elif A_prior_type == 'uniform':
            self.A_params = torch.ones((num_agents, num_actions, num_actions))  # Uniform observation model prior
        else:
            raise ValueError(f"Invalid A_prior_type: {A_prior_type}")
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)

        if B_prior_type == 'identity':
            self.B_params = torch.stack([
                torch.eye(num_actions) 
                for _ in range(num_actions) 
                for _ in range(num_agents)
            ]).reshape(num_agents, num_actions, num_actions, num_actions)
        elif B_prior_type == 'uniform':
            self.B_params = torch.ones((num_agents, num_actions, num_actions, num_actions))
        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)

        self.log_C = game_matrix.to(torch.float) # Payoffs for joint actions (in matrix form, from row player's perspective) (force to float)
        self.s = torch.stack([Dirichlet(theta).mean for theta in self.theta])  # Categorical state prior (D)
        self.E = torch.ones(num_actions**policy_length) / num_actions if E_prior is None else E_prior  # Habits 

        # Learning parameters --------------------------------------------------
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.beta_0 = beta_0  # beta in Gamma distribution (stays fixed)
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean)
        self.gamma = self.beta_1 / self.beta_0 # Current precision
        self.interoception = interoception  # Ego can perceive its own hidden state
        self.inference_num_iterations = inference_num_iterations
        self.inference_num_samples = inference_num_samples
        self.inference_learning_rate = inference_learning_rate
        self.learn_every_t_steps = learn_every_t_steps  # Learn every t steps
        self.learning_offset = learning_offset  # Random offset for learning
        self.current_random_offset_learning = random.randint(-self.learning_offset, self.learning_offset)  # Random offset for learning
        self.A_learning = A_learning  # Learn the observation model
        self.B_learning = B_learning  # Learn the transition model
        self.A_BMR = A_BMR
        self.B_BMR = B_BMR
        self.decay = decay  # Forgetting rate for learning
        self.compute_novelty = compute_novelty

        # Store blanket states history for learning ----------------------------
        self.o_history = []  # Observations (opponent actions) history
        self.s_history = []  # Hidden states (opponent policies) history
        self.u_history = []  # Actions (ego) history
        
        # Agents values (change at each time step) -----------------------------
        self.u = torch.multinomial(self.E, 1).item()  # Starting action randomly sampled according to habits
        self.deterministic_actions = deterministic_actions
        self.policy_length = policy_length  # Length of the policy (number of actions to consider)

        self.VFE = [None] * num_agents  # Variational Free Energy for each agent (state factor)
        self.accuracy = [None] * num_agents
        self.complexity = [None] * num_agents
        self.energy = [None] * num_agents
        self.entropy = [None] * num_agents
        # q(s_j|u_i) Posterior predictive state distribution (for each factor) conditional on action u_i
        self.q_s_u = torch.empty((num_agents, num_actions, num_actions))  # shape (n_agents, n_actions, n_actions)

        self.EFE = torch.zeros(num_actions)  # Expected Free Energy (for each possible action)
        self.ambiguity = torch.zeros(num_actions)
        self.risk = torch.zeros(num_actions)
        self.salience = torch.zeros(num_actions)
        self.pragmatic_value = torch.zeros(num_actions)
        self.novelty = torch.zeros(num_actions)
        self.q_u = torch.zeros(num_actions)  # q(u_i) Policy of ego (self) agent
        
        self.expected_EFE = None  # Expected EFE averaged over my expected 

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
            for i, state in enumerate(self.s)
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
            self.s[factor_idx] = self.q_u.clone().detach()  # Ego factor is the previous timestep's policy
            self.theta[factor_idx] = torch.zeros_like(self.theta[factor_idx])  # PLACEHOLDER
            self.VFE[factor_idx] = 0  # PLACEHOLDER
            self.entropy[factor_idx] = 0  # PLACEHOLDER
            self.energy[factor_idx] = 0  # PLACEHOLDER
            self.accuracy[factor_idx] = 0  # PLACEHOLDER
            self.complexity[factor_idx] = 0  # PLACEHOLDER
            # Alter factors
            factors = range(1, len(self.s))
        else:
            '''
            Otherwise, ego has to infer all hidden states including its own, 
            i.e. "introspection" (towards self) and "theory of mind" (towards others)
            '''
            factors = range(len(self.s))
        
        # Iterate over (remaining) factors
        for factor_idx in factors:
            s_prev = self.s[factor_idx].clone().detach()  # State t-1
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
            self.s[factor_idx] = Dirichlet(variational_params).mean.detach()  # Variational posterior
            self.theta[factor_idx] = variational_params.detach()  # Store variational parameters (for next timestep)
            self.VFE[factor_idx] = VFE.detach()
            
            # Compute additional metrics (for validation and plotting)
            self.entropy[factor_idx] = entropy = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
            self.energy[factor_idx] = energy = -torch.sum(s_samples * (log_prior + log_likelihood), dim=-1).mean().detach()
            self.accuracy[factor_idx] = accuracy = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
            self.complexity[factor_idx] = complexity = -torch.sum(s_samples * (log_prior - log_s), dim=-1).mean().detach()
            assert torch.allclose(VFE, energy - entropy, atol=1e-6), "VFE != energy + entropy"
            assert torch.allclose(VFE, complexity - accuracy, atol=1e-6), "VFE != complexity - accuracy"

        # Data collection (for learning and plotting)
        self.s_history.append(self.s)
        self.o_history.append(o)
        
        return self.s

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
        
        # Predictive state posterior -------------------------------------------
        # (per factor 'f' and per possible action 'u')
        # E_{q(s)}[p(s' | s, u_i_hat)]
        self.q_s_u = torch.einsum(
            'funk,fk->fun',
            self.B,          # (f, u, n, k): factor, u (action), next (state), kurrent (state)
            self.s           # (f, k)
        )  # (f, u, n)

        # Predictive observation posterior -------------------------------------
        # (per factor 'f' and per possible action 'u')
        # E_{q(s'|u_i_hat)}[p(o | s)]
        self.q_o_u = torch.einsum(
            'fos,fus->fuo',
            self.A,         # (f, o, s)
            self.q_s_u      # (f, u, s)
        )  # (f, u, o)
        
        # If ego was to take action u_i_hat, the observation o_i would be guaranteed
        # to be o_i = u_i_hat, so replace q(o_i | u_i_hat) = one_hot(u_i_hat) for this action
        self.q_o_u[0] = torch.stack([
            F.one_hot(torch.tensor(u_i_hat), self.num_actions).to(torch.float)
            for u_i_hat in range(self.num_actions)
        ])

        # EFE of each action ---------------------------------------------------
        for u_i_hat in range(self.num_actions):

            # Per-factor terms
            for factor_idx in range(self.num_agents):

                # Expected ambiguity term (per factor) -------------------------
                H = -torch.diag(self.A[factor_idx] @ torch.log(self.A[factor_idx] + EPSILON))  # Conditional (pseudo?) entropy (of the generated emissions matrix)
                assert H.ndimension() == 1, "H is not a 1-dimensional tensor"

                s_pred = self.q_s_u[factor_idx, u_i_hat]  # shape (2, )
                assert s_pred.ndimension() == 1, "s_pred is not a 1-dimensional tensor"
                
                ambiguity[u_i_hat] += (H @ s_pred) # Ambiguity is conditional entropy of emissions
                # FIXME: not sure if these definitions are correct
                # risk[u_i] += (o_pred @ (torch.log(o_pred + EPSILON)))  - (o_pred @ log_C_modality) # Risk is negative posterior predictive entropy minus pragmatic value
                # salience[u_i] += -(o_pred @ (torch.log(o_pred + EPSILON)))  - (H @ s_pred) # Salience is negative posterior predictive entropy minus ambiguity (0)
                # pragmatic_value[u_i] += (o_pred @ log_C_modality) # Pragmatic value is negative cross-entropy

            # Joint predictive observation posterior ---------------------------
            # q(o_i, o_j, o_k | u_i_hat)
            # Create the einsum subscripts string dynamically for n_agents
            # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
            einsum_str = (
                ','.join([chr(105 + i) for i in range(self.num_agents)]) 
                + '->' 
                + ''.join([chr(105 + i) for i in range(self.num_agents)])
            )
            q_o_joint_u = torch.einsum(
                einsum_str, 
                *[self.q_o_u[i, u_i_hat] for i in range(self.num_agents)]
            )
            assert q_o_joint_u.shape == (self.num_actions, ) * (self.num_agents), (
                f"q_o_joint_u shape {q_o_joint_u.shape} != {(self.num_actions, ) * (self.num_agents)}"
            )
            assert torch.allclose(q_o_joint_u.sum(), torch.tensor(1.0)), (
                f"q_o_joint_u sum {q_o_joint_u.sum()} != 1.0"
            )

            # Risk term (joint) ------------------------------------------------
            # i.e. KL[q(o|u) || p*(o)]
            risk[u_i_hat] = torch.tensordot(
                (torch.log(q_o_joint_u + EPSILON) - self.log_C),
                q_o_joint_u,
                dims=self.num_agents
            )

            # Novelty ----------------------------------------------------------
            if self.compute_novelty:
                novelty[u_i_hat] += self.compute_A_novelty(u_i_hat)  # TODO: some sort of regularisation, novelty can be really large
                # TODO: B novelty?

        EFE = ambiguity + risk - novelty
        assert not torch.any(torch.isnan(EFE)), f"EFE has NaN: {EFE}"
        # assert torch.allclose(risk[u_i] + ambiguity[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] risk + ambiguity ({risk[u_i]} + {ambiguity[u_i]}={risk[u_i] + ambiguity[u_i]}) does not equal EFE (={EFE[u_i]})"
        # assert torch.allclose(-salience[u_i] - pragmatic_value[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] -salience - pragmatic value (-{salience[u_i]} - {pragmatic_value[u_i]}={-salience[u_i] - pragmatic_value[u_i]}) does not equal EFE (={EFE[u_i]})"

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
            u (torch.Tensor): action
            q_s_u (torch.Tensor): variational posterior over states given action
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
            H = -torch.diag(A[factor_idx] @ torch.log(A[factor_idx] + EPSILON))  # Conditional (pseudo?) entropy (of the generated emissions matrix)
            assert H.ndimension() == 1, "H is not a 1-dimensional tensor"

            s_pred = q_s_u[factor_idx]  # shape (2, )
            assert s_pred.ndimension() == 1, "s_pred is not a 1-dimensional tensor"
            
            ambiguity += (H @ s_pred) # Ambiguity is conditional entropy of emissions
            # FIXME: not sure if these definitions are correct
            # risk[u_i] += (o_pred @ (torch.log(o_pred + EPSILON)))  - (o_pred @ log_C_modality) # Risk is negative posterior predictive entropy minus pragmatic value
            # salience[u_i] += -(o_pred @ (torch.log(o_pred + EPSILON)))  - (H @ s_pred) # Salience is negative posterior predictive entropy minus ambiguity (0)
            # pragmatic_value[u_i] += (o_pred @ log_C_modality) # Pragmatic value is negative cross-entropy

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
        # assert q_o_joint_u.shape == (num_actions, ) * (num_agents), (
        #     f"q_o_joint_u shape {q_o_joint_u.shape} != {(num_actions, ) * (num_agents)}"
        # )
        # assert torch.allclose(q_o_joint_u.sum(), torch.tensor(1.0)), (
        #     f"q_o_joint_u sum {q_o_joint_u.sum()} != 1.0"
        # )

        # Risk term (joint) ------------------------------------------------
        # i.e. KL[q(o|u) || p*(o)]
        risk = torch.tensordot(
            (torch.log(q_o_joint_u + EPSILON) - log_C),
            q_o_joint_u,
            dims=self.num_agents
        )

        # Novelty ----------------------------------------------------------
        # if self.compute_novelty:
        #     novelty[u] += self.compute_A_novelty(u)  # TODO: some sort of regularisation, novelty can be really large
        #     # TODO: B novelty?

        # EFE of this action ---------------------------------------------------
        EFE = ambiguity + risk - novelty
        # assert not torch.any(torch.isnan(EFE)), f"EFE has NaN: {EFE}"
        # assert torch.allclose(risk[u_i] + ambiguity[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] risk + ambiguity ({risk[u_i]} + {ambiguity[u_i]}={risk[u_i] + ambiguity[u_i]}) does not equal EFE (={EFE[u_i]})"
        # assert torch.allclose(-salience[u_i] - pragmatic_value[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] -salience - pragmatic value (-{salience[u_i]} - {pragmatic_value[u_i]}={-salience[u_i] - pragmatic_value[u_i]}) does not equal EFE (={EFE[u_i]})"

        # Data collection ------------------------------------------------------
        # self.EFE[u] = EFE
        # self.ambiguity[u] = ambiguity
        # self.risk[u] = risk
        # self.salience[u] = salience
        # self.pragmatic_value[u] = pragmatic_value
        # self.novelty[u] = novelty
        
        return EFE
    
    def compute_A_novelty(self, u_i):
        '''
        Compute the novelty of the likelihood model A for action u_i
        '''
        novelty = 0
        # Da Costa et al. (2020; Eq. D.17)
        W = 0.5 * (1/self.A_params - 1/self.A_params.sum(dim=1, keepdim=True))
        for factor_idx in range(self.num_agents):
            s_pred = self.q_s_u[factor_idx, u_i]
            novelty += torch.dot(
                self.A[factor_idx] @ s_pred, 
                W[factor_idx] @ s_pred)
        return novelty
    
    def collect_policies(
            self,
            node, 
            q_s,
            policy_EFEs=None,
            current_policy=None,
        ):
        '''Function to traverse the tree and collect policies (as tensors)'''
        
        # Root node case
        if current_policy is None:
            current_policy = []
        if policy_EFEs is None:
            policy_EFEs = []
        if node.u is None:
            node.q_s_u = q_s
            new_policy_EFEs = policy_EFEs
        # Other nodes
        else:
            # Compute q(s|u) and EFE(u) for the current node
            node.q_s_u = torch.einsum(
                    'funk,fk->fun',
                    self.B,            # (f, u, n, k): factor, u (action), next (state), kurrent (state)
                    q_s           # (f, k)
                )[:, node.u].squeeze()  # (f, u, n) -> (f, n)
        
            node.EFE_u = self.compute_efe(node.u, node.q_s_u, self.A, self.log_C).unsqueeze(0)
            new_policy_EFEs = policy_EFEs + [node.EFE_u]  # EFEs collected top-down
        
        # Base case (leaf node)
        if not node.children:
            return [torch.cat(new_policy_EFEs)], [torch.cat(current_policy)]

        # Recursive case
        EFEs = []
        policies = []
        for child in node.children:
            new_policy = current_policy + [child.u]  # Policies collected bottom-up
            node_EFE, sub_policy = self.collect_policies(child, node.q_s_u, new_policy_EFEs, new_policy)
            EFEs.extend(node_EFE)
            policies.extend(sub_policy)

        return torch.vstack(EFEs), torch.vstack(policies)

    def select_action(self):
        
        # Build the policy tree and "collect" policies
        root = build_policy_tree(torch.arange(self.num_actions), self.policy_length)
        EFEs, policies = self.collect_policies(
            root, 
            q_s=self.s
        )

        EFE_policies = EFEs.sum(dim=1)
        self.EFE = EFE_policies  # Data collection
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
            self.s_history = torch.stack(self.s_history)  # Shape: (T, n_agents, n_actions)
            self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
            self.u_history = torch.tensor(self.u_history)  # Shape: (T, )
        
            # Learn
            if self.A_learning:
                self.learn_A()
            if self.B_learning:
                self.learn_B()

            # Reset history
            self.s_history = []
            self.o_history = []
            self.u_history = []
            self.current_random_offset_learning = random.randint(-self.learning_offset, self.learning_offset)  # Renew random offset

    def learn_A(self):

        # Expand dimensions to prepare for broadcasting
        s_hist_expanded = self.s_history.unsqueeze(-2)  # Shape: (T, n_agents, 1, n_actions)
        o_hist_expanded = self.o_history.unsqueeze(-1)  # Shape: (T, n_agents, n_actions, 1)

        # Perform the row-wise outer product
        outer_products = s_hist_expanded * o_hist_expanded  # Shape: (T, n_agents, n_actions, n_actions)

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
            elif self.A_BMR == 'uniform':
                A_reduced = torch.ones_like(self.A_params)
            else:
                raise ValueError(f"Invalid A_BMR: {self.A_BMR}")

            mixture = self.decay
            A_reduced_params = (
                mixture * self.A_params 
                + (1 - mixture) * A_reduced
            )

            # Update model parameters if they reduce the free energy
            self.delta_F = []
            for factor_idx in range(len(self.s)):
                delta_F = delta_free_energy(
                    A_posterior_params[factor_idx].flatten(), 
                    self.A_params[factor_idx].flatten(), 
                    A_reduced_params[factor_idx].flatten()
                )
                self.delta_F.append(delta_F)  # Data collection
                if delta_F < 0:
                    # Reduced model is preferred -> replace full model with reduced model
                    self.A_params[factor_idx] = A_reduced_params[factor_idx]
                    # print(f"Factor {factor_idx}: Reduced model is preferred.")
                    # print(f"Delta F: {delta_F}")
                    # print(f"Reduced: {A_reduced_params[factor_idx]}")
                    # print(f"Posterior: {A_posterior_params[factor_idx]}")
                else:
                    # Full model is preferred -> update posterior
                    self.A_params[factor_idx] = A_posterior_params[factor_idx]
        else:
            self.A_params = A_posterior_params
        
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)  # Shape: (n_agents, n_actions, n_actions)

    def learn_B(self):

        # Shift arrays for prev and next
        s_prev = self.s_history[:-1]  # Shape: (T-1, n_agents, n_actions)
        s_next = self.s_history[1:]  # Shape: (T-1, n_agents, n_actions)
        
        # Expand dimensions to prepare for broadcasting
        s_prev_expanded = s_prev.unsqueeze(-2)  # Shape: (T, n_agents, 1, n_actions)
        s_next_expanded = s_next.unsqueeze(-1)  # Shape: (T, n_agents, n_actions, 1)

        # Perform the row-wise outer product
        outer_products = s_prev_expanded * s_next_expanded  # Shape: (T, n_agents, n_actions, n_actions)
        T = outer_products.shape[0]

        # Update parameters for every transition (s, u, s') in the history
        B_posterior_params = self.B_params.clone()
        for t in range(outer_products.shape[0]):
            # Likelihood parameters update
            delta_params = outer_products[t] / T  # Shape: (n_agents, n_actions, n_actions)
            u_it = self.u_history[t].item()   # Action u_i at time t
            B_posterior_params[:, u_it] = self.B_params[:, u_it] + delta_params

        # Bayesian Model Reduction ---------------------------------------------
        if self.B_BMR:
            
            if self.B_BMR == 'identity':
                B_reduced = torch.stack([
                    torch.eye(self.num_actions)
                    for action_idx in range(self.B.shape[1])
                    for factor_idx in range(self.B.shape[0])
                ]).view(*self.B.shape)
            elif self.B_BMR == 'uniform':
                B_reduced = torch.ones_like(self.B_params)
            else:
                raise ValueError(f"Invalid B_BMR: {self.B_BMR}")
            
            mixture = self.decay
            B_reduced_params = (
                mixture * self.B_params 
                + (1 - mixture) * B_reduced
            )

            # Update model parameters if they reduce the free energy
            self.delta_F = []
            for factor_idx in range(len(self.s)):
                delta_F = delta_free_energy(
                    B_posterior_params[factor_idx].flatten(), 
                    self.B_params[factor_idx].flatten(), 
                    B_reduced_params[factor_idx].flatten()
                )
                self.delta_F.append(delta_F)  # Data collection
                if delta_F < 0:
                    # Reduced model is preferred -> replace full model with reduced model
                    self.B_params[factor_idx] = B_reduced_params[factor_idx]
                else:
                    # Full model is preferred -> update posterior
                    self.B_params[factor_idx] = B_posterior_params[factor_idx]
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

