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

torch.set_printoptions(precision=2)

PROPRIOCEPTION = False
NOVELTY = False
LEARN_A = True
LEARN_A_JOINT = True
LEARN_B = True
BETA_0 = 1.0

class Agent:

    def __init__(self, id=None, game_matrix=None, beta_1=1, decay=0.99, dynamic_precision=False):

        self.id = id
        
        # Learning parameters --------------------------------------------------
        self.beta_0 = BETA_0  # beta in Gamma distribution (stays fixed)
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean)
        self.gamma = self.beta_1 / self.beta_0 #Current precision
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.decay = decay  # Forgetting rate for learning

        # Generative model hyperparameters -------------------------------------
        self.game_matrix = game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
        self.num_actions = num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        self.num_agents = num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model parameters ------------------------------------------
        # The shape of parameters is generally (num_actions ** num_agents), i.e.
        # a flat representation of the joint action space
        self.theta = torch.ones(num_actions ** num_agents)  # Variational Dirichlet distribution parameters
        
        # self.A_params = torch.ones((num_actions ** num_agents, num_actions ** num_agents))  # Flat observation model prior
        self.A_params = torch.eye(num_actions ** num_agents) + 1e-9  # Identity observation model prior
        self.A = self.A_params / self.A_params.sum(dim=0, keepdim=True)
        
        self.B_params = torch.stack([
            torch.eye(num_actions ** num_agents) 
            # torch.ones(num_actions ** num_agents) 
            for _ in range(num_actions) 
        ])
        self.B = self.B_params / self.B_params.sum(dim=1, keepdim=True)
        
        self.log_C = game_matrix.to(torch.float)  # Payoffs for joint actions (in matrix form, from row player's perspective) (force to float)
        self.s = Dirichlet(self.theta).mean  # Categorical state prior (D)
        self.E = torch.ones(num_actions) / num_actions  # Habits 

        # Store blanket states history for learning ----------------------------
        self.o_history = []  # Observations (opponent actions) history
        self.s_history = []  # Hidden states (opponent policies) history
        self.u_history = []  # Actions (ego) history
        
        # Agents values (change at each time step) -----------------------------
        self.u = torch.multinomial(self.E, 1).item()  # Starting action randomly sampled according to habits
        
        self.VFE = None  # Variational Free Energy
        self.accuracy = None
        self.complexity = None
        self.energy = None
        self.entropy = None
        self.q_s_u = torch.empty((num_actions, *self.s.shape))  # q(s|u_i) Predictive state posterior distribution (conditional on action u_i)
        
        self.EFE = torch.zeros(num_actions)  # Expected Free Energy (for each possible action)
        self.ambiguity = torch.zeros(num_actions)
        self.risk = torch.zeros(num_actions)
        self.risk = torch.zeros(num_actions)
        self.salience = torch.zeros(num_actions)
        self.pragmatic_value = torch.zeros(num_actions)
        self.novelty = torch.zeros(num_actions)
        self.q_u = torch.zeros(num_actions)  # q(u_i) Policy of ego agent
        
        self.expected_EFE = [None]  # Expected EFE averaged over my expected 

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

    # ========================================
    # Perception 
    # =======================================
    def infer_state(self, o, learning_rate=1e-2, num_iterations=10, num_samples=100):
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
            learning_rate (float): Learning rate for the optimizer
            num_iterations (int): Number of iterations to run the optimizer
            num_samples (int): Number of MC samples to draw from the variational distribution
        
        Returns:
            s (torch.Tensor): Hidden state tensor of shape (n_agents, n_actions)
        '''
        # if PROPRIOCEPTION:
        #     '''
        #     If proprioception is enabled, ego can perceive the true hidden state for the ego factor, 
        #     i.e. q(s_i) = q(u_i)
        #     and will have to infer the hidden states of the other agents ("theory of mind")
        #     '''
        #     # Ego factor
        #     factor_idx = 0
        #     self.s[factor_idx] = self.q_u.clone().detach()  # Ego factor is the previous timestep's policy
        #     self.theta[factor_idx] = torch.zeros_like(self.theta[factor_idx])  # PLACEHOLDER
        #     self.VFE[factor_idx] = 0  # PLACEHOLDER
        #     self.entropy[factor_idx] = 0  # PLACEHOLDER
        #     self.energy[factor_idx] = 0  # PLACEHOLDER
        #     self.accuracy[factor_idx] = 0  # PLACEHOLDER
        #     self.complexity[factor_idx] = 0  # PLACEHOLDER
        #     # Alter factors
        #     factors = range(1, len(self.s))
        # else:
        #     '''
        #     Otherwise, ego has to infer all hidden states including its own, 
        #     i.e. "introspection" (towards self) and "theory of mind" (towards others)
        #     '''
        #     factors = range(len(self.s))

        # Pre-process observation ----------------------------------------------
        # Convert one-hot encoding to action indexes (for each agent)
        o_idx = [torch.argmax(o_i) for o_i in o]  
        # Convert binary o_idx (each agent) to a single decimal value (joint action) 
        # in [0, n_actions^n_agents - 1]
        o_idx = sum([idx * 2**i for i, idx in enumerate(o_idx)])  
        # Convert joint action index to one-hot encoding (joint)
        self.o = torch.zeros_like(self.s)
        self.o[o_idx] = 1.0  # one-hot encoding
        assert torch.allclose(self.o.sum(), torch.tensor(1.0)), f"'o' tensor sum {self.o.sum()} != 1.0"

        s_prev = self.s.clone().detach()  # State t-1
        assert torch.allclose(s_prev.sum(), torch.tensor(1.0)), f"'s_prev' tensor sum {s_prev.sum()} != 1.0"
        log_prior = torch.log(self.B[self.u] @ s_prev + 1e-9)  # New prior is old posterior
        log_likelihood = torch.log(self.A.T @ self.o + 1e-9)  # Probability of observation given hidden states

        variational_params = self.theta.clone().detach().requires_grad_(True)  # Variational Dirichlet distribution parameters
        # variational_params = variational_params + torch.rand_like(variational_params)  # Add noise
        # variational_params.data.clamp_(min=1e-3)  # Ensure parameters are positive
        # variational_params = torch.ones_like(self.theta).requires_grad_(True)  # Random initialization
        optimizer = torch.optim.Adam([variational_params.detach().requires_grad_(True)], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            s_samples = Dirichlet(variational_params).rsample((num_samples,))  # Variational dist samples
            log_s = torch.log(s_samples + 1e-9)  # Log variational dist samples
            vfe_samples = torch.sum(s_samples * (log_s - log_likelihood - log_prior), dim=-1) 
            VFE = vfe_samples.mean()

            VFE.backward()
            optimizer.step()
            variational_params.data.clamp_(min=1e-3)

            # Results of variational inference: variational posterior, variational parameters, VFE
            self.s = Dirichlet(variational_params).mean.detach()  # Variational posterior
            self.theta = variational_params.detach()  # Store variational parameters (for next timestep)
            self.VFE = VFE.detach()
            
        # Compute additional metrics (for validation and plotting)
        self.entropy = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
        self.energy = -torch.sum(s_samples * (log_prior + log_likelihood), dim=-1).mean().detach()
        self.accuracy = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
        self.complexity = -torch.sum(s_samples * (log_prior - log_s), dim=-1).mean().detach()
        assert torch.allclose(self.energy - self.entropy, VFE, atol=1e-6), "VFE != energy + entropy"
        assert torch.allclose(self.complexity - self.accuracy, VFE, atol=1e-6), "VFE != complexity - accuracy"

        # Data collection (for learning and plotting)
        self.s_history.append(self.s)
        self.o_history.append(self.o)
        
        return self.s

    # ==========================================================================
    # Action 
    # ==========================================================================
    def compute_efe(self):
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
        
        # For each action
        for u_i in range(self.num_actions):

            # Predictive state posterior ---------------------------------------
            # E_{q(s)}[p(s'|s, u_i)]
            q_s_u = self.B[u_i] @ self.s  # Predicted state q(s' | s, u_i)
            assert q_s_u.ndimension() == 1, "s_pred is not a 1-dimensional tensor"
            self.q_s_u[u_i] = q_s_u  # Store for later use

            # Expected ambiguity term ------------------------------------------
            H = -torch.diag(self.A @ torch.log(self.A + 1e-9))  # Conditional (pseudo?) entropy (of the generated emissions matrix)
            assert H.ndimension() == 1, "H is not a 1-dimensional tensor"

            ambiguity[u_i] += (H @ q_s_u) # Ambiguity is conditional entropy of emissions
            
            # Predictive observation posterior ---------------------------------
            q_o_u_full = self.A @ q_s_u
            q_o_u = q_o_u_full.view(*(self.num_actions, )*self.num_agents)[u_i]  # Select the action u_i, so shape is now (2, 2)
            q_o_u = q_o_u / q_o_u.sum()  # Normalise to a probability distribution
            q_o_u = q_o_u.view(-1)  # flatten

            assert q_o_u.shape == (self.num_actions * (self.num_agents-1), ), (
                f"q_o_joint_u shape {q_o_u.shape} should be {self.num_actions * (self.num_agents-1)}."
            )
            assert torch.allclose(q_o_u.sum(), torch.tensor(1.0)), (
                f"q_o_joint_u tensor does not sum to 1 ({q_o_u.sum()})."
            )

            # Risk term --------------------------------------------------------
            # i.e. KL[q(o|u) || p(o)]
            risk[u_i] = (
                (torch.log(q_o_u + 1e-9) - self.log_C[u_i].view(-1))
                @
                q_o_u
            )
            # risk[u_i] = kl_divergence(
            #     Dirichlet(q_o_joint_u),
            #     Dirichlet(self.log_C[u_i].view(-1))
            # )

            salience[u_i] += -(q_o_u @ (torch.log(q_o_u + 1e-9)))  - (H @ q_s_u) # Salience is negative posterior predictive entropy minus ambiguity
            pragmatic_value[u_i] += (q_o_u @ self.log_C[u_i].view(-1)) # Pragmatic value is negative cross-entropy
        
            # Novelty ----------------------------------------------------------
            if NOVELTY:
                novelty[u_i] += self.compute_A_novelty(self.A_params, q_s_u)

        EFE = ambiguity + risk - novelty
        assert not torch.any(torch.isnan(EFE)), f"EFE has NaN: {EFE}"
        assert torch.allclose(EFE, risk + ambiguity - novelty, atol=1e-4), (
            f"EFE (={EFE}) != risk + ambiguity - novelty ({risk} + {ambiguity} - {novelty}={risk + ambiguity - novelty})"
        )
        assert torch.allclose(EFE, -salience - pragmatic_value - novelty, atol=1e-4), (
            f"EFE (={EFE}) != -salience - pragmatic value - novelty (-{salience} - {pragmatic_value} - {novelty}={-salience - pragmatic_value - novelty})"
        )

        # Data collection ------------------------------------------------------
        self.EFE = EFE
        self.ambiguity = ambiguity
        self.risk = risk
        self.salience = salience
        self.pragmatic_value = pragmatic_value
        self.novelty = novelty
        
        return EFE

    def compute_A_novelty(self, A_params, q_s_u):
        '''
        Compute the novelty of the likelihood model A under the current state posterior q(s|u_i)
        '''
        novelty = 0
        # Da Costa et al. (2020; Eq. D.17)
        W = 0.5 * (1/A_params - 1/A_params.sum(dim=0, keepdim=True))
        novelty += torch.dot(
            self.A @ q_s_u, 
            W @ q_s_u)
        return novelty / ((A_params.shape[0])**2)  # regularize by size of square matrix

    def select_action(self):
        EFE = self.compute_efe()
        q_u = torch.softmax(torch.log(self.E) - self.gamma * EFE, dim=0)
        assert torch.allclose(q_u.sum(), torch.tensor(1.0)), (
            "q_u policy tensor does not sum to 1.",
            f"q_u: {q_u}",
            f"q_u.sum(): {q_u.sum()}",
            f"EFE: {EFE}"
        )
        self.q_u = q_u

        self.u = torch.multinomial(q_u, 1).item()
        self.u_history.append(self.u)

        if self.dynamic_precision:
            self.update_precision(EFE, q_u)

        return self.u

    def update_precision(self, EFE, q_u):
        # Compute the expected EFE as a scalar value
        self.expected_EFE = torch.dot(q_u, EFE).item()
        
        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.beta_1 / (self.beta_0 - self.expected_EFE)
        
        return self.gamma

    # ========================================
    # Learning p(u_j | u_i)
    # =======================================
    
    def learn(self):

        # Convert history to tensors
        self.s_history = torch.stack(self.s_history)  # Shape: (T, n_agents, n_actions)
        self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
        self.u_history = torch.tensor(self.u_history)  # Shape: (T, )
    
        if LEARN_A:
            self.learn_A()
        if LEARN_B:
            self.learn_B()

        # Reset history
        self.s_history = []
        self.o_history = []
        self.u_history = []

    def learn_A(self):

        # Expand dimensions to prepare for broadcasting
        s_hist_expanded = self.s_history.unsqueeze(-2)  # Shape: (T, 1, n_actions**n_agents)
        o_hist_expanded = self.o_history.unsqueeze(-1)  # Shape: (T, n_actions**n_agents, 1)

        # Perform the row-wise outer product
        outer_products = s_hist_expanded * o_hist_expanded  # Shape: (T, n_actions**n_agents, n_actions**n_agents)

        # Posterior parameters
        delta_params = outer_products.mean(dim=0)  # Shape: (n_agents, n_actions, n_actions)
        A_posterior_params = self.A_params + delta_params  # Shape: (n_agents, n_actions, n_actions)

        # Bayesian Model Reduction ---------------------------------------------
        BMR = True
        if BMR:
            A_identity = torch.eye(self.A_params.shape[0]) + 1e-9  # Identity observation model prior
            A_uniform = torch.ones_like(self.A_params)  # Flat observation model prior
            mixture = self.decay
            A_reduced_params = (
                mixture * self.A_params 
                + (1 - mixture) * A_uniform
            )

            # Update model parameters if they reduce the free energy
            delta_F = delta_free_energy(
                A_posterior_params.flatten(), 
                self.A_params.flatten(), 
                A_reduced_params.flatten()
            )
            self.A_params = A_reduced_params if delta_F < 0 else A_posterior_params
        else:
            self.A_params = A_posterior_params
        
        self.A = self.A_params / self.A_params.sum(dim=0, keepdim=True)  # Shape: (n_agents, n_actions, n_actions)

    def learn_B(self):

        # Shift arrays for prev and next
        s_prev = self.s_history[:-1]  # Shape: (T-1, n_actions)
        s_next = self.s_history[1:]  # Shape: (T-1, n_actions)
        
        # Expand dimensions to prepare for broadcasting
        s_prev_expanded = s_prev.unsqueeze(-2)  # Shape: (T, 1, n_actions)
        s_next_expanded = s_next.unsqueeze(-1)  # Shape: (T, n_actions, 1)

        # Perform the row-wise outer product
        outer_products = s_prev_expanded * s_next_expanded  # Shape: (T, n_actions, n_actions)
        T = outer_products.shape[0]

        # Update parameters for every transition (s, u, s') in the history
        B_posterior_params = self.B_params.clone()
        for t in range(T):
            # Likelihood parameters update
            delta_params = outer_products[t] / T  # Shape: (n_actions, n_actions)
            u_it = self.u_history[t].item()   # Action u_i at time t
            B_posterior_params[u_it] = self.B_params[u_it] + delta_params

        # Bayesian Model Reduction ---------------------------------------------
        BMR = True
        if BMR:
            B_identity = torch.stack([
                torch.eye(self.num_actions ** self.num_agents) 
                # torch.ones(num_actions ** num_agents) 
                for _ in range(self.num_actions) 
            ])
            mixture = self.decay
            B_reduced_params = (
                mixture * self.B_params 
                + (1 - mixture) * B_identity
            )

            # Update model parameters if they reduce the free energy
            for u in range(self.num_actions):
                delta_F = delta_free_energy(
                    B_posterior_params[u].flatten(), 
                    self.B_params[u].flatten(), 
                    B_reduced_params[u].flatten()
                )
                B_posterior_params[u] = B_reduced_params[u] if delta_F < 0 else B_posterior_params[u]
        else:
            self.B_params = B_posterior_params

        self.B = self.B_params / self.B_params.sum(dim=1, keepdim=True)

    # def learn_A_joint(self, o=None):
    #     '''
    #     Compute the joint observation likelihood

    #     Requires the current hidden state distribution (self.s), and
    #     the joint observation likelihood (self.A_joint) to be updated.
        
    #     Args:
    #         o (torch.Tensor): Observation tensor of shape (n_agents, n_actions),
    #             i.e. one-hot encoding of actions for each agent
    #     '''
    #     n_agents = self.A.shape[0]  # Number of agents (including self)
    #     n_actions = self.A.shape[-1]  # Number of actions
    #     if o is None:
    #         T = len(self.s_history) # Number of timesteps
    #         s_history = self.s_history
    #         o_history = self.o_history
    #     else:
    #         T = 1
    #         s_history = [self.s] 
    #         o_history = [o] 

    #     # Learning -------------------------------------------------------------
    #     # Create the einsum subscripts string dynamically for n_agents
    #     # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
    #     einsum_str = (
    #         ','.join([chr(105 + i) for i in range(n_agents)]) 
    #         + '->' 
    #         + ''.join([chr(105 + i) for i in range(n_agents)])
    #     )

    #     A_joint_posterior = self.A_joint.clone()
    #     for t in range(T):
    #         s_t = s_history[t]  # Shape: (n_agents, n_actions)
    #         o_t = o_history[t]  # Shape: (n_agents, n_actions)

    #         # Compute joint state prior
    #         q_s_joint = torch.einsum(einsum_str, *[s_t[factor_idx] for factor_idx in range(n_agents)])
    #         assert q_s_joint.shape == (n_actions, ) * (n_agents), f"q_s_joint shape {q_s_joint.shape} is not correct."

    #         o_indices = tuple([torch.argmax(o_t[factor_idx]).item() for factor_idx in range(n_agents)])
    #         assert self.A_joint[o_indices].shape == q_s_joint.shape, f"A_joint[o_indices] shape {self.A_joint[o_indices].shape} is not correct."
            
    #         A_joint_posterior[o_indices] += q_s_joint

    #         # TODO: Consider using this instead of the current method
    #         # A_joint_square = self.A_joint.view(n_actions ** n_agents, n_actions ** n_agents)
    #         # A_joint_posterior_square = A_joint_square.clone()
    #         # o_joint = torch.einsum(einsum_str, *[o_t[factor_idx] for factor_idx in range(n_agents)])
    #         # A_joint_posterior_square += torch.outer(q_s_joint.flatten(), o_joint.flatten())
    #         # ... then reshape into (n_actions, ) * (2 * n_agents) and update A_joint_posterior

    #     # Bayesian Model Reduction ---------------------------------------------
    #     BMR = True
    #     if BMR:
    #         self.A_joint = self.decay * A_joint_posterior
    #         # A_joint_reduced = self.decay * A_joint_posterior
    
    #         # # Update model parameters if they reduce the free energy
    #         # delta_F = delta_free_energy(
    #         #     A_joint_posterior.flatten(), 
    #         #     self.A_joint.flatten(), 
    #         #     A_joint_reduced.flatten()
    #         # )
    #         # if delta_F < 0:
    #         #     print('reduced')
    #         #     # Reduced model is preferred -> replace full model with reduced model
    #         #     self.A_joint = A_joint_reduced
    #         #     # print(f"Factor {factor_idx}: Reduced model is preferred.")
    #         #     # print(f"Delta F: {delta_F}")
    #         #     # print(f"Reduced: {A_reduced_params[factor_idx]}")
    #         #     # print(f"Posterior: {A_posterior_params[factor_idx]}")
    #         # else:
    #         #     # Full model is preferred -> update posterior
    #         #     self.A_joint = A_joint_posterior
    #     else:
    #         self.A_joint = A_joint_posterior

    #     # TODO: do we need to normalise or not? If it is a likelihood *function*, no need.


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