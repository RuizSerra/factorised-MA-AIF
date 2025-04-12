'''
Active Inference Agent in a Network

Authors: Jaime Ruiz Serra
Date: 2025-04
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

INFERENCE_NUM_ITERATIONS = 10
INFERENCE_NUM_SAMPLES = 100
INFERENCE_LEARNING_RATE = 1e-2

from agent import multivariate_beta, delta_free_energy, TreeNode, build_policy_tree

# class DataCollector:

#     def __init__(self):
#         self.data = {
#             "q_s": [],
#             "o": [],
#             "u": [],
#             "VFE": [],
#             "accuracy": [],
#             "complexity": [],
#             "energy": [],
#             "entropy": [],
#             "ambiguity": [],
#             "risk": [],
#             "salience": [],
#             "pragmatic_value": []
#         }

class GenerativeModel:

    def __init__(
            self,
            agent,
            A_prior,
            B_prior,
            C_prior,
            D_prior,
            E_prior,
            num_agents=2,
            num_actions=2,
            policy_length=1,
        ):
        
        self.agent = agent
        self.num_agents = num_agents
        self.num_actions = num_actions

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

        self.set_log_C(C_prior)  # Log preference over observations (payoffs)
        self.theta = [torch.ones(num_actions) for _ in range(num_agents)] if D_prior is None else D_prior  # Dirichlet state prior
        self.D = self.q_s = torch.stack([Dirichlet(theta).mean for theta in self.theta])  # Categorical state prior (D)
        self.E = torch.ones(num_actions**policy_length) / num_actions if E_prior is None else E_prior  # Habits 
        
        self.gamma = self.agent.beta_1 / self.agent.beta_0 # Current precision

        # Store histories for learning -----------------------------------------
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

        self.policy_tree_root = build_policy_tree(torch.arange(num_actions), policy_length)

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
        if self.agent.interoception:
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
            optimizer = torch.optim.Adam([variational_params], lr=INFERENCE_LEARNING_RATE)

            for _ in range(INFERENCE_NUM_ITERATIONS):
                optimizer.zero_grad()

                s_samples = Dirichlet(variational_params).rsample((INFERENCE_NUM_SAMPLES,))  # Variational dist samples
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
        B_prime_params = self.B_params[:, u].squeeze() + self.agent.B_learning_rate * outer_product
        B_prime = B_prime_params / B_prime_params.sum(dim=1, keepdim=True)

        # KL divergence D[ B'[u] || B[u] ] for each factor
        kl_div = (
            B_prime * torch.log(B_prime / (self.B[:, u].squeeze() + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))

        return kl_div.sum()  # over factors

    def compute_efe(self, u, q_s_u):
        '''
        Compute the Expected Free Energy (EFE) of a given action
        
        Args:
            u (int): action
            q_s_u (torch.Tensor): variational posterior over states given action is u

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

        A = self.A  # shorthand
        log_C = self.log_C
        
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
        if self.agent.compute_novelty:
            if self.agent.A_learning:
                novelty += self.compute_A_novelty(q_s_u, q_o_u)
            if self.agent.B_learning:
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

    def collect_policies(
            self,
            node=None, 
            q_s=None,
            policy_EFEs=None,
            policy_EFE_terms=None,
            current_policy=None,
        ):
        '''Function to traverse the tree and collect policies (as tensors)'''
        
        # Root node case
        # if node.u is None:
        if node is None:
            node = self.policy_tree_root
            node.q_s_u = q_s if q_s is not None else self.q_s
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
        
            node.EFE_u, node.EFE_terms = self.compute_efe(node.u, node.q_s_u)
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

    def update_precision(self, EFE, q_u):
        # Compute the expected EFE as a scalar value
        self.expected_EFE = torch.dot(q_u, EFE).item()
        
        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.agent.beta_1 / (self.agent.beta_0 - self.expected_EFE)
        
        return self.gamma
    
    # ==========================================================================
    # Learning
    # ==========================================================================
    def learn(self):

        # Convert history to tensors
        self.q_s_history = torch.stack(self.q_s_history)  # Shape: (T, n_agents, n_actions)
        self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
        self.u_history = torch.tensor(self.u_history)  # Shape: (T, )
    
        # Learn
        if self.agent.A_learning:
            self.learn_A()
        if self.agent.B_learning:
            self.learn_B()

        # Reset history
        self.q_s_history = []
        self.o_history = []
        self.u_history = []

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
        if self.agent.A_BMR:

            if self.agent.A_BMR == 'identity':
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
            elif self.agent.A_BMR == 'uniform':
                A_reduced = torch.ones_like(self.A_params)
                raise NotImplementedError("Uniform BMR for A disabled.")
            elif self.agent.A_BMR == 'softmax':
                STRENGTH = self.agent.decay
                A_reduced = F.softmax(
                    STRENGTH * self.A_params,
                    dim=-1
                )
            else:
                raise ValueError(f"Invalid A_BMR: {self.agent.A_BMR}")
            
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
        LEARNING_RATE = self.agent.B_learning_rate if self.agent.B_learning_rate is not None else 1/T
        for t in range(outer_products.shape[0]):
            # Likelihood parameters update
            delta_params = outer_products[t]  # Shape: (n_agents, n_actions, n_actions)
            u_it = self.agent.u_history[t]   # Action u_i at time t
            B_posterior_params[:, u_it] = self.B_params[:, u_it] + LEARNING_RATE * delta_params

        # Bayesian Model Reduction ---------------------------------------------
        if self.agent.B_BMR:
            
            if self.agent.B_BMR == 'identity':
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
            elif self.agent.B_BMR == 'uniform':
                B_reduced = torch.ones_like(self.B_params)
                raise NotImplementedError("Uniform BMR for B disabled.")
            elif self.agent.B_BMR == 'softmax':
                STRENGTH = self.agent.decay
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
                raise ValueError(f"Invalid B_BMR: {self.agent.B_BMR}")

        else:
            self.B_params = B_posterior_params

        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)



class NetworkAgent:

    def __init__(
            self, 
            id=None, 
            game_matrices=None, 
            # Perception
            interoception:bool=False,
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
            B_BMR:Union[str, None]='softmax',
        ):
        '''Initialise an agent with the following parameters
        
        Args:
            - id (int): The identifier of the agent
            - game_matrices (list of torch.Tensor): List of game matrices (payoffs) of the games (one for each neighbour)
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

        # Learning parameters --------------------------------------------------
        # Precision
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.beta_0 = beta_0  # beta in Gamma distribution (stays fixed)
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean)
        # Perception
        self.interoception = interoception  # Ego can perceive its own hidden state
        self.inference_num_iterations = INFERENCE_NUM_ITERATIONS
        self.inference_num_samples = INFERENCE_NUM_SAMPLES
        self.inference_learning_rate = INFERENCE_LEARNING_RATE
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

        # Generative model hyperparameters -------------------------------------
        self.game_matrices = [
            game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
            for game_matrix in game_matrices
        ]
        # FIXME: assumes all games have the same number of actions
        self.num_actions = num_actions = game_matrices[0].shape[0]  # Number of actions (assuming symmetrical actions)
        self.num_agents = num_agents = game_matrices[0].ndim  # Number of players (rank of game tensor)

        self.generative_models = [
            GenerativeModel(
                agent=self,
                A_prior=A_prior,
                B_prior=B_prior,
                C_prior=game_matrix,
                D_prior=D_prior,
                E_prior=E_prior,
            )
            for game_matrix in game_matrices
        ]

        # TODO: there is an E (habits) within each generative model, as well as a global E?
        self.E = torch.ones(num_actions**policy_length) / num_actions if E_prior is None else E_prior  # Habits 

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

        # o contains (o_ego, o_neighbour1, o_neighbour2, ...)
        for neighbour_idx, generative_model in enumerate(self.generative_models):
            generative_model.infer_state(
                o[[0, neighbour_idx + 1]]  # Pass the observations for ego and the neighbour
            )

    # ==========================================================================
    # Action 
    # ==========================================================================
    def select_action(self):
        
        # Collect the EFE and precision values from each game
        gammas_per_game = torch.empty((len(self.generative_models),))  # shape: (n_neighbours, )
        EFE_per_game = torch.empty((len(self.generative_models), self.num_actions))  # shape: (n_neighbours, n_actions)
        for idx, generative_model in enumerate(self.generative_models):
            EFEs, EFE_terms, policies = generative_model.collect_policies()

            EFE_policies = EFEs.sum(dim=1)
            gammas_per_game[idx] = generative_model.gamma
            EFE_per_game[idx] = EFE_policies
        
        # Select final action from all games' EFEs and precisions
        q_u = torch.softmax(
            torch.log(self.E) - torch.einsum('ij,i->j', EFE_per_game, gammas_per_game),  # shape: (n_actions, )
            dim=0
        )

        assert torch.allclose(q_u.sum(), torch.tensor(1.0)), (
            "q_u policy tensor does not sum to 1.",
            f"q_u: {q_u}",
            f"q_u.sum(): {q_u.sum()}",
            # f"EFE: {EFE_policies}"
        )
        # Data collection ------------------------------------------------------
        self.EFE = EFE_per_game
        # self.EFE_terms = EFE_terms.reshape(*EFEs.shape, -1)
        self.q_u = q_u

        # Select action
        policy_idx = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u = policies[policy_idx][0].item()
        # self.u = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u_history.append(self.u)

        # Update precision of each generative model
        if self.dynamic_precision:
            for idx, generative_model in enumerate(self.generative_models):
                generative_model.update_precision(EFE_per_game[idx], q_u)  # TODO: q_u is global, not per game

        return self.u

    # ==========================================================================
    # Learning
    # ==========================================================================

    def learn(self):
        '''Update generative models.
        
        This method is called every step, but the agent should only actually 
        learn every t steps, hence the logic below.
        '''

        # Learn every t steps
        if len(self.u_history) == (self.learn_every_t_steps + self.current_random_offset_learning):

            for generative_model in self.generative_models:
                generative_model.learn()

            self.current_random_offset_learning = random.randint(
                -self.learning_offset, 
                self.learning_offset
            )  # Renew random offset