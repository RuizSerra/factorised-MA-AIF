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

class Agent:
    def __init__(self, id=None, game_matrix=None, beta_1=1, decay=0.99, dynamic_precision=False):

        # ========================================
        # Attributes
        # =======================================
        self.id = id
        
        # Learning parameters
        self.beta_0 = 1.0  # beta in Gamma distribution (stays fixed)
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean)
        self.gamma = self.beta_1 / self.beta_0 #Current precision
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.decay = decay  # Forgetting rate for learning

        # Generative model hyperparameters
        self.game_matrix = game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
        num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model/variational posterior parameters
        self.alpha = [torch.ones(num_actions) for _ in range(num_agents)]  # Dirichlet state prior
        self.A_params = torch.ones((num_agents, num_actions, num_actions))
        # self.A_params = torch.stack([torch.eye(num_actions) for _ in range(num_agents)]) + 1e-9  # Identity observation model
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)
        self.A_joint = torch.ones((num_actions,) * 2 * num_agents)  # Joint observation model
        # self.B_params = torch.ones((num_agents, num_actions, num_actions, num_actions))
        self.B_params = torch.stack([
            torch.eye(num_actions) 
            for _ in range(num_actions) 
            for _ in range(num_agents)
        ]).reshape(num_agents, num_actions, num_actions, num_actions)
        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)
        self.log_C = game_matrix.to(torch.float)  # Payoffs for joint actions (in matrix form, from row player's perspective) (force to float)
        self.prob_C = None
        self.s = torch.stack([Dirichlet(alpha).mean for alpha in self.alpha])  # Categorical state prior (D)
        self.E = torch.ones(num_actions) / num_actions  # Habits 

        # Store blanket states history for learning
        self.s_history = []
        self.o_history = []
        self.u_history = []
        
        # Values for t = 0, plus resetting each timestep
        self.action = torch.multinomial(self.E, 1).item()  # Starting action randomly sampled according to habits
        self.VFE = [None] * num_agents  # Variational Free Energy for each agent (state factor)
        self.accuracy = [None] * num_agents
        self.complexity = [None] * num_agents
        self.energy = [None] * num_agents
        self.entropy = [None] * num_agents
        self.EFE = torch.zeros(num_actions)  # EFE for each possible action
        self.ambiguity = torch.zeros(num_actions) #for each possible action
        self.risk = torch.zeros(num_actions) #for each possible action
        self.salience = torch.zeros(num_actions) #for each possible action
        self.pragmatic_value = torch.zeros(num_actions)
        self.novelty = torch.zeros(num_actions)

        self.expected_EFE = [None]  # Expected EFE averaged over my expected 

    # ========================================
    # Summaries
    # =======================================

    def __repr__(self):
        return (f"Agent(id={self.id!r}, beta_1={self.beta_1}, decay={self.decay}, "
                f"gamma={self.gamma:.3f}, game_matrix_shape={self.game_matrix.shape}, "
                f"action={self.action}, VFE={self.VFE}, expected_EFE={self.expected_EFE})")

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
                f"Current Action: {self.action}\n"
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
    def infer_state(self, o, learning_rate=1e-2, num_iterations=100, num_samples=100):
        '''
        Infer the hidden state of each agent (factor_idx) (i.e. the probability 
        distribution over actions of each agent) given the observation `o` 
        (i.e. the action taken by each agent).

        Employs self.A (observation model) and self.B (transition model),
        and updates the variational parameters self.alpha for each agent
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
        for factor_idx in range(len(self.s)):  # Loop over each state factor (me + the other agents)
            s_prev = self.s[factor_idx].clone().detach()  # State t-1
            assert torch.allclose(s_prev.sum(), torch.tensor(1.0)), "s_prev tensor does not sum to 1."
            log_prior = torch.log(self.B[factor_idx, self.action] @ s_prev + 1e-9)  # New prior is old posterior
            log_likelihood = torch.log(self.A[factor_idx].T @ o[factor_idx] + 1e-9)  # Probability of observation given hidden states

            variational_params = self.alpha[factor_idx].clone().detach().requires_grad_(True)  # Variational Dirichlet distribution for each factor (agent)
            optimizer = torch.optim.Adam([variational_params], lr=learning_rate)

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
            self.s[factor_idx] = Dirichlet(variational_params).mean.detach()  # Variational posterior
            self.alpha[factor_idx] = variational_params.detach()  # Store variational parameters (for next timestep)
            self.VFE[factor_idx] = VFE.detach()
            
            # Compute additional metrics (for validation and plotting)
            entropy = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
            energy = -torch.sum(s_samples * (log_prior + log_likelihood), dim=-1).mean().detach()
            accuracy = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
            complexity = -torch.sum(s_samples * (log_prior - log_s), dim=-1).mean().detach()
            assert torch.allclose(energy - entropy, VFE, atol=1e-6), "Assertion failed: energy + entropy does not equal VFE"
            assert torch.allclose(complexity - accuracy, VFE, atol=1e-6), "Assertion failed: complexity - accuracy does not equal VFE"
            self.entropy[factor_idx] = entropy
            self.energy[factor_idx] = energy
            self.accuracy[factor_idx] = accuracy
            self.complexity[factor_idx] = complexity

        # Data collection (for learning and plotting)
        self.s_history.append(self.s)
        self.o_history.append(o)
        
        return self.s

    # ========================================
    # Action 
    # =======================================
    def compute_efe(self):
        '''
        Compute the Expected Free Energy (EFE) for each possible action
        
        Returns:
            EFE (torch.Tensor): Expected Free Energy for each possible action
        '''
        n_agents = self.A.shape[0]   # Number of agents (including self)
        n_actions = self.A.shape[-1]  # Number of actions
        self.q_s_u = torch.empty((n_actions, n_agents, n_actions))  # q(s|u_i) posterior predictive state distribution (for each factor) conditional on action u_i
        # self.q_o_u = torch.empty((n_actions, n_agents, n_actions))  # q(o|u_i) posterior predictive observation distribution (for each factor) conditional on action u_i
        # (TODO: maybe self.q_o_u should be (n_agents, n_actions, n_actions) - not too important though) 
        
        EFE = torch.zeros(n_actions)  # n-action length vector of zeros
        ambiguity = torch.zeros(n_actions)  # n-action length vector of zeros
        risk = torch.zeros(n_actions)  # n-action length vector of zeros
        salience = torch.zeros(n_actions)  # n-action length vector of zeros
        pragmatic_value = torch.zeros(n_actions)  # n-action length vector of zeros
        
        novelty = torch.zeros((n_actions,))
        
        # For each action
        for u_i in range(n_actions):

            # -----------------------------------------------------------
            # Compute predictive joint state posterior
            
            # Create the einsum subscripts string dynamically for n_agents
            # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
            einsum_str = (
                ','.join([chr(105 + i) for i in range(n_agents)]) 
                + '->' 
                + ''.join([chr(105 + i) for i in range(n_agents)])
            )
            q_s_u = torch.stack(
                [self.B[factor_idx, u_i] @ self.s[factor_idx]  # Predicted state q(s' | s, u_i) for each factor (agent)
                for factor_idx in range(n_agents)]
            )
            q_s_joint_u = torch.einsum(einsum_str, *[q_s_u[i] for i in range(n_agents)])
            assert q_s_joint_u.shape == (n_actions, ) * (n_agents), f"q_s_joint_u shape {q_s_joint_u.shape} is not correct."
            
            # -----------------------------------------------------------
            # Compute predictive joint observation posterior
            indices_left = list(range(n_agents, n_agents*2))  # [3, 4, 5]
            indices_right = list(range(n_agents))  # [3, 4, 5]
            q_o_joint_u = torch.tensordot(
                self.A_joint,  # (2, 2, 2, 2, 2, 2)
                q_s_joint_u,   #          (2, 2, 2) 
                dims=(indices_left, indices_right)
            )  # (2, 2, 2)
            q_o_joint_u = q_o_joint_u / q_o_joint_u.sum()  # Normalise to a probability distribution
            assert q_o_joint_u.shape == (n_actions, ) * n_agents, f"q_o_joint_u shape {q_o_joint_u.shape} is not correct."
            assert torch.allclose(q_o_joint_u.sum(), torch.tensor(1.0)), "q_o_joint_u tensor does not sum to 1."
            
            # For each factor, the expected value is the value of the states (log C), multiplied b y the action probabilities of the opponent
            for factor_idx in range(n_agents):

                H = -torch.diag(self.A[factor_idx] @ torch.log(self.A[factor_idx] + 1e-9))  # Conditional (pseudo?) entropy (of the generated emissions matrix)
                assert H.ndimension() == 1, "log_C_modality (F0) is not a 1-dimensional tensor"
                
                ### ==== MY PREFERENCES OVER MY ACTIONS === ### -  [What I wish to observe me doing, given what I expect they will do]
                if factor_idx == 0:

                    # Posterior predictive state for ego is E_{q(s)}[p(s'|s, u_i)]
                    s_pred = q_s_u[factor_idx]  # shape (2, )
                    
                    # Posterior predictive observation for ego is the 
                    # one-hot encoding of the action currently under consideration
                    q_o_u = torch.tensor(
                        [1.0 if _ == u_i else 0.0 for _ in range(n_actions)]
                    )  # shape (2, )

                    # Posterior predictive observation (joint for alters)
                    q_o_others_u = torch.sum(q_o_joint_u, dim=factor_idx)  # shape (2, 2)
                    
                    # This factor's preferences are the expected value of the joint payoffs
                    # under the joint (marginal) posterior predictive observation
                    #   in math: log C^i = E_{q(o_{-i}|u_i)}[log C]
                    #   in code: log_C_modality = E_{q_o_others_u}[self.log_C]
                    indices_left = list(range(1, n_agents))     # [1, ..., n-1] 
                    indices_right = list(range(n_agents - 1))   # [0, ..., n-2]
                    log_C_modality = torch.tensordot(
                        self.log_C,     # (2, 2, 2)
                        q_o_others_u,   #    (2, 2)
                        dims=(indices_left, indices_right)
                    )  # (2, )
                    
                ### ==== MY PREFERENCES OVER THEIR ACTIONS === [what I wish to observe j doing, given what I plan on doing, and what I expect k will do]
                else:  
                    # Predictive posterior state for alters is E_{q(s)}[p(s'|s, u_i)]
                    s_pred = q_s_u[factor_idx]  # shape (2, )

                    # Compute this factor's preferences and posterior predictive state
                    if n_agents == 2:
                        '''
                        Special case for 2 agents: if we were to marginalise there would be no distribution left, so we index straight from the tensors
                        '''
                        log_C_modality = self.log_C[u_i]
                        # q_o_u = TODO
                    elif n_agents > 2:
                        '''
                        General case for n agents: marginalise out the current factor agent to get expected probs for all other agents (not i, not j)
                            Example for n=3 agents (i, j, k): 
                                q(o_{-j}|u_i) = q(o_i=u_i, o_k | u_i) = sum_{u_j} q(o_i, o_j, o_k | u_i)
                        '''
                        
                        # Posterior predictive observation for alters is
                        # the marginal distribution of the joint observation (marginalising all others)
                        # FIXME: or should the ego variable take the value u_i here?
                        dimensions = torch.arange(n_agents)
                        dimensions = torch.cat((dimensions[:factor_idx], dimensions[factor_idx+1:]))  # marginalise out all others (except current factor)
                        q_o_u = torch.sum(q_o_joint_u, dim=tuple(dimensions))  # shape (2, )
                        
                        # Posterior predictive observation (joint for others except current factor)
                        # i.e. marginalise current factor
                        q_o_others_u = torch.sum(q_o_joint_u, dim=factor_idx)  # shape (2, 2)

                        # This factor's preferences are the expected value of the joint payoffs
                        # under the joint (marginal) posterior predictive observation
                        #   in math: log C^i = E_{q(o_{-i}|u_i)}[log C]
                        #   in code: log_C_modality = E_{q_o_others_u}[self.log_C]
                        indices_left = list(range(1, n_agents))     # [1, ..., n-1] 
                        indices_right = list(range(n_agents - 1))   # [0, ..., n-2]
                        log_C_modality = torch.tensordot(
                            self.log_C,     # (2, 2, 2)
                            q_o_others_u,   #    (2, 2)
                            dims=(indices_left, indices_right)
                        )  # (2, )
                        
                assert q_o_u.squeeze().shape == (n_actions, ), f'Wrong shape {q_o_u.squeeze().shape} for q_o_u.'
                
                self.q_s_u[u_i, factor_idx] = s_pred
                assert torch.allclose(s_pred.sum(), torch.tensor(1.0)), f"s_pred (factor {factor_idx}) tensor does not sum to 1."
                assert log_C_modality.ndimension() == 1, f"log_C_modality (factor {factor_idx}) is not a 1-dimensional tensor."
                # assert torch.allclose(torch.exp(log_C_modality).sum(), torch.FloatTensor(n_agents)), "C does not sum to n agents."  # TODO: Legit check for C that each modality is a prob dist? )
            
                # Posterior predictive observation(s) for both factors
                # o_pred = self.A[factor_idx] @ s_pred  # In the earlier version, the predicted observation came from the factor model
                o_pred = q_o_u  # Now the prediced observation comes from the joint model
                assert torch.allclose(o_pred.sum(), torch.tensor(1.0)), f"o_pred (factor {factor_idx}) tensor does not sum to 1."
                assert o_pred.shape == (n_actions, ), f"o_pred (factor {factor_idx}) tensor is not the correct shape."
                # self.q_o_u[u_i, factor_idx] = o_pred
                
                # EFE = Expected ambiguity + risk 
                EFE[u_i] += H @ s_pred + (o_pred @ (torch.log(o_pred + 1e-9) - log_C_modality))

                ambiguity[u_i] += (H @ s_pred) # Ambiguity is conditional entropy of emissions (0)
                risk[u_i] += (o_pred @ (torch.log(o_pred + 1e-9)))  - (o_pred @ log_C_modality) # Risk is negative posterior predictive entropy minus pragmatic value
                salience[u_i] += -(o_pred @ (torch.log(o_pred + 1e-9)))  - (H @ s_pred) # Salience is negative posterior predictive entropy minus ambiguity (0)
                pragmatic_value[u_i] += (o_pred @ log_C_modality) # Pragmatic value is negative cross-entropy

            assert torch.allclose(risk[u_i] + ambiguity[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] risk + ambiguity ({risk[u_i]} + {ambiguity[u_i]}={risk[u_i] + ambiguity[u_i]}) does not equal EFE (={EFE[u_i]})"
            assert torch.allclose(-salience[u_i] - pragmatic_value[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] -salience - pragmatic value (-{salience[u_i]} - {pragmatic_value[u_i]}={-salience[u_i] - pragmatic_value[u_i]}) does not equal EFE (={EFE[u_i]})"
            assert not torch.isnan(EFE[u_i]), f"EFE[{u_i}] is NaN"
        
            # Novelty ----------------------------------------------------------
            # novelty[u_i] = self.compute_A_novelty(u_i)

        EFE = EFE - novelty
        assert not torch.isnan(EFE[u_i]), f"EFE[{u_i}] is NaN"

        # Data collection ------------------------------------------------------
        self.EFE = EFE
        self.ambiguity = ambiguity
        self.risk = risk
        self.salience = salience
        self.pragmatic_value = pragmatic_value
        self.novelty = novelty
        
        return EFE
    
    def compute_novelty_original(self, u_i):
        '''
        Original novelty implementation by Pat, left here for now to be able to 
        compare results with new implementation(s) below
        '''
        n_agents = self.psi_params.dim()  # Number of agents (including self)
        n_actions = self.psi_params.shape[-1]  # Number of actions
        psi = Dirichlet(torch.flatten(self.psi_params))  # Prior over joint actions

        # Compute the joint distribution (incremental Dirichlet concentration) of predicted actions
        joint_s_pred = torch.einsum(
            ','.join(chr(ord('i') + n) for n in range(n_agents)) 
            + '->' 
            + ''.join(chr(ord('i') + n) for n in range(n_agents)),
            *torch.unbind(self.q_s_u[u_i], dim=0))
        
        # Append the joint distribution to the list
        # delta_psi_params[u_i] = joint_s_pred

        psi_pred_params = self.psi_params + joint_s_pred
        psi_pred = Dirichlet(torch.flatten(psi_pred_params))
        novelty_u_i = kl_divergence(psi_pred, psi)
        return novelty_u_i
    
    def compute_novelty_0(self, u_i):
        '''
        First compute psi_prime_params: what psi_params would be if updated 
        with posterior predictive observation (this is the same procedure 
        as in self.bayesian_learning(), except using o_pred instead of o).
        Find the joint action index
        FIXME: instead of argmax as if o_pred was one-hot, we should compute an expectation/soft update
        '''
        ETA = 1  # FIXME: hardcoded ETA
        n_agents = self.psi_params.dim()  # Number of agents (including self)
        n_actions = self.psi_params.shape[-1]  # Number of actions

        joint_action_idx = [torch.argmax(self.q_o_u[u_i, factor_idx]).item() for factor_idx in range(n_agents)]
        joint_action_idx = tuple(joint_action_idx)
        
        # Update based on observed action
        psi_prime_params = self.psi_params.clone()
        psi_prime_params[joint_action_idx] += ETA  
        psi_prime_params *= self.decay
        psi_prime_params += 1e-9

        psi_given_o = Dirichlet(torch.flatten(psi_prime_params))
        psi = Dirichlet(torch.flatten(self.psi_params))
        novelty_u_i = kl_divergence(psi_given_o, psi)

        return novelty_u_i
    
    def compute_novelty_1(self, u_i):
        '''A brute force approach that won't scale because it is exponential in the number of agents'''
        
        ETA = 1  # FIXME: hardcoded ETA
        n_agents = self.psi_params.dim()  # Number of agents (including self)
        n_actions = self.psi_params.shape[-1]  # Number of actions
        psi = Dirichlet(torch.flatten(self.psi_params))  # Prior over joint actions
            
        # Compute the joint distribution (incremental Dirichlet concentration) of predicted actions
        # i.e. what is the probability that we will see (o_1, o_2, ..., o_n) if we take action u_i
        joint_o_pred = torch.einsum(
            ','.join(chr(ord('i') + n) for n in range(n_agents)) 
            + '->' 
            + ''.join(chr(ord('i') + n) for n in range(n_agents)),
            *torch.unbind(self.q_o_u[u_i], dim=0))

        # Enumerate all possible joint actions (ccc, ccd, cdc, ..., ddd)
        all_possible_joint_actions = torch.cartesian_prod(
            *[torch.arange(n_actions) for _ in range(n_agents)]
        )
        # Loop over all possible joint actions (FIXME this is exponential in the number of agents)
        novelty_u_i = 0
        for joint_action in all_possible_joint_actions:
            # What would our Psi look like if we updated it with this joint action?
            psi_pred_params = self.psi_params.clone()
            psi_pred_params[tuple(joint_action)] += ETA
            psi_pred_params *= self.decay
            psi_pred_params += 1e-9
            psi_pred = Dirichlet(torch.flatten(psi_pred_params))
            # novelty = E_{q(o|u)}[KL(q(psi|o) || p(psi))]
            # but here we do the sum over o explicitly instead of the expectation
            novelty_u_i += joint_o_pred[tuple(joint_action)] * kl_divergence(psi_pred, psi)

        return novelty_u_i
    
    def compute_novelty_2(self, u_i):
        '''
        A brute force approach that won't scale because it is exponential in the number of agents
        
        Here Psi is conditioned on u_i
        '''
        
        ETA = 1  # FIXME: hardcoded ETA
        n_agents = self.psi_params.dim()  # Number of agents (including self)
        n_actions = self.psi_params.shape[-1]  # Number of actions
        psi_u = Dirichlet(torch.flatten(self.psi_params[u_i]))  # Prior over joint actions given my action
        # TODO: or should psi_u be obtained similarly to:
        #        conditional_joint_prior = self.psi_params / self.psi_params.sum(dim=list(range(1, n_agents)), keepdim=True)  # p(u_{-i} | u_i)

        # Compute the joint distribution (incremental Dirichlet concentration) of predicted actions
        # i.e. what is the probability that we will see (o_1, o_2, ..., o_n) if we take action u_i
        # FIXME: will not work if n_agents == 2
        joint_o_pred_u = torch.einsum(
            ','.join(chr(ord('i') + n) for n in range(n_agents-1)) 
            + '->' 
            + ''.join(chr(ord('i') + n) for n in range(n_agents-1)),
            *torch.unbind(self.q_o_u[u_i, 1:], dim=0))

        # Enumerate all possible joint actions (cc, cd, dc, dd)
        # of OPPONENTS (i.e. not including the ego action, which is assumed to be u_i here)
        all_possible_joint_actions = torch.cartesian_prod(
            *[torch.arange(n_actions) for _ in range(n_agents-1)]
        )
        # Loop over all possible joint actions (FIXME this is exponential in the number of agents)
        novelty_u_i = 0
        for joint_action in all_possible_joint_actions:
            # What would our Psi look like if we updated it with this joint action?
            psi_pred_params = self.psi_params[u_i].clone()
            psi_pred_params[tuple(joint_action)] += ETA
            psi_pred_params *= self.decay
            psi_pred_params += 1e-9
            psi_pred_u = Dirichlet(torch.flatten(psi_pred_params))
            # novelty = E_{q(o|u)}[KL(q(psi|o,u) || p(psi|u))]
            # but here we do the sum over o explicitly instead of the expectation
            novelty_u_i += joint_o_pred_u[tuple(joint_action)] * kl_divergence(psi_pred_u, psi_u)

        return novelty_u_i

    def compute_A_novelty(self, u_i):
        '''
        Compute the novelty of the likelihood model A for action u_i
        '''
        novelty = 0
        # Da Costa et al. (2020; Eq. D.17)
        W = 0.5 * (1/self.A_params - 1/self.A_params.sum(dim=1, keepdim=True))
        for factor_idx in range(len(self.s)):
            s_pred = self.q_s_u[u_i, factor_idx]
            novelty += torch.dot(
                self.A[factor_idx] @ s_pred, 
                W[factor_idx] @ s_pred) 
        return novelty

    def select_action(self):
        EFE = self.compute_efe()
        q_u = torch.softmax(torch.log(self.E) - self.gamma * EFE, dim=0)
        assert torch.allclose(q_u.sum(), torch.tensor(1.0)), (
            "q_u policy tensor does not sum to 1.",
            f"q_u: {q_u}",
            f"q_u.sum(): {q_u.sum()}",
            f"EFE: {EFE}"
        )

        self.action = torch.multinomial(q_u, 1).item()

        if self.dynamic_precision:
            self.update_precision(EFE, q_u)
        self.q_u = q_u

        self.u_history.append(self.action)

        return self.action

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
    
        self.learn_A()
        self.learn_A_joint()
        # self.learn_B()

        # Reset history
        self.s_history = []
        self.o_history = []
        self.u_history = []

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
        BMR = True
        if BMR:
            shrinkage = self.decay
            mixture = 0.8
            # assert 0 < shrinkage < 1, "Shrinkage parameter must be in [0, 1]"
            # A_reduced_params = shrinkage * self.A_params 
            # assert shrinkage >= 1, "Shrinkage parameter must be greater than 1"
            # A_reduced_params = torch.softmax(shrinkage * self.A_params, dim=-2)
            # A_reduced_params = self.A_params ** (1/shrinkage)

            A_identity = torch.stack([
                torch.tensor([[1., 0.], [0., 1.]])
                for _ in range(self.A.shape[0])
            ]).view(*self.A.shape)
            
            A_reduced_params = (
                mixture * self.A_params 
                + (1 - mixture) * A_identity  # torch.softmax(shrinkage * A_identity, dim=-2)
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
        BMR = True
        if BMR:
            shrinkage = self.decay
            # assert shrinkage >= 1, "Shrinkage parameter must be greater than 1"
            B_reduced_params = torch.softmax(shrinkage * self.B_params, dim=-2)

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

    def learn_A_joint(self, o=None):
        '''
        Compute the joint observation likelihood

        Requires the current hidden state distribution (self.s), and
        the joint observation likelihood (self.A_joint) to be updated.
        
        Args:
            o (torch.Tensor): Observation tensor of shape (n_agents, n_actions),
                i.e. one-hot encoding of actions for each agent
        '''
        n_agents = self.A.shape[0]  # Number of agents (including self)
        n_actions = self.A.shape[-1]  # Number of actions
        if o is None:
            T = len(self.s_history) # Number of timesteps
            s_history = self.s_history
            o_history = self.o_history
        else:
            T = 1
            s_history = [self.s] 
            o_history = [o] 


        # Create the einsum subscripts string dynamically for n_agents
        # e.g., if n_agents = 3, this will be 'i,j,k->ijk'
        einsum_str = (
            ','.join([chr(105 + i) for i in range(n_agents)]) 
            + '->' 
            + ''.join([chr(105 + i) for i in range(n_agents)])
        )

        A_joint_posterior = self.A_joint.clone()
        for t in range(T):
            s_t = s_history[t]  # Shape: (n_agents, n_actions)
            o_t = o_history[t]  # Shape: (n_agents, n_actions)

            # Compute joint state prior
            q_s_joint = torch.einsum(einsum_str, *[s_t[factor_idx] for factor_idx in range(n_agents)])
            assert q_s_joint.shape == (n_actions, ) * (n_agents), f"q_s_joint shape {q_s_joint.shape} is not correct."

            o_indices = tuple([torch.argmax(o_t[factor_idx]).item() for factor_idx in range(n_agents)])
            assert self.A_joint[o_indices].shape == q_s_joint.shape, f"A_joint[o_indices] shape {self.A_joint[o_indices].shape} is not correct."
            
            A_joint_posterior[o_indices] += q_s_joint

        # Bayesian Model Reduction ---------------------------------------------
        BMR = False
        if BMR:
            mixture = 0.8
            # A_identity = torch.stack([
            #     torch.tensor([[1., 0.], [0., 1.]])
            #     for _ in range(self.A.shape[0])
            # ]).view(*self.A.shape)
            
            # A_reduced_params = (
            #     mixture * self.A_params 
            #     + (1 - mixture) * A_identity  # torch.softmax(shrinkage * A_identity, dim=-2)
            # )

            # # Update model parameters if they reduce the free energy
            # self.delta_F = []
            # for factor_idx in range(len(self.s)):
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
        else:
            self.A_joint = A_joint_posterior

        # TODO: do we need to normalise or not? If it is a likelihood *function*, no need.


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