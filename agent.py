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
        # self.game_matrix = (game_matrix / game_matrix.sum()) #Input probabilities instead instead of log rewards?
        num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model/variational posterior parameters
        self.alpha = [torch.ones(num_actions) for _ in range(num_agents)]  # Dirichlet state prior
        self.A = [torch.eye(num_actions) for _ in range(num_agents)]  # Identity observation model
        self.B = lambda s, u: s  # Identity transition model (given s and u, return s)
        self.log_C = game_matrix.to(torch.float)  # Payoffs for joint actions (in matrix form, from row player's perspective) (force to float)
        self.prob_C = None
        self.s = [Dirichlet(alpha).mean for alpha in self.alpha]  # Categorical state prior (D?)
        self.E = torch.ones(num_actions) / num_actions  # Habits 
        self.opp_pred = [None] * num_agents

        # self.A_params = torch.eye(2)
        # self.B_params = torch.tensor([torch.eye(2), torch.eye(2)])
        # Opponent modeling (what I think their preferences are, currently unused - no learning)
        # self.C_opp_params = self.log_C.T.clone().detach()    ### <--------------- TEST transpose log_C as if ego knew exactly alters preferences
        self.C_opp_params = torch.ones_like(self.log_C)    ### <--------------- TEST uniform prior for opponent preferences
        
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
        formatted_opponent_params = format_tensor(self.C_opp_params)

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
                f"Log C Opp Params (ToM Payoffs):\n{formatted_opponent_params}\n"

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
        for factor_idx in range(len(self.s)):  # Loop over each state factor (me + the other agents)
            s_prev = self.s[factor_idx].clone().detach()  # State t-1
            assert torch.allclose(s_prev.sum(), torch.tensor(1.0)), "s_prev tensor does not sum to 1."
            log_prior = torch.log(self.B(s_prev, self.action) + 1e-9)  # New prior is old posterior
            log_likelihood = torch.log(self.A[factor_idx] @ o[factor_idx] + 1e-9)  # Probability of observation given hidden states

            variational_params = self.alpha[factor_idx].clone().detach().requires_grad_(True)  # Variational Dirichlet distribution for each factor (agent)
            optimizer = torch.optim.Adam([variational_params], lr=learning_rate)

            for _ in range(num_iterations):
                optimizer.zero_grad()

                s_samples = Dirichlet(variational_params).rsample((num_samples,))  # Variational dist samples
                log_s = torch.log(s_samples + 1e-9)  # Log variational dist samples
                vfe_samples = torch.sum(s_samples * (log_s - log_likelihood - log_prior), dim=-1) 
                VFE = vfe_samples.mean()

                entropy = -torch.sum(s_samples.detach() * log_s.detach(), dim=-1).mean()
                energy = -torch.sum(s_samples.detach() * (log_prior.detach() + log_likelihood.detach()), dim=-1).mean()
                accuracy = torch.sum(s_samples.detach() * log_likelihood.detach(), dim=-1).mean()
                complexity = -torch.sum(s_samples.detach() * (log_prior.detach() - log_s.detach()), dim=-1).mean()

                assert torch.allclose(energy - entropy, VFE, atol=1e-6), "Assertion failed: energy + entropy does not equal VFE"
                assert torch.allclose(complexity - accuracy, VFE, atol=1e-6), "Assertion failed: complexity - accuracy does not equal VFE"

                VFE.backward()
                optimizer.step()
                variational_params.data.clamp_(min=1e-3)

            self.alpha[factor_idx] = variational_params.detach()
            self.s[factor_idx] = Dirichlet(variational_params).mean.detach()
            self.VFE[factor_idx] = VFE.detach()
            
            self.entropy[factor_idx] = entropy
            self.energy[factor_idx] = energy
            self.accuracy[factor_idx] = accuracy
            self.complexity[factor_idx] = complexity

        return self.s

    # ========================================
    # Action 
    # =======================================
    def compute_efe(self):
        num_actions = self.E.shape[0]  # Scalar (number of actions)
        EFE = torch.zeros(num_actions)  # n-action length vector of zeros
        ambiguity = torch.zeros(num_actions)  # n-action length vector of zeros
        risk = torch.zeros(num_actions)  # n-action length vector of zeros
        salience = torch.zeros(num_actions)  # n-action length vector of zeros
        pragmatic_value = torch.zeros(num_actions)  # n-action length vector of zeros
        novelty = torch.zeros(num_actions)  # n-action length vector of zeros

        n_agents = self.C_opp_params.dim()  # Number of agents (including self)

        # For each action
        for action in torch.arange(num_actions):
            # For each factor, the expected value is the value of the states (log C), multiplied by the action probabilities of the opponent
            for factor_idx in range(len(self.s)):
                H = -torch.diag(self.A[factor_idx] @ torch.log(self.A[factor_idx] + 1e-9))  # Conditional (pseudo?) entropy (of the generated emissions matrix) - ZERO
                assert H.ndimension() == 1, "log_C_modality (F0) is not a 1-dimensional tensor"
                
                ### ==== MY PREFERENCES OVER MY ACTIONS === ### -  [What I wish to observe me doing, given what I expect they will do]
                if factor_idx == 0:  
                    #My action - 
                    s_pred = self.B(self.s[factor_idx], action)  # Predicted state q(s | s, u)
                    assert torch.allclose(s_pred.sum(), torch.tensor(1.0)), "s_pred (F0) tensor does not sum to 1."
                    assert torch.allclose(s_pred, self.s[factor_idx], atol=1e-6), "s_pred (F0) does not equal my last fictitious play estimate"
                    #Should the above just use my actual action?
                    
                    # PPS 
                    # For each of my actions, what are the probabilities of the other agents action combos? e.g. p(CC | me = C), p(CD | me = C), p(DC | me = C), etc.
                    expected_probs = self.C_opp_params / self.C_opp_params.sum(dim=list(range(1, n_agents)), keepdim=True)   # p(u_{-i} | u_i)
                    assert expected_probs.ndimension() == n_agents, "Expected joint actions (F0) is not an n-agent dimensional tensor"
                    assert torch.prod(torch.tensor(expected_probs.shape[:-1])) == num_actions ** (n_agents-1), "# Expected joint action (F0) vectors != num_actions^(n_agents - 1)"
                    assert torch.allclose(expected_probs.sum(), torch.tensor(float(num_actions))), "Expected joint action probs (F0) do not sum to num actions."
                    assert torch.allclose(expected_probs[action].sum(), torch.tensor(1.0)), "Expected probs[action] (F0) tensor does not sum to 1."
                    # This one isn't technically conditional probabilities, so not quite clocking it.

                    # Indices for tensor dot product
                    indices_left = list(range(1, n_agents))     # [1, ..., n-1] 
                    indices_right = list(range(n_agents - 1))   # [0, ..., n-2]
                    
                    #Multiply joint payoffs by probability of joint action
                    log_C_modality = torch.tensordot(
                        self.log_C,  # (2, 2, 2)
                        expected_probs[action],  # (2, 2)
                        dims=(indices_left, indices_right)
                    )
                    assert log_C_modality.ndimension() == 1, "log_C_modality (F0) is not a 1-dimensional tensor."
                    
                    
                ### ==== MY PREFERENCES OVER THEIR ACTIONS === [what I’d wish to observe them doing, were I to do action u, given my model of their preferences]
                else:  

                    n_agents = self.C_opp_params.dim()  # Number of agents (including self)

                    # #For each of my actions, what are the probabilities of the other agents action combos? e.g. p(CC | me = C), p(CD | me = C), p(DC | me = C), etc.
                    expected_probs = self.C_opp_params / self.C_opp_params.sum(dim=list(range(1, n_agents)), keepdim=True)  # p(u_{-i} | u_i)
                    assert expected_probs.ndimension() == n_agents, "Expected joint actions (F_j) is not an n-agent dimensional tensor"
                    assert torch.prod(torch.tensor(expected_probs.shape[:-1])) == num_actions ** (n_agents-1), "# Expected joint action (F_j) vectors != num_actions^(n_agents - 1)"
                    assert torch.allclose(expected_probs.sum(), torch.tensor(float(num_actions))), "Expected joint action probs (F_j) do not sum to num actions."
                    assert torch.allclose(expected_probs[action].sum(), torch.tensor(1.0)), "Expected probs[action] (F_j) tensor does not sum to 1."

                    # Marginalise out the current factor agent to get expected probs for all other agents (not i, not j)
                    if n_agents == 2:
                        expected_probs_marginal = expected_probs 
                    else:
                        expected_probs_marginal = expected_probs.sum(dim=factor_idx, keepdim=True)  # p(u_{-j-i}|u_i) = sum_{u_j} p(u_{-i} | u_i)
                    
                    # Indices for tensor dot product
                    indices_left = list(range(n_agents-1))      # [0, ..., n-2]
                    indices_left.remove(factor_idx-1)           # Remove the current factor index  [1, ..., n-1] \ j
                    indices_right = list(range(n_agents - 2))   # [0, ..., n-3]  because we've removed both i (ego; conditional) and j (current factor; marginalised)

                    if n_agents == 2:
                        log_C_modality = self.log_C[action] * expected_probs_marginal[action]
                    else:
                        log_C_modality = torch.tensordot(
                            self.log_C[action], # (2, 2)
                            expected_probs_marginal[action].squeeze(),   # (2,)
                            dims=(indices_left, indices_right)
                        )
                    assert log_C_modality.ndimension() == 1, "log_C_modality (F_j) is not a 1-dimensional tensor."

                    if n_agents == 2:
                        s_pred = expected_probs_marginal[action].squeeze() #Original which was throwing an error
                        assert torch.allclose(s_pred.sum(), torch.tensor(1.0)), "s_pred (F_j) tensor does not sum to 1."
                    else:
                        s_pred = expected_probs.sum(dim=indices_right)[action]
                        s_pred = s_pred / s_pred.sum() #Legit to normalise this? 
                        assert torch.allclose(s_pred.sum(), torch.tensor(1.0)), "s_pred (F_j) tensor does not sum to 1."

                    
            
                # # Posterior predictive observation(s) for both factors: THIS SHOULD BE A VECTOR EACH TIME
                o_pred = self.A[factor_idx].T @ s_pred
                assert torch.allclose(o_pred.sum(), torch.tensor(1.0)), "o_pred (F_j) tensor does not sum to 1."

                assert log_C_modality.ndimension() == 1, "log_C_modality (main) is not a 1-dimensional tensor."

                # EFE = Expected ambiguity + risk 
                EFE[action] += H @ s_pred + (o_pred @ (torch.log(o_pred + 1e-9) - log_C_modality))

                ambiguity[action] += (H @ s_pred) # Ambiguity is conditional entropy of emissions (0)
                risk[action] += (o_pred @ (torch.log(o_pred + 1e-9)))  - (o_pred @ log_C_modality) # Risk is negative posterior predictive entropy minus pragmatic value
                salience[action] += -(o_pred @ (torch.log(o_pred + 1e-9)))  - (H @ s_pred) # Salience is negative posterior predictive entropy minus ambiguity (0)
                pragmatic_value[action] += (o_pred @ log_C_modality) # Pragmatic value is negative cross-entropy
       
        #Summed over each factor (for each action)
        self.EFE = EFE
        self.ambiguity = ambiguity
        self.risk = risk
        self.salience = salience
        self.pragmatic_value = pragmatic_value
        # self.novelty = novelty

        assert torch.allclose(risk + ambiguity, EFE, atol=1e-6), "Risk + ambiguity does not equal EFE"
        assert torch.allclose(-salience - pragmatic_value, EFE, atol=1e-6), "-Salience - pragmatic_value does not equal EFE"

        
        return EFE

    def select_action(self):
        EFE = self.compute_efe()
        q_pi = torch.softmax(torch.log(self.E) - self.gamma * EFE, dim=0)
        assert torch.allclose(q_pi.sum(), torch.tensor(1.0)), "q_pi policy tensor does not sum to 1."

        self.action = torch.multinomial(q_pi, 1).item()

        if self.dynamic_precision:
            self.update_precision(EFE, q_pi)
        self.q_pi = q_pi
        return self.action

    def update_precision(self, EFE, q_pi):
        # Compute the expected EFE as a scalar value
        self.expected_EFE = [torch.dot(q_pi, EFE).item()]
        
        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.beta_1 / (self.beta_0 - self.expected_EFE[0])
        
        return self.gamma

    # ========================================
    # Learning p(a_j | a_i)
    # =======================================

    # Define the n-player generalized function
    def bayesian_learning(self, o, eta=2):
        
        # Find the joint action index
        joint_action_idx = [torch.argmax(o[agent]).item() for agent in range(self.C_opp_params.dim())]
        joint_action_idx = tuple(joint_action_idx)
        
        # Update based on observed action
        self.C_opp_params[joint_action_idx] += eta / self.C_opp_params.numel()

        # Temporal discounting
        self.C_opp_params *= self.decay
        self.C_opp_params += 1e-9
        
        return self.C_opp_params
