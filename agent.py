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
        # self.game_matrix = (game_matrix / game_matrix.sum()) #Input probabilities instead instead of log rewards?
        num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model/variational posterior parameters
        self.alpha = [torch.ones(num_actions) for _ in range(num_agents)]  # Dirichlet state prior
        self.A_params = torch.ones((num_agents, num_actions, num_actions))
        self.A_params = torch.stack([torch.eye(num_actions) for _ in range(num_agents)]) + 1e-9  # Identity observation model
        # self.A = [torch.eye(num_actions) for _ in range(num_agents)]  # Identity observation model
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)
        # self.B = lambda s, u: s  # Identity transition model (given s and u, return s)
        # self.B_params = torch.ones((num_agents, num_actions, num_actions, num_actions))
        # self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)
        self.B = torch.stack([
            torch.eye(num_actions) 
            for _ in range(num_actions) 
            for _ in range(num_agents)
        ]).reshape(num_agents, num_actions, num_actions, num_actions)
        self.log_C = game_matrix.to(torch.float)  # Payoffs for joint actions (in matrix form, from row player's perspective) (force to float)
        self.prob_C = None
        self.s = [Dirichlet(alpha).mean for alpha in self.alpha]  # Categorical state prior (D?)
        self.E = torch.ones(num_actions) / num_actions  # Habits 
        # self.q_s_u = [None] * num_agents
        # self.psi_pred = [None] * num_agents

        # self.A_params = torch.eye(2)
        # self.B_params = torch.tensor([torch.eye(2), torch.eye(2)])
        # Opponent modeling (what I think their preferences are, currently unused - no learning)
        # Store blanket states history for learning
        self.s_history = []
        self.o_history = []
        self.u_history = []
        # self.C_opp_params = self.log_C.T.clone().detach()    ### <--------------- TEST transpose log_C as if ego knew exactly alters preferences
        self.psi_params = torch.ones_like(self.log_C)    ### <--------------- TEST uniform prior for opponent preferences
        
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
        formatted_opponent_params = format_tensor(self.psi_params)

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

        self.s_history.append(torch.stack(self.s))
        self.o_history.append(o)
        
        return self.s

    # ========================================
    # Action 
    # =======================================
    def compute_efe(self):
        n_actions = self.E.shape[0]  # Scalar (number of actions)
        n_agents = self.psi_params.dim()  # Number of agents (including self)
        self.q_s_u = torch.empty((n_actions, n_agents, n_actions))  # q(s|u_i) posterior predictive state distribution (for each factor) conditional on action u_i
        self.q_o_u = torch.empty((n_actions, n_agents, n_actions))  # q(o|u_i) posterior predictive observation distribution (for each factor) conditional on action u_i
        # (TODO: maybe self.q_o_u should be (n_agents, n_actions, n_actions) - not too important though) 
        
        EFE = torch.zeros(n_actions)  # n-action length vector of zeros
        ambiguity = torch.zeros(n_actions)  # n-action length vector of zeros
        risk = torch.zeros(n_actions)  # n-action length vector of zeros
        salience = torch.zeros(n_actions)  # n-action length vector of zeros
        pragmatic_value = torch.zeros(n_actions)  # n-action length vector of zeros
        
        novelty = torch.zeros((n_actions,))

        # For each action
        for u_i in range(n_actions):
            # For each factor, the expected value is the value of the states (log C), multiplied by the action probabilities of the opponent
            for factor_idx in range(n_agents):
                H = -torch.diag(self.A[factor_idx] @ torch.log(self.A[factor_idx] + 1e-9))  # Conditional (pseudo?) entropy (of the generated emissions matrix) - ZERO
                assert H.ndimension() == 1, "log_C_modality (F0) is not a 1-dimensional tensor"
                
                # p(u_{-i} | u_i)
                # For each of my actions, what are the probabilities of the other agents action combos? e.g. p(CC | me = C), p(CD | me = C), p(DC | me = C), etc.
                conditional_joint_prior = self.psi_params / self.psi_params.sum(dim=list(range(1, n_agents)), keepdim=True)  # p(u_{-i} | u_i)
                assert conditional_joint_prior.ndimension() == n_agents, "Expected joint actions (F0) is not an n-agent dimensional tensor"
                assert torch.prod(torch.tensor(conditional_joint_prior.shape[:-1])) == n_actions ** (n_agents-1), "# Expected joint action (F0) vectors != num_actions^(n_agents - 1)"
                assert torch.allclose(conditional_joint_prior.sum(), torch.tensor(float(n_actions))), "Expected joint action probs (F0) do not sum to num actions."
                assert torch.allclose(conditional_joint_prior[u_i].sum(), torch.tensor(1.0)), "Expected probs[action] (F0) tensor does not sum to 1."
                
                ### ==== MY PREFERENCES OVER MY ACTIONS === ### -  [What I wish to observe me doing, given what I expect they will do]
                if factor_idx == 0:  
                    # Hidden state prediction
                    s_pred = self.B[factor_idx, u_i] @ self.s[factor_idx]  # Predicted state q(s | s, u)
                    # TODO: This is placeholder for when B is an actual matrix
                    # TODO: but maybe we could use q(u) (i.e. self.q_u) here

                    # Indices for tensor dot product
                    indices_left = list(range(1, n_agents))     # [1, ..., n-1] 
                    indices_right = list(range(n_agents - 1))   # [0, ..., n-2]
                    
                    #Multiply joint payoffs by probability of joint action
                    log_C_modality = torch.tensordot(
                        self.log_C,  # (2, 2, 2)
                        conditional_joint_prior[u_i],  # (2, 2)
                        dims=(indices_left, indices_right)
                    )
                    
                ### ==== MY PREFERENCES OVER THEIR ACTIONS === [what I wish to observe j doing, given what I plan on doing, and what I expect k will do]
                else:  

                    # Compute this factor's preferences and posterior predictive state
                    if n_agents == 2:
                        '''
                        Special case for 2 agents: if we were to marginalise there would be no distribution left, so we index straight from the tensors
                        '''
                        log_C_modality = self.log_C[u_i]
                        s_pred = conditional_joint_prior[u_i].squeeze()
                    elif n_agents > 2:
                        '''
                        General case for n agents: marginalise out the current factor agent to get expected probs for all other agents (not i, not j)
                            Example for n=3 agents (i, j, k): 
                                p(u_{-j-i}|u_i) = p(u_k | u_i) = sum_{u_j} p(u_j, u_k | u_i) = sum_{u_j} p(u_{-i} | u_i)
                        '''
                        # Marginalise out the current factor agent to get expected probs for all other agents (not i, not j)
                        conditional_joint_prior_marginal = conditional_joint_prior[u_i].sum(dim=factor_idx-1, keepdim=True)  # p(u_{-j-i}|u_i) = sum_{u_j} p(u_{-i} | u_i)
                    
                        # Indices for tensor dot product
                        indices_left = list(range(n_agents-1))      # [0, ..., n-2]
                        indices_left.remove(factor_idx-1)           # Remove the current factor index  [1, ..., n-1] \ j
                        indices_right = list(range(n_agents - 2))   # [0, ..., n-3]  because we've removed both i (ego; conditional) and j (current factor; marginalised)

                        log_C_modality = torch.tensordot(
                            self.log_C[u_i], # (2, 2)
                            conditional_joint_prior_marginal.squeeze(),   # (2,)
                            dims=(indices_left, indices_right)
                        )

                        # Compute the posterior predictive state for the current factor agent
                        indices = list(range(n_agents-1))   # [0, ..., n-1]  because we've removed i (ego; conditional)
                        indices.remove(factor_idx-1)        # Remove the current factor index  [0, ..., n-1] \ (j-1)  --> (j-1) because we've removed i (ego; conditional)
                        s_pred = conditional_joint_prior[u_i].sum(dim=indices)  # p(u_j | u_i) = sum_{u_{-i-j}} p(u_j, u_{-i-j} | u_i)
                        
                self.q_s_u[u_i, factor_idx] = s_pred
                assert torch.allclose(s_pred.sum(), torch.tensor(1.0)), f"s_pred (factor {factor_idx}) tensor does not sum to 1."
                assert log_C_modality.ndimension() == 1, f"log_C_modality (factor {factor_idx}) is not a 1-dimensional tensor."
                # assert torch.allclose(torch.exp(log_C_modality).sum(), torch.FloatTensor(n_agents)), "C does not sum to n agents."  # TODO: Legit check for C that each modality is a prob dist? 
            
                # Posterior predictive observation(s) for both factors
                o_pred = self.A[factor_idx] @ s_pred
                assert torch.allclose(o_pred.sum(), torch.tensor(1.0)), f"o_pred (factor {factor_idx}) tensor does not sum to 1."
                assert o_pred.shape == (n_actions, ), f"o_pred (factor {factor_idx}) tensor is not the correct shape."
                self.q_o_u[u_i, factor_idx] = o_pred
                
                # EFE = Expected ambiguity + risk 
                EFE[u_i] += H @ s_pred + (o_pred @ (torch.log(o_pred + 1e-9) - log_C_modality))

                ambiguity[u_i] += (H @ s_pred) # Ambiguity is conditional entropy of emissions (0)
                risk[u_i] += (o_pred @ (torch.log(o_pred + 1e-9)))  - (o_pred @ log_C_modality) # Risk is negative posterior predictive entropy minus pragmatic value
                salience[u_i] += -(o_pred @ (torch.log(o_pred + 1e-9)))  - (H @ s_pred) # Salience is negative posterior predictive entropy minus ambiguity (0)
                pragmatic_value[u_i] += (o_pred @ log_C_modality) # Pragmatic value is negative cross-entropy

            assert torch.allclose(risk[u_i] + ambiguity[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] risk + ambiguity ({risk[u_i]} + {ambiguity[u_i]}={risk[u_i] + ambiguity[u_i]}) does not equal EFE (={EFE[u_i]})"
            assert torch.allclose(-salience[u_i] - pragmatic_value[u_i], EFE[u_i], atol=1e-4), f"[u_i = {u_i}] -salience - pragmatic value (-{salience[u_i]} - {pragmatic_value[u_i]}={-salience[u_i] - pragmatic_value[u_i]}) does not equal EFE (={EFE[u_i]})"
        
            # Novelty ----------------------------------------------------------
            # novelty[u_i] = self.compute_A_novelty(u_i)

        EFE = EFE - novelty

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
        self.expected_EFE = [torch.dot(q_u, EFE).item()]
        
        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.beta_1 / (self.beta_0 - self.expected_EFE[0])
        
        return self.gamma

    # ========================================
    # Learning p(u_j | u_i)
    # =======================================

    # Define the n-player generalized function
    def bayesian_learning(self, o, eta=1):
        
        # Find the joint action index
        joint_action_idx = [torch.argmax(o[agent]).item() for agent in range(self.psi_params.dim())]
        joint_action_idx = tuple(joint_action_idx)
        
        # Update based on observed action
        self.psi_params[joint_action_idx] += eta #/ self.C_opp_params.numel()

        # Temporal discounting
        self.psi_params *= self.decay
        self.psi_params += 1e-9
        
        return self.psi_params
    
    def learn(self):

        # Convert history to tensors
        self.s_history = torch.stack(self.s_history)  # Shape: (T, n_agents, n_actions)
        self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
        self.u_history = torch.tensor(self.u_history)  # Shape: (T, )
    
        self.learn_A()
        # self.learn_B()
        # self.learn_Psi()

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
            shrinkage = 1.1
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
                + (1 - mixture) * torch.softmax(shrinkage * A_identity, dim=-2)
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

        # Update parameters for every transition (s, u, s') in the history
        for t in range(outer_products.shape[0]):
            # Likelihood parameters update
            delta_params = outer_products[t]  # Shape: (n_agents, n_actions, n_actions)
            u_it = self.u_history[t].item()   # Action u_i at time t
            self.B_params[:, u_it] = self.B_params[:, u_it] + delta_params

        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)

    def learn_Psi(self, eta=1):

        psi_posterior_params = self.psi_params.clone()

        for o in self.o_history:
        
            # Find the joint action index
            joint_action_idx = [torch.argmax(o[agent]).item() for agent in range(self.psi_params.dim())]
            joint_action_idx = tuple(joint_action_idx)
            
            # Update based on observed action
            psi_posterior_params[joint_action_idx] += eta #/ self.C_opp_params.numel()

        # Bayesian Model Reduction ---------------------------------------------
        # shrinkage = 0.8
        # psi_reduced_params = torch.softmax(
        #     (1/shrinkage) * self.psi_params.flatten(), 
        #     dim=-1).reshape(self.psi_params.shape)
        psi_reduced_params = self.psi_params ** (1/1.3) + 1e-9

        # Update model parameters if they reduce the free energy
        delta_F = delta_free_energy(
            psi_posterior_params.flatten(), 
            self.psi_params.flatten(), 
            psi_reduced_params.flatten()
        )
        if delta_F < 0:
            # Reduced model is preferred -> replace full model with reduced model
            self.psi_params = psi_reduced_params
        else:
            # Full model is preferred -> update posterior
            self.psi_params = psi_posterior_params


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