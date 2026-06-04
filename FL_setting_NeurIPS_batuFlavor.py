import random
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
import heapq
class FederatedLearning:
    def __init__(self, mode, num_users, device, 
                    cos_similarity, model, TrainSetUsers, epochs, optimizer, criteron, fraction, 
                    testloader, learning_rate_server, train_mode, keepProbAvail, keepProbNotAvail, 
                    bufferLimit, theta_inner, unit_gradients, adam, temp, cos_similarity_type):
        
        #Arguements
        self.learning_rate_server = learning_rate_server
        self.epochs = epochs
        self.num_users = num_users
        self.fraction = fraction
        self.mode = mode
        self.cos_similarity = cos_similarity
        self.bufferLimit = bufferLimit
        self.theta_inner = theta_inner
        self.train_mode = train_mode
        self.unit_gradients = unit_gradients
        self.adam = adam

        #Device
        self.device = device 

        #Weights in each user 
        self.w_user = [[param.data.clone().to("cpu") for param in model.parameters()] for _ in range(num_users)]

        #Global Weights
        self.w_global = [param.data.clone().to(self.device) for param in model.parameters()]

        #Sparse gradients of users 
        self.sparse_gradient = [[torch.zeros_like(param).to("cpu") for param in model.parameters()] for _ in range(num_users)]

        #Aggregation buffer
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]

        #Training components 
        self.model = model
        self.optimizer = optimizer
        self.criteron = criteron
        self.TrainSetUsers = TrainSetUsers 
        self.testloader = testloader

        #Intermittent user model
        self.keepProbAvail = keepProbAvail
        self.keepProbNotAvail = keepProbNotAvail
        self.intermittentStateOneHot = np.array([1 if (1-self.keepProbNotAvail[u])/(2-self.keepProbAvail[u]-self.keepProbNotAvail[u]) > random.random() else 0 for u in range(num_users)])
        self.intermittentUsers = np.where(self.intermittentStateOneHot)

        #Age based variables
        self.UserAgeUL = torch.zeros(self.num_users, 1).to("cpu")
        self.UserAgeDL = torch.ones(self.num_users, 1).to("cpu") 
        self.UserAgeMemory = torch.zeros(self.num_users, 1).to("cpu")
        self.allOnes = torch.ones(self.num_users, 1).to("cpu")


        #Inner Product Test variables 
        self.bufferSize = 0
        self.userListUL = set(range(self.num_users))
        self.setAllUsers = set(range(self.num_users))
        self.nu_orthogonal = 5.67 #tan(80)

        #Policy calculation
        self.temperature = temp
        self.pi = self.calculate_policy()
        
        # Tracking variables
        self.contribution = np.zeros((self.num_users, 1))
        self.expected_gradient_magnitude = np.zeros((self.num_users, 1))
        self.num_send = 0

        #Cosine similarity variables
        self.lastGradient = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
        self.cosine_similarity_type = cos_similarity_type

        # Adam parameters
        if self.adam:
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.tau = 1e-3

            self.adamMomentum = [torch.zeros_like(param).to(self.device) for param in self.w_global]
            self.adamVariance = [torch.full_like(param, self.tau**2).to(self.device) for param in self.w_global]

        # FedStale state — per-client gradient memories and participation probs
        if self.mode == 'fedstale':
            # p_i = marginal probability that client i lands in the aggregated
            # set S_t. Here S_t = K users drawn uniformly at random from the
            # available set A_t, so
            #     p_i = P_on_i * E[ min(1, K/|A_t|) | i in A_t ]
            #         = E[ 1{i in A_t} * min(1, K/|A_t|) ],
            # estimated by Monte-Carlo over the availability Markov chains.
            # (Stationary availability P_on alone is NOT p_i: it ignores the
            #  K-of-available selection bottleneck and would make 1/p_i wrong.)
            self.fedstale_pi = self._estimate_fedstale_pi()
            print(f"FedStale p_i (MC estimate): "
                  f"min={self.fedstale_pi.min():.4f} "
                  f"mean={self.fedstale_pi.mean():.4f} "
                  f"max={self.fedstale_pi.max():.4f}")
            # h_i: gradient memory per user, initialised to zero (stored on CPU)
            self.fedstale_memory = [
                [torch.zeros_like(p).to("cpu") for p in self.w_global]
                for _ in range(self.num_users)
            ]
            # Running sum Σ_i h_i kept on device for O(|S|) per-round update
            self.fedstale_memory_sum = [torch.zeros_like(p).to(self.device) for p in self.w_global]

    def _estimate_fedstale_pi(self, n_rounds=100000, burn_in=2000, seed=20240507):
        """Monte-Carlo estimate of the marginal FedStale participation probability.

        S_t (the aggregated set) is K = bufferLimit users drawn uniformly at
        random from the available set A_t. The probability that client i ends
        up in S_t is therefore

            p_i = P_on_i * E[ min(1, K/|A_t|) | i in A_t ]
                = E[ 1{i in A_t} * min(1, K/|A_t|) ],

        which is exactly the quantity the 1/p_i importance weight in
        simulate_fedstale requires. Client availability follows the same
        two-state Markov chain as stepState(); this routine simulates those
        chains and averages the per-round inclusion probability.
        """
        rng = np.random.default_rng(seed)
        N = self.num_users
        K = float(self.bufferLimit)
        keep_av = np.asarray(self.keepProbAvail, dtype=float)     # P[stay avail | avail]
        keep_un = np.asarray(self.keepProbNotAvail, dtype=float)  # P[stay unavail | unavail]
        P01 = 1.0 - keep_un
        P10 = 1.0 - keep_av
        denom = P01 + P10
        with np.errstate(invalid='ignore', divide='ignore'):
            pon = np.where(denom > 0, P01 / denom, 0.5)           # stationary availability

        # start each chain from (approximately) its stationary distribution
        state = (rng.random(N) < pon).astype(np.int8)
        acc = np.zeros(N)
        counted = 0
        for t in range(burn_in + n_rounds):
            # one Markov step — identical transition rule to stepState()
            stay_prob = np.where(state == 1, keep_av, keep_un)
            flip = rng.random(N) >= stay_prob
            state = np.where(flip, 1 - state, state).astype(np.int8)
            if t < burn_in:
                continue
            n_av = int(state.sum())
            if n_av > 0:
                acc += state * min(1.0, K / n_av)
            counted += 1
        return acc / max(counted, 1)

    def lp_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, p: int = 2) -> float:
        """
        Compute the Lp cosine similarity between two flattened gradient vectors.
    
        Args:
            x (torch.Tensor): 1D tensor.
            y (torch.Tensor): 1D tensor.
            p (int): Norm degree (e.g., 2 for L2).
    
        Returns:
            float: The Lp cosine similarity.
        """
        norm_x = torch.norm(x, p=p)
        norm_y = torch.norm(y, p=p)
        norm_x_plus_y_sq = torch.norm(x + y, p=p) ** 2
        norm_x_sq = norm_x ** 2
        norm_y_sq = norm_y ** 2

        numerator = 0.5 * (norm_x_plus_y_sq - norm_x_sq - norm_y_sq)
        denominator = norm_x * norm_y + 1e-12  # avoid division by zero

        return (numerator / denominator).item()
    
    def cosine_similarity_policy(self) -> List[int]:

        """ Select user with highest cosine similarity to last gradient """

        valList = []
        userList = []

        for user in self.intermittentUsers:
            user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user]]).to(self.device)
            cos_sim = self.lp_cosine_similarity(user_grad_vector, self.lastGradient, p = self.cos_similarity)
            valList.append(cos_sim)
            userList.append(user)
            print(f"Cosine Similarity for user {user}: {cos_sim}")
        
        chosen_list = heapq.nlargest(self.bufferLimit, zip(valList, userList), key=lambda x: x[0]) if self.cosine_similarity_type else heapq.nsmallest(self.bufferLimit, zip(valList, userList), key=lambda x: x[0])

        chosen_list = [user for val, user in chosen_list]

        return chosen_list
    
    def compute_pi(self, r, pon, alpha):
        """Unified alpha-fair waterfilling policy (KKT solution).

        Solves: max Σ_u U_α(p_u · R_u)  s.t.  Σ_u P_on_u · p_u ≤ K
        KKT: p*_u = min(1, (R_u^(1-α) / (λ* · P_on_u))^(1/α))
        Implemented in log-space for numerical stability across all α values.

        alpha controls the fairness–efficiency trade-off:
          alpha = 0   : throughput maximisation (greedy by R_u / P_on_u)
          alpha = 1   : proportional fairness   (p*_u ∝ 1 / P_on_u)
          alpha = 2   : harmonic fairness       (p*_u ∝ 1 / sqrt(R_u · P_on_u))
          alpha = inf : max-min fairness        (p*_u · R_u = const)
        """
        K = float(self.bufferLimit)
        N = self.num_users
        valid = (pon > 1e-12) & (r > 1e-12)

        # If total availability is within budget, all users participate freely
        if np.dot(pon, np.ones(N)) <= K:
            return np.ones(N)

        # --- alpha = 0: throughput, greedy by R_u / P_on_u ---
        if alpha == 0:
            ratio = np.where(valid, r / pon, -np.inf)
            order = np.argsort(-ratio)
            p = np.zeros(N)
            budget = K
            for u in order:
                if not valid[u] or budget <= 0:
                    break
                alloc = min(1.0, budget / pon[u])
                p[u] = alloc
                budget -= pon[u] * alloc
            return p

        # --- alpha → ∞: max-min, p*_u · R_u = const ---
        if np.isinf(alpha):
            def load_mm(c):
                p = np.where(valid, np.minimum(1.0, c / r), 0.0)
                return np.dot(pon, p)
            r_max = r[valid].max() if valid.any() else 1.0
            c_lo, c_hi = 0.0, r_max * 1e9
            for _ in range(200):
                c_mid = (c_lo + c_hi) / 2.0
                if load_mm(c_mid) < K:
                    c_lo = c_mid
                else:
                    c_hi = c_mid
            c_star = (c_lo + c_hi) / 2.0
            return np.where(valid, np.minimum(1.0, c_star / r), 0.0)

        # --- General: 0 < alpha < ∞, log-space to avoid float overflow ---
        # log(phi_u) = (1-alpha)*log(R_u) - log(P_on_u)
        log_phi = np.where(valid, (1.0 - alpha) * np.log(r) - np.log(pon), -np.inf)

        def load(log_nu):
            # p_u = min(1, exp((log_phi_u - log_nu) / alpha))
            lp = (log_phi - log_nu) / alpha
            p = np.where(valid, np.where(lp >= 0.0, 1.0, np.exp(lp)), 0.0)
            return np.dot(pon, p)

        # Bisect in log(nu) space; load is decreasing in log_nu
        log_phi_valid = log_phi[valid]
        log_nu_lo = log_phi_valid.min() - 100.0  # all p=1 → load = Σ P_on_u > K
        log_nu_hi = log_phi_valid.max() + 100.0  # all p≈0 → load ≈ 0 < K
        for _ in range(200):
            log_nu_mid = (log_nu_lo + log_nu_hi) / 2.0
            if load(log_nu_mid) > K:
                log_nu_lo = log_nu_mid
            else:
                log_nu_hi = log_nu_mid

        log_nu_star = (log_nu_lo + log_nu_hi) / 2.0
        lp_star = np.where(valid, (log_phi - log_nu_star) / alpha, -np.inf)
        return np.where(valid, np.where(lp_star >= 0.0, 1.0, np.exp(lp_star)), 0.0)

    def calculate_policy(self):
        """Compute FEDFresh participation probabilities via alpha-fair policy.

        Solves: max Σ_u U_alpha(p_u · R_u)  s.t.  Σ_u P_on_u · p_u ≤ K
        where alpha = self.temperature controls the fairness–efficiency trade-off:
          alpha = 0   : throughput maximisation
          alpha = 1   : proportional fairness
          alpha = 2   : harmonic fairness
          alpha = inf : max-min fairness
        """
        r = np.zeros(self.num_users)
        pon = np.zeros(self.num_users)

        for iii in range(self.num_users):
            P10 = 1 - self.keepProbAvail[iii]    # P_{1,0}: available -> unavailable
            P01 = 1 - self.keepProbNotAvail[iii]  # P_{0,1}: unavailable -> available

            # R_u: asymptotic average inverse-staleness rate (renewal reward, d(tau)=1/tau)
            term1 = (1 - P10)
            term2 = (P10 * P01) / (1 - P01)
            term3 = (P10 * P01) / ((1 - P01) ** 2) * np.log(P01)
            numerator = term1 - term2 - term3
            denominator = 1 + P10 / P01
            r[iii] = numerator / denominator
            pon[iii] = P01 / (P01 + P10)

        pi = self.compute_pi(r, pon, alpha=self.temperature)
        print(f"FEDFresh pi (alpha={self.temperature}): {pi}")
        return pi

    def innerProductTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)
        
        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)
        print(self.bufferSize)
        varEst = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([(g/self.UserAgeMemory[user]).view(-1) for g in self.sparse_gradient[user]]).t()
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            print(accInner/torch.norm(user_grad_vector)/torch.norm(global_grad_vector))
            varEst = varEst + torch.square(accInner-gradMag)
        varEst = varEst/max(1, self.bufferSize-1)
        
        conLHS = varEst/self.bufferSize
        conRHS = torch.square(self.theta_inner*gradMag)
        print("Inner Product Test:", conLHS, "<=", conRHS)
        check = conLHS <= conRHS
        return check 

    def orthogonalityTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)

        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)

        orthTest = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user]])
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            grad = user_grad_vector - accInner/gradMag*global_grad_vector
            orthTest = orthTest + torch.dot(grad, grad)
        
        
        conLHS = orthTest/(max(1, self.bufferSize-1)*self.bufferSize)
        conRHS = (self.nu_orthogonal*self.nu_orthogonal)*gradMag
        print("Orthoganality Test:", conLHS, "<=", conRHS)

        check = conLHS <= conRHS 
        return check

    def stepState(self):
        for iii in range(self.num_users):
            if (self.intermittentStateOneHot[iii]):
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
            else:
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbNotAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
        self.intermittentUsers = np.where(self.intermittentStateOneHot)[0]

    # Calculate gradient difference between two sets of weights
    def calculate_gradient_difference(self, w_before, w_after):
        return [w_after[k] - w_before[k] for k in range(len(w_after))]
    
    # Sparsify the model weights
    def top_k_sparsificate_model_weights(self, weights, fraction):
        flat_weights = torch.cat([w.view(-1) for w in weights])
        threshold_value = torch.quantile(torch.abs(flat_weights), 1 - fraction)
        new_weights = []
        for w in weights:
            mask = torch.abs(w) >= threshold_value
            new_weights.append(w * mask.float())
        return new_weights
    

    def train_users(self, list_users):
        for user_id in list_users:

            # Reset model weights to the initial weights before each user's local training
            model = [param.data.clone().to(self.device) for param in self.w_user[user_id]]
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), model):
                    param.copy_(saved) 
            torch.cuda.empty_cache()

            # Retrieve the user's training data (combined from all memory cells)
            trainloader = self.TrainSetUsers[user_id]
            
            if self.train_mode == "MNIST":
                for epoch in range(self.epochs):
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)     
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()
                        self.optimizer.step()
                        torch.cuda.empty_cache()
            else: 
                for epoch in range(self.epochs): 
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()

                        self.optimizer.step()
        
            w_new = [param.data.clone().to(self.device) for param in self.model.parameters()]
            gradient_diff = self.calculate_gradient_difference(model, w_new)
            sparse_gradient = self.top_k_sparsificate_model_weights(gradient_diff, self.fraction[0]) 
            self.sparse_gradient[user_id] = [sg.to("cpu") for sg in sparse_gradient]

    def aggregate_gradients(self, tempUserAgeDL):

        # Normalize gradients if unit_gradients is set
        if self.unit_gradients:
            acc = 0 
            for user in self.selected_users_UL:
                norm = np.sqrt(sum([torch.sum(g**2).item() for g in self.sparse_gradient[user]]))
                if norm > 0:
                    self.sparse_gradient[user] = [g / norm for g in self.sparse_gradient[user]]
                
                norm = np.sqrt(sum([torch.sum(g**2).item() for g in self.sparse_gradient[user]]))
                print(f"Norm of sparse gradient for user {user}: {norm}")
    
        #Sum of trained gradients
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        for user in self.selected_users_UL:
            self.UserAgeUL[user] = 0
            self.contribution[user] += np.sqrt(sum([torch.sum((g/tempUserAgeDL[user].item())**2).item() for g in self.sparse_gradient[user]]))
            temp_gradient = [sg.to(self.device) for sg in self.sparse_gradient[user]]
            self.expected_gradient_magnitude[user] += np.sqrt(sum([torch.sum(g**2).item() for g in temp_gradient]))
            self.sum_terms = [self.sum_terms[j] + temp_gradient[j]/(tempUserAgeDL[user]) for j in range(len(self.sum_terms))] 
        
        if self.adam:
            # Adam update
            self.adamMomentum = [self.beta1 * m + (1 - self.beta1) * (s / len(self.selected_users_UL)) for m, s in zip(self.adamMomentum, self.sum_terms)]
            self.adamVariance = [self.beta2 * v + (1 - self.beta2) * ((s / len(self.selected_users_UL)) ** 2) for v, s in zip(self.adamVariance, self.sum_terms)]

            self.lastGradient = [ self.learning_rate_server * self.adamMomentum[j] / (torch.sqrt(self.adamVariance[j]) + self.tau) for j in range(len(self.sum_terms))]
            
            self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
            
            # Update global model
            self.w_global = [self.w_global[j] + self.learning_rate_server * self.adamMomentum[j] / (torch.sqrt(self.adamVariance[j]) + self.tau) for j in range(len(self.sum_terms))] 
        else:
            self.lastGradient = [s / len(self.selected_users_UL) for s in self.sum_terms]
            self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
            
            # Update global model
            self.w_global = [self.w_global[j] + self.learning_rate_server * self.sum_terms[j]/len(self.selected_users_UL) for j in range(len(self.sum_terms))] 
        
    def simulate_fedbuff(self, run, seed_index, timeframe):
        """FedBuff: Buffered Asynchronous Federated Learning (Nguyen et al., 2022).

        K = bufferLimit chosen users (random from available) compute gradients on their
        local (possibly stale) model. The server aggregates with staleness scaling
        s(τ) = 1/τ^0.5 and updates once per round. ALL available users then
        receive the new global model so their local copy stays fresh for next round.
        """
        self.stepState()
        if len(self.intermittentUsers) == 0:
            print("No users available, skipping round")
            self.UserAgeDL = self.UserAgeDL + self.allOnes
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        # Select K chosen users randomly from available
        k = min(self.bufferLimit, len(self.intermittentUsers))
        chosen_idx = np.random.choice(len(self.intermittentUsers), k, replace=False)
        self.selected_users_UL = self.intermittentUsers[chosen_idx]
        self.num_send += k
        print(f"Chosen Users (FedBuff) = {self.selected_users_UL.tolist()}")

        # Only chosen users compute gradients on their local (possibly stale) model
        self.train_users(self.selected_users_UL.tolist())

        # Staleness-scaled aggregation: agg = Σ s(τ_u)·g_u,  s(τ) = 1/τ^0.5
        agg = [torch.zeros_like(p).to(self.device) for p in self.w_global]
        for user in self.selected_users_UL:
            tau   = float(self.UserAgeDL[user].item())
            scale = 1.0 / tau ** 0.5
            for j in range(len(agg)):
                agg[j] = agg[j] + scale * self.sparse_gradient[user][j].to(self.device)
            self.contribution[user] += np.sqrt(
                sum(torch.sum((scale * g) ** 2).item() for g in self.sparse_gradient[user])
            )
            self.expected_gradient_magnitude[user] += np.sqrt(
                sum(torch.sum(g ** 2).item() for g in self.sparse_gradient[user])
            )

        # Server update: normalise by K (number of buffered gradients)
        self.w_global = [
            self.w_global[j] + self.learning_rate_server * agg[j] / k
            for j in range(len(self.w_global))
        ]

        # ALL available users receive the updated global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0   # reset staleness (becomes 1 after +allOnes)

        self.UserAgeDL = self.UserAgeDL + self.allOnes
        return self.w_global

    def simulate_fedstale(self, run, seed_index, timeframe):
        """FedStale: leveraging stale client updates (Rodio & Neglia, 2024).

        K = bufferLimit chosen users (random from available) receive the current global
        model and compute a FRESH gradient. The server aggregates:

            Δ = (β/N)·Σ_all h_i  +  (1/N)·Σ_{i∈S^t} (1/p_i)·(g_i − β·h_i)

        Memory update: h_i ← g_i for i∈S^t; unchanged for i∉S^t.
        Σ_i h_i is tracked incrementally to keep per-round cost O(|S^t|·layers).
        ALL available users receive the updated model at the end of the round.

        β = self.temperature:  0 → importance-weighted FedAvg,  1 → FedVARP
        p_i = marginal probability i is in the aggregated set S^t, estimated by
              Monte-Carlo in _estimate_fedstale_pi() (NOT the raw availability P_on).
        """
        self.stepState()
        if len(self.intermittentUsers) == 0:
            print("No users available, skipping round")
            self.UserAgeDL = self.UserAgeDL + self.allOnes
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        N    = float(self.num_users)
        beta = self.temperature

        # Select K chosen users randomly from available
        k = min(self.bufferLimit, len(self.intermittentUsers))
        chosen_idx = np.random.choice(len(self.intermittentUsers), k, replace=False)
        self.selected_users_UL = self.intermittentUsers[chosen_idx]
        self.num_send += k
        print(f"Chosen Users (FedStale) = {self.selected_users_UL.tolist()}")

        # Give chosen users the current global model, then compute fresh gradient
        for user in self.selected_users_UL:
            self.w_user[user] = [w.clone() for w in self.w_global]
        self.train_users(self.selected_users_UL.tolist())

        # ── Aggregation ──────────────────────────────────────────────────────────
        delta = [torch.zeros_like(p).to(self.device) for p in self.w_global]

        # Memory term: (β/N) · Σ_all h_i  (uses running sum — O(layers), not O(N·layers))
        for j in range(len(delta)):
            delta[j] = delta[j] + (beta / N) * self.fedstale_memory_sum[j]

        # Fresh-correction term: (1/N) · Σ_{i∈S^t} (1/p_i) · (g_i − β·h_i)
        for user in self.selected_users_UL:
            p_i   = max(float(self.fedstale_pi[user]), 1e-6)
            inv_p = 1.0 / p_i
            for j in range(len(delta)):
                g_i = self.sparse_gradient[user][j].to(self.device)
                h_i = self.fedstale_memory[user][j].to(self.device)
                delta[j] = delta[j] + (inv_p / N) * (g_i - beta * h_i)

        # Server model update
        self.w_global = [
            self.w_global[j] + self.learning_rate_server * delta[j]
            for j in range(len(self.w_global))
        ]

        # Update memories and running sum for chosen users only
        for user in self.selected_users_UL:
            for j in range(len(self.w_global)):
                new_h = self.sparse_gradient[user][j]       # on CPU
                old_h = self.fedstale_memory[user][j]
                self.fedstale_memory_sum[j] = self.fedstale_memory_sum[j] + \
                    new_h.to(self.device) - old_h.to(self.device)
                self.fedstale_memory[user][j] = new_h.clone()
            self.contribution[user] += np.sqrt(
                sum(torch.sum(g ** 2).item() for g in self.sparse_gradient[user])
            )
            self.expected_gradient_magnitude[user] += np.sqrt(
                sum(torch.sum(g ** 2).item() for g in self.sparse_gradient[user])
            )

        # ALL available users receive the updated global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0   # reset staleness (becomes 1 after +allOnes)

        self.UserAgeDL = self.UserAgeDL + self.allOnes
        return self.w_global

    def simulate_async_Asymp_EI(self, run, seed_index, timeframe):
        """FEDFresh-style: probabilistic selection policy p_u(λ), stale-model training.

        Each available user independently decides to transmit with prob p_u(λ)
        (the FEDFresh blend of fairness and efficiency policies).
        Only the chosen users train — on their local (possibly stale) model.
        The server aggregates with 1/τ staleness weighting.
        ALL available users then receive the updated global model.
        """
        self.stepState()
        if len(self.intermittentUsers) == 0:
            print("No users available, skipping round")
            self.UserAgeDL = self.UserAgeDL + self.allOnes
            return self.w_global
        print(f"Available Users S_t = {self.intermittentUsers}")

        # Probabilistic selection: each available user transmits with prob p_u(λ)
        tempPi = self.pi[self.intermittentUsers].flatten()
        bernoulli_flips = np.random.rand(len(self.intermittentUsers)) < tempPi
        self.selected_users_UL = self.intermittentUsers[bernoulli_flips]
        self.num_send += len(self.selected_users_UL)
        print(f"Transmitting Users K_t = {self.selected_users_UL.tolist()}")

        if len(self.selected_users_UL) > 0:
            # Train only chosen users on their stale local model w_user[u]
            self.train_users(self.selected_users_UL.tolist())

            # Capture τ before the age reset, then aggregate with 1/τ weighting
            tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
            self.aggregate_gradients(tempUserAgeDL)

        # ALL available users receive the updated global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0  # becomes 1 after +allOnes

        self.UserAgeDL = self.UserAgeDL + self.allOnes
        return self.w_global
    
    def simulate_async_Asymp_Age(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        tempUserAgeUL = self.UserAgeUL[self.intermittentUsers]
        print(f"User Age UL: {tempUserAgeUL.squeeze()}")
        tempUserAgeDL = self.UserAgeDL[self.intermittentUsers] 
        print(f"User Age DL: {tempUserAgeDL.squeeze()}")

        # Calculate age difference and select top-k users
        age_diff = (tempUserAgeUL).squeeze()
        k = min(int(self.bufferLimit), len(self.intermittentUsers))        
        sorted_indices = torch.atleast_1d(torch.argsort(age_diff, descending=True))
        topk_indices = sorted_indices[:k]
        self.selected_users_UL = self.intermittentUsers[topk_indices.cpu().numpy()]
        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        self.train_users(self.selected_users_UL.tolist())

        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0

        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_CosSim(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        
        self.train_users(self.intermittentUsers.tolist())


        self.selected_users_UL = self.cosine_similarity_policy()

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_random(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        idx = torch.randint(0, len(self.intermittentUsers), (self.bufferLimit,)).tolist()

        self.selected_users_UL = [self.intermittentUsers[i] for i in idx]

        self.train_users(self.selected_users_UL)

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_Fresh(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        FreshUsers = np.where(self.UserAgeDL[self.intermittentUsers]==1)[0]

        if len(FreshUsers) == 0:
            print("No fresh users available")
            return self.w_global

        k = min(self.bufferLimit, len(FreshUsers))
        idx = torch.randint(0, len(FreshUsers), (k,)).tolist()
        self.selected_users_UL = self.intermittentUsers[FreshUsers[idx]]

        self.train_users(self.selected_users_UL)

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_test(self, run, seed_index, timeframe):
        self.train_users(list(range(self.num_users)))
        for user_id in range(self.num_users):
            for user_id2 in range(user_id, self.num_users):
                # Flatten gradients into 1D vectors
                user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user_id]])
                global_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user_id2]])

                # Compute cosine similarity
                lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
                print(f"Similarity between {user_id} and {user_id2} = {lp_cos_val}")

    def run(self, runNo, seed_index, timeframe):
        """Dispatch based on the FL mode."""
        if self.mode == 'test':
            return self.test(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_EI':
            return self.simulate_async_Asymp_EI(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_age':
            return self.simulate_async_Asymp_Age(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_cossim':
            return self.simulate_async_Asymp_CosSim(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_random':
            return self.simulate_async_Asymp_random(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_fresh':
            return self.simulate_async_Asymp_Fresh(runNo, seed_index, timeframe)
        elif self.mode == 'fedbuff':
            return self.simulate_fedbuff(runNo, seed_index, timeframe)
        elif self.mode == 'fedstale':
            return self.simulate_fedstale(runNo, seed_index, timeframe)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
 