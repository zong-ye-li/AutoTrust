import numpy as np
import warnings

class AgentMeshEnvironment:
    """
    Macro-Simulation Environment for Federated Trust Network
    Handles graph state, dynamic routing, EigenTrust matrix operations, 
    and utility-driven trust updates.
    """
    def __init__(self, num_nodes=50, malign=0.1, w_local=0.6, w_global=0.25, w_random=0.15, seed=None, pretrust=0, threshold_population='all-alive'):
        if num_nodes < 1 or not 0 <= malign <= 1:
            raise ValueError("Invalid population")
        if not np.isfinite([w_local,w_global,w_random]).all() or min(w_local,w_global,w_random)<0 or w_local+w_global+w_random<=0:
            raise ValueError("Invalid routing weights")
        if threshold_population not in ('all-alive', 'non-anchor-alive'):
            raise ValueError("Invalid threshold population")
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.malign = malign
        self.threshold_population = threshold_population
        
        # Routing weights
        self.w_local = w_local
        self.w_global = w_global
        self.w_random = w_random
        
        # Node identities: 0 for Blue (Benign), 1 for Red (Malicious)
        self.identities = np.zeros(self.num_nodes, dtype=int)
        num_red = int(self.num_nodes * self.malign)
        self.identities[:num_red] = 1
        self.rng.shuffle(self.identities)
        blues = np.flatnonzero(self.identities == 0)
        if not isinstance(pretrust, int) or not 0 <= pretrust <= len(blues):
            raise ValueError("Invalid pretrust count")
        self.pretrusted = self.rng.permutation(blues)[:pretrust]
        
        self.reset()
        
    def reset(self):
        """Initializes or resets the network state."""
        # A: Alive Status Vector (1 = alive, 0 = eradicated)
        self.A = np.ones(self.num_nodes)
        
        # 1. Latent Trust Score Matrix (Z_latent) for utility-driven update
        # Initial prior c_B = 0.9 corresponds to Z approx 2.197
        initial_z = np.log(0.9 / 0.1)
        self.Z_latent = np.full((self.num_nodes, self.num_nodes), initial_z)
        np.fill_diagonal(self.Z_latent, 0.0) # Self-trust is meaningless

        # 2. Red_Est_Z: Red nodes' private local memory for estimating target's latent trust
        self.Red_Est_Z = np.full((self.num_nodes, self.num_nodes), initial_z)
        np.fill_diagonal(self.Red_Est_Z, 0.0)

        # 3. Relative Trust Matrix C_rel (Markov transition matrix)
        self.C_rel = np.zeros((self.num_nodes, self.num_nodes))
        self._update_C_rel() 
        
        # t: Global Trust Vector
        self.t = np.ones(self.num_nodes) / self.num_nodes
        self.eradication_diagnostics = None

    def _update_C_rel(self):
        """
        Maps Z_latent to absolute credence (c_B) via Sigmoid, 
        then applies row normalization to generate C_rel.
        """
        # Step 1: Sigmoid mapping Z -> c_B (strictly between 0 and 1)
        # Clip Z to [-3, 3] to prevent overflow and rigid trust bounds (~0.047 to  0.952)
        Z_clipped = np.clip(self.Z_latent, -3.0, 3.0)
        C_abs = 1.0 / (1.0 + np.exp(-Z_clipped))
        
        # Mask out dead nodes
        C_abs[self.A == 0, :] = 0.0
        C_abs[:, self.A == 0] = 0.0
        np.fill_diagonal(C_abs, 0.0)
        
        sums = C_abs.sum(axis=1, keepdims=True)
        self.C_rel = np.divide(C_abs, sums, out=np.zeros_like(C_abs), where=sums>0)

    def get_absolute_prior(self, i, j):
        """
        Retrieves the absolute prior trust (c_B) of node i towards node j.
        Used by the Bayesian Calculator for expected utility calculation.
        """
        z_val = np.clip(self.Z_latent[i, j], -3.0, 3.0)
        return 1.0 / (1.0 + np.exp(-z_val))

    def update_local_trust(self, evaluator_id, target_id, v_score, learning_rate=0.1):
        """
        Updates the latent trust score.
        :param evaluator_id: The node evaluating the interaction (updating its belief).
        :param target_id: The node being evaluated.
        :param v_score: The expected utility (evidence gradient).
        """
        if self.A[evaluator_id] == 0 or self.A[target_id] == 0:
            return
            
        # Evaluator updates its actual latent trust towards the target
        self.Z_latent[evaluator_id, target_id] += learning_rate * v_score
        self.Z_latent[evaluator_id, target_id] = np.clip(self.Z_latent[evaluator_id, target_id], -3.0, 3.0)
        
        self._update_C_rel()

    def override_local_trust(self, evaluator_id, target_id, z_value):
        """Replace one directed latent-trust value for an explicit Red action."""
        if self.A[evaluator_id] == 0 or self.A[target_id] == 0:
            return

        z_value = float(np.clip(z_value, -3.0, 3.0))
        self.Z_latent[evaluator_id, target_id] = z_value

        self._update_C_rel()

    def compute_routing_probs(self, initiator_id):
        """
        Computes the routing probabilities using dynamic masking.
        Returns: 1D numpy array of length |V|
        """
        alive_count = np.sum(self.A)
        if alive_count <= 1 or self.A[initiator_id] == 0:
            return np.zeros(self.num_nodes)
            
        random_prob = 1.0 / (alive_count - 1)
        
        # Use C_rel for routing probability calculation
        P = (self.w_local * self.C_rel[initiator_id] + 
             self.w_global * self.t + 
             self.w_random * random_prob)
             
        # Dynamic Masking
        P[initiator_id] = 0.0 
        P[self.A == 0] = 0.0  
        
        P_sum = np.sum(P)
        if P_sum > 0:
            P = P / P_sum
        else:
            P = np.ones(self.num_nodes)
            P[initiator_id] = 0.0
            P[self.A == 0] = 0.0
            P = P / np.sum(P)
            
        return P

    def update_global_trust(self, beta=0.15, epsilon=1e-6, max_iter=1000):
        if not 0 < beta <= 1 or epsilon <= 0 or max_iter < 1:
            raise ValueError("Invalid EigenTrust settings")
        alive = np.flatnonzero(self.A)
        p = np.zeros(self.num_nodes)
        anchors = self.pretrusted[self.A[self.pretrusted] == 1]
        # Anchors are not immune to eradication; if none survive use alive-uniform.
        support = anchors if len(anchors) else alive
        if len(support):
            p[support] = 1 / len(support)
        if len(alive) <= 1:
            self.t = p
            self.eigen_diagnostics = dict(converged=True, iterations=0, residual=0., anchor_fallback=not len(anchors))
            return self.eigen_diagnostics
        t = self.t * self.A
        t = t / t.sum() if t.sum() else p.copy()
        for iteration in range(1, max_iter+1):
            t = (1-beta) * (self.C_rel.T @ t) + beta*p
            residual = float(np.linalg.norm((1-beta)*(self.C_rel.T @ t)+beta*p-t))
            if residual < epsilon:
                break
        self.t = t
        self.eigen_diagnostics = dict(converged=residual < epsilon, iterations=iteration, residual=residual, anchor_fallback=not len(anchors))
        if residual >= epsilon:
            warnings.warn("EigenTrust did not converge", RuntimeWarning)
        return self.eigen_diagnostics

    def apply_eradication(self, n_std=2.5):
        """
        Eradicates nodes below the threshold τ (μ_t - n * σ_t) 
        and reconstructs matrices to prevent sinkholes.
        Returns: list of newly dead node indices
        """
        alive_indices = np.where(self.A == 1)[0]
        if len(alive_indices) == 0:
            return []
            
        if self.threshold_population == 'non-anchor-alive':
            sample_indices = np.setdiff1d(alive_indices, self.pretrusted, assume_unique=True)
            # A degenerate all-anchor survivor set has no non-anchor reference;
            # retain the original all-alive definition rather than invent a cutoff.
            if not len(sample_indices):
                sample_indices = alive_indices
        else:
            sample_indices = alive_indices
        sample_t = self.t[sample_indices]
        mu_t = np.mean(sample_t)
        sigma_t = np.std(sample_t)
        
        tau = mu_t - n_std * sigma_t
        self.eradication_diagnostics = dict(
            population=self.threshold_population,
            sample_size=int(len(sample_indices)),
            eligible_size=int(len(alive_indices)),
            mean=float(mu_t),
            std=float(sigma_t),
            threshold=float(tau),
        )
        
        newly_dead = []
        for idx in alive_indices:
            if self.t[idx] < tau:
                self.A[idx] = 0
                newly_dead.append(idx)
                
        if not newly_dead:
            return []
            
        # Post-Eradication Matrix Reconstruction
        self.Z_latent[self.A == 0, :] = -3.0 # Set to minimum latent score
        self.Z_latent[:, self.A == 0] = -3.0
        
        # Update C_rel based on the new graph structure
        self._update_C_rel()
            
        self.t[self.A == 0] = 0.0
        if np.sum(self.t) > 0:
            self.t = self.t / np.sum(self.t)
            
        return newly_dead
    
    def get_node_reputation_state(self, idx):
        """Helper method: Returns the relative reputation state (0-4) of a specific node."""
        alive_indices = np.where(self.A == 1)[0]
        if len(alive_indices) == 0:
            return 0
        alive_t = self.t[alive_indices]
        mu_t = np.mean(alive_t)
        sigma_t = np.std(alive_t)
        if sigma_t < 1e-15:
            return 2
        t_val = self.t[idx]

        if t_val < mu_t - 0.75 * sigma_t: return 0
        elif t_val < mu_t - 0.25 * sigma_t: return 1
        elif t_val < mu_t + 0.25 * sigma_t: return 2
        elif t_val < mu_t + 0.75 * sigma_t: return 3
        else: return 4

    def get_syndicate_state(self):
        """
        Calculates S_syndicate (Red Syndicate Collective Health Indicator).
        Returns: 0 (Danger), 1 (Neutral), 2 (Safe).
        """
        alive_reds = [i for i in range(self.num_nodes) if self.identities[i] == 1 and self.A[i] == 1]
        if not alive_reds:
            return 0
            
        states = [self.get_node_reputation_state(r) for r in alive_reds]
        avg_state = int(np.floor(sum(states) / len(states) + 0.5))
        
        if avg_state < 2: return 0
        elif avg_state == 2: return 1
        else: return 2

    def update_red_estimate(self, red_id, target_id, bayes_model, is_attack, learning_rate=0.1):
        z = self.Red_Est_Z[red_id, target_id]
        c = 1 / (1 + np.exp(-z))
        moments = bayes_model.moments(c)
        mu = moments['mu_m' if is_attack else 'mu_b']
        self.Red_Est_Z[red_id, target_id] = np.clip(z+learning_rate*mu, -3, 3)

    def get_est_credence_state(self, red_id, target_id, mu_m, learning_rate=0.1):
        z = self.Red_Est_Z[red_id, target_id]
        post = 1 / (1 + np.exp(-np.clip(z + learning_rate*mu_m, -3, 3)))
        return 0 if post < .5 else (1 if post < .9 else 2)
