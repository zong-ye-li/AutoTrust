import numpy as np
import json
from pathlib import Path
import os

class SharedRedSyndicateAgent:
    """
    Tabular Q-Learning Agent for the Red Syndicate.
    Maintains a shared Q-table across all Red nodes to avoid the curse of dimensionality.
    State space is normally the 5-tuple
    (S_self, S_target, I_target, S_syndicate, S_est_credence).  The explicitly
    configured ablation may omit only its final estimated-credence coordinate.
    Action space: 0 (Farming/Normal), 1 (Attacking/Boosting), 2 (Smearing)
    """
    def __init__(self, alpha_lr=0.1, gamma=0.95, epsilon=0.2, min_epsilon=0.05, epsilon_decay=0.995, seed=None, training=True, manipulation=True,
                 include_estimated_postattack_credence=True):
        self.rng = np.random.default_rng(seed)
        self.training = training
        self.manipulation = manipulation
        if not isinstance(include_estimated_postattack_credence, bool):
            raise ValueError('include_estimated_postattack_credence must be boolean')
        self.include_estimated_postattack_credence = include_estimated_postattack_credence
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # State dimensions: 5 x 5 x 2 x 3 [x 3 estimated post-attack credence].
        # Small random initialization to break symmetry
        self.q_table = self.rng.uniform(low=-0.01, high=0.01, size=self.q_shape)

    @property
    def state_shape(self):
        return (5, 5, 2, 3, 3) if self.include_estimated_postattack_credence else (5, 5, 2, 3)

    @property
    def q_shape(self):
        return self.state_shape + (3,)

    def _get_valid_actions(self, i_target):
        """
        Action Masking based on target identity.
        If Blue (0): Farming (0), Attacking (1), Smearing (2)
        If Red (1): Normal (0), Boosting (1) - Smearing is disabled against teammates.
        """
        if not self.manipulation:
            return [0, 1] if i_target == 0 else [0]
        if i_target == 0:
            return [0, 1, 2]
        else:
            return [0, 1]

    def choose_action(self, state, context=None):
        """
        Epsilon-greedy action selection with masking.
        :param state: Full 5-tuple or the configured reduced 4-tuple.
        :param context: Optional decision-time information. Learned policies
            intentionally ignore it; frozen baselines may use only their own
            Red-side credence estimate supplied by the simulator.
        :return: Chosen action (int)
        """
        i_target = state[2]
        valid_actions = self._get_valid_actions(i_target)
        
        # Exploration
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
            
        # Exploitation
        q_values = self.q_table[state]
        
        # Mask out invalid actions by setting them to negative infinity
        masked_q_values = np.copy(q_values)
        for a in range(3):
            if a not in valid_actions:
                masked_q_values[a] = -np.inf
                
        return int(np.argmax(masked_q_values))

    def learn(self, state, action, reward, next_state, done):
        """Apply one transition through the same synchronous batch path."""
        self.learn_batch([(state, action, reward, next_state, done)])

    def learn_batch(self, transitions):
        """Apply a synchronous tabular Q-learning update for one rollout batch.

        Every TD target is calculated from the same frozen Q-table snapshot.
        Repeated samples for the same state-action pair are averaged before the
        table is updated, so batch ordering cannot change the result.
        """
        if not self.training:
            raise RuntimeError("Frozen policy cannot learn")
        if not transitions:
            return

        q_snapshot = np.copy(self.q_table)
        td_errors_by_key = {}

        for state, action, reward, next_state, done in transitions:
            state = tuple(state)
            next_state = tuple(next_state)
            action = int(action)

            valid_actions = self._get_valid_actions(state[2])
            if action not in valid_actions:
                raise ValueError(
                    f"Action {action} is invalid for target identity {state[2]}"
                )

            current_q = q_snapshot[state + (action,)]
            if done:
                max_next_q = 0.0
            else:
                valid_next_actions = self._get_valid_actions(next_state[2])
                next_q_values = [
                    q_snapshot[next_state + (next_action,)]
                    for next_action in valid_next_actions
                ]
                max_next_q = np.max(next_q_values)

            td_target = reward + self.gamma * max_next_q
            key = (state, action)
            td_errors_by_key.setdefault(key, []).append(td_target - current_q)

        for (state, action), td_errors in td_errors_by_key.items():
            old_q = q_snapshot[state + (action,)]
            self.q_table[state + (action,)] = (
                old_q + self.alpha_lr * np.mean(td_errors)
            )

    def decay_epsilon(self):
        """Decays the exploration rate at the end of an epoch."""
        if self.training and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

    def save(self, path, metadata=None):
        path = Path(path)
        payload = dict(alpha_lr=self.alpha_lr, gamma=self.gamma, epsilon=self.epsilon,
                       min_epsilon=self.min_epsilon, epsilon_decay=self.epsilon_decay,
                       training=self.training, manipulation=self.manipulation,
                       include_estimated_postattack_credence=self.include_estimated_postattack_credence,
                       rng=self.rng.bit_generator.state, metadata=metadata or {})
        temp = path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as stream:
            np.savez(stream, q=self.q_table, payload=json.dumps(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    @classmethod
    def load(cls, path, training=None):
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data['payload']))
            rng = payload.pop('rng')
            metadata = payload.pop('metadata')
            if training is not None:
                payload['training'] = training
            obj = cls(**payload)
            obj.q_table = data['q'].copy()
        if obj.q_table.shape != obj.q_shape or not np.isfinite(obj.q_table).all():
            raise ValueError('Invalid Q checkpoint')
        obj.rng.bit_generator.state = rng
        return obj, metadata
