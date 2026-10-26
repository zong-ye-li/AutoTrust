import argparse
import copy
import numpy as np
from pathlib import Path
from Environment import AgentMeshEnvironment
from TrustBayes import TrustBayesCalculator
from Agent import SharedRedSyndicateAgent

class Simulation:
    """
    Macro-Simulation Main Loop.
    Integrates Environment, Bayesian Calculators (WildJailbreak & FormAI), and the RL Agent.
    """
    def __init__(
        self,
        num_nodes=50,
        malign=0.1,
        epochs=1000,
        reference_version="qwen3.7-plus",
        prior_dir=None,
        seed=0, policy_seed=1, pretrust=0, n_std=2.5, threshold_population='all-alive',
        training=True, agent=None, event_sink=None, retain_logs=True,
        smoothing=0., alpha=9., terminal_penalty=-100., smearing_reward=-.1,
        include_estimated_postattack_credence=True,
    ):
        if epochs < 0:
            raise ValueError("Negative horizon")
        self.seed, self.policy_seed = seed, policy_seed
        self.n_std, self.training = n_std, training
        if not isinstance(include_estimated_postattack_credence, bool):
            raise ValueError('include_estimated_postattack_credence must be boolean')
        self.include_estimated_postattack_credence = include_estimated_postattack_credence
        self.terminal_penalty, self.smearing_reward = terminal_penalty, smearing_reward
        self.event_sink, self.retain_logs = event_sink, retain_logs
        streams = np.random.SeedSequence(seed).spawn(4)
        self.rng = np.random.default_rng(streams[1])
        self.epochs = epochs
        self.reference_version = reference_version
        
        # Initialize Environment
        self.env = AgentMeshEnvironment(
            num_nodes=num_nodes,
            malign=malign,
            seed=streams[0],
            pretrust=pretrust,
            threshold_population=threshold_population,
        )
        
        # Initialize RL Agent (Shared Brain for Red Syndicate)
        self.agent = agent if agent is not None else SharedRedSyndicateAgent(
            seed=policy_seed, training=training,
            include_estimated_postattack_credence=include_estimated_postattack_credence,
        )
        if getattr(self.agent, 'include_estimated_postattack_credence', True) != include_estimated_postattack_credence:
            raise ValueError('Simulation and agent state-observation configurations differ')
        
        # Initialize Bayesian Calculators for two different contexts.
        # Resolve paths relative to this source file so execution does not
        # depend on the caller's current working directory.
        if prior_dir is None:
            prior_dir = Path(__file__).resolve().parent / "prior"
        else:
            prior_dir = Path(prior_dir).expanduser().resolve()

        reference_paths = {
            "wj_distribution": prior_dir / f"distribution_wildjailbreak_{self.reference_version}.csv",
            "wj_delta_v": prior_dir / f"deltaV_lookup_table_wildjailbreak_{self.reference_version}.csv",
            "fa_distribution": prior_dir / f"distribution_formai_{self.reference_version}.csv",
            "fa_delta_v": prior_dir / f"deltaV_lookup_table_formai_{self.reference_version}.csv",
        }
        missing_paths = [path for key, path in reference_paths.items() if "distribution" in key and not path.is_file()]
        if missing_paths:
            missing = "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(
                f"Missing reference files for version '{self.reference_version}':\n{missing}"
            )

        # WJ -> Used when evaluating a Request (Initiator -> Target)
        self.bayes_wj = TrustBayesCalculator(
            reference_paths["wj_distribution"],
            alpha=alpha, smoothing=smoothing, rng=np.random.default_rng(streams[2]),
        )
        # FA -> Used when evaluating a Generation (Target -> Initiator)
        self.bayes_fa = TrustBayesCalculator(
            reference_paths["fa_distribution"],
            alpha=alpha, smoothing=smoothing, rng=np.random.default_rng(streams[3]),
        )

        # Keep the monitor observation and whether the resulting utility update
        # was replaced by an explicit Red smear/boost trust override.
        self.feedback_log = []

        # A transition is closed only when the same Red reaches its next
        # decision event. Completed transitions are learned as an epoch batch.
        self.pending_red_transitions = {}
        self.transition_log = []

    def _get_red_state(self, red_id, target_id, bayes_model):
        """Helper to construct the 5-Tuple state for the RL agent."""
        s_self = self.env.get_node_reputation_state(red_id)
        s_target = self.env.get_node_reputation_state(target_id)
        i_target = self.env.identities[target_id]
        s_syndicate = self.env.get_syndicate_state()
        
        # Red estimates its absolute credence in the target's mind
        z_est = np.clip(self.env.Red_Est_Z[red_id, target_id], -3.0, 3.0)
        c_B_est = 1.0 / (1.0 + np.exp(-z_est))
        
        mu_m = bayes_model.moments(c_B_est)['mu_m']
        s_est_credence = self.env.get_est_credence_state(red_id, target_id, mu_m)

        core_state = (s_self, s_target, i_target, s_syndicate)
        # Keep the private estimate calculation intact: fixed policies and
        # diagnostics retain their permitted Red-side post-attack context.
        return core_state + (s_est_credence,) if getattr(
            self, 'include_estimated_postattack_credence', True
        ) else core_state

    def _get_red_policy_context(self, red_id, target_id, bayes_model):
        """Return only the Red's permitted information for fixed policies."""
        z_est = float(np.clip(self.env.Red_Est_Z[red_id, target_id], -3.0, 3.0))
        estimated_credence = float(1.0 / (1.0 + np.exp(-z_est)))
        mu_m = float(bayes_model.moments(estimated_credence)['mu_m'])
        post_attack_z = float(np.clip(z_est + 0.1 * mu_m, -3.0, 3.0))
        return dict(
            estimated_z=z_est,
            estimated_credence=estimated_credence,
            mu_m=mu_m,
            post_attack_credence=float(1.0 / (1.0 + np.exp(-post_attack_z))),
        )

    def _choose_red_action(self, red_id, target_id, bayes_model, state):
        return self.agent.choose_action(
            state, self._get_red_policy_context(red_id, target_id, bayes_model)
        )

    def _evaluate_and_update(
        self,
        epoch,
        evaluator_id,
        evaluated_id,
        bayes_model,
        reference_kind,
        actual_action_is_attack=False,
        trust_override=None,
    ):
        """Run one monitor evaluation and update one directed trust edge.

        ``actual_action_is_attack`` is simulator-only ground truth used to sample
        the monitor observation. It is not knowledge available to the evaluator.
        For a normal action, the observed utility updates trust. For an explicit
        smear/boost action, the observation is still recorded but the acting
        Red changes only its own directed trust value.
        """
        prior = self.env.get_absolute_prior(evaluator_id, evaluated_id)
        observed_score, observed_v = bayes_model.get_v_score(
            prior,
            is_malicious_action=actual_action_is_attack,
        )

        if trust_override is None:
            self.env.update_local_trust(
                evaluator_id=evaluator_id,
                target_id=evaluated_id,
                v_score=observed_v,
            )
            update_mode = "utility"
        else:
            self.env.override_local_trust(
                evaluator_id=evaluator_id,
                target_id=evaluated_id,
                z_value=trust_override,
            )
            update_mode = "override"

        if self.env.identities[evaluated_id] == 1:
            self.env.update_red_estimate(evaluated_id, evaluator_id, bayes_model, actual_action_is_attack)
        self._emit('feedback', {
            "epoch": epoch,
            "reference_kind": reference_kind,
            "evaluator_id": evaluator_id,
            "evaluated_id": evaluated_id,
            "actual_action_is_attack": actual_action_is_attack,
            "observed_score": int(observed_score),
            "observed_v": float(observed_v),
            "update_mode": update_mode,
            "trust_override": trust_override,
        })

    def _sample_index(self, probabilities, uniform):
        total = float(np.sum(probabilities))
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Invalid routing probabilities")
        cumulative = np.cumsum(probabilities)
        idx = min(int(np.searchsorted(cumulative, uniform, side='right')), self.env.num_nodes - 1)
        return idx

    def _epoch_schedule(self):
        """Create a full-node random tape before applying the alive mask."""
        order = self.rng.permutation(self.env.num_nodes)
        route_uniforms = self.rng.random(self.env.num_nodes)
        wj_uniforms = self.bayes_wj.rng.random(self.env.num_nodes)
        fa_uniforms = self.bayes_fa.rng.random(self.env.num_nodes)

        schedule = []
        if np.sum(self.env.A) <= 1:
            return schedule
        for initiator in order:
            if self.env.A[initiator] == 0:
                continue
            probabilities = self.env.compute_routing_probs(initiator)
            target = self._sample_index(probabilities, float(route_uniforms[initiator]))
            schedule.append((
                int(initiator),
                int(target),
                float(wj_uniforms[initiator]),
                float(fa_uniforms[initiator]),
            ))
        return schedule

    def _close_pending_transition(self, red_id, next_state, completed):
        """Close Red's previous decision at its actual next decision state."""
        pending = self.pending_red_transitions.pop(red_id, None)
        if pending is None:
            return

        completed.append((
            red_id,
            pending["state"],
            pending["action"],
            pending["reward"],
            next_state,
            False,
        ))

    def _store_pending_transition(self, red_id, state, action, reward):
        """Keep a Red decision open until that Red acts again or is removed."""
        initiator, target = self.current_pair
        other = target if red_id == initiator else initiator
        model = self.bayes_wj if red_id == initiator else self.bayes_fa
        c = self.env.get_absolute_prior(other, red_id)
        estimated_c = float(1/(1+np.exp(-self.env.Red_Est_Z[red_id, other])))
        self._emit('decision', dict(epoch=self.current_epoch, red_id=int(red_id),
                   target_id=int(other), state=state, action=int(action), reward=float(reward),
                   eligible=not bool(self.env.identities[other]), c=c, estimated_c=estimated_c,
                   reference_kind='input' if red_id == initiator else 'output', **model.moments(c)))
        self.pending_red_transitions[red_id] = {
            "state": state,
            "action": action,
            "reward": reward,
        }

    def _finalize_dead_red_transitions(self, newly_dead):
        """Apply terminal punishment only to each dead Red's last action."""
        terminal_transitions = []
        for red_id in newly_dead:
            if self.env.identities[red_id] != 1:
                continue

            pending = self.pending_red_transitions.pop(red_id, None)
            if pending is None:
                continue

            terminal_transitions.append((
                red_id,
                pending["state"],
                pending["action"],
                pending["reward"] + getattr(self, "terminal_penalty", -100.0),
                pending["state"],  # Ignored by learn_batch when done is True.
                True,
            ))

        return terminal_transitions

    def _run_interaction(self, epoch, initiator, target, wj_uniform=None, fa_uniform=None):
        """Execute both feedback directions and return any Red transitions."""
        self.current_epoch = epoch
        self.current_pair = (initiator, target)
        is_red_init = (self.env.identities[initiator] == 1)
        is_red_target = (self.env.identities[target] == 1)

        request_is_attack = False
        generation_is_attack = False
        request_trust_override = None
        generation_trust_override = None
        completed_transitions = []

        # Blue -> Blue: both directions use their observed monitor utility.
        if not is_red_init and not is_red_target:
            pass

        # Red -> Blue: Red's action concerns its request or its rating of output.
        elif is_red_init and not is_red_target:
            state = self._get_red_state(initiator, target, self.bayes_wj)
            self._close_pending_transition(
                initiator, state, completed_transitions
            )
            action = self._choose_red_action(initiator, target, self.bayes_wj, state)
            if action == 0:  # Farming: normal request
                reward = -0.1
            elif action == 1:  # Attacking: harmful request
                request_is_attack = True
                reward = 10.0
            elif action == 2:  # Smearing: Red sets only Z[Red, Blue] to -3
                generation_trust_override = -3.0
                reward = getattr(self, "smearing_reward", -.1)
            else:
                raise ValueError(f"Invalid Red-vs-Blue action: {action}")
            self._store_pending_transition(
                initiator, state, action, reward
            )

        # Blue -> Red: Red's action concerns its output or its rating of input.
        elif not is_red_init and is_red_target:
            state = self._get_red_state(target, initiator, self.bayes_fa)
            self._close_pending_transition(
                target, state, completed_transitions
            )
            action = self._choose_red_action(target, initiator, self.bayes_fa, state)
            if action == 0:  # Farming: normal generation
                reward = -0.1
            elif action == 1:  # Attacking: harmful generation
                generation_is_attack = True
                reward = 10.0
            elif action == 2:  # Smearing: Red sets only Z[Red, Blue] to -3
                request_trust_override = -3.0
                reward = getattr(self, "smearing_reward", -.1)
            else:
                raise ValueError(f"Invalid Red-vs-Blue action: {action}")
            self._store_pending_transition(
                target, state, action, reward
            )

        # Red -> Red: each Red independently chooses Normal or Boosting.
        else:
            target_state = self._get_red_state(target, initiator, self.bayes_fa)
            initiator_state = self._get_red_state(initiator, target, self.bayes_wj)

            self._close_pending_transition(
                target, target_state, completed_transitions
            )
            self._close_pending_transition(
                initiator, initiator_state, completed_transitions
            )

            # Choose both actions before applying either feedback direction.
            target_action = self._choose_red_action(target, initiator, self.bayes_fa, target_state)
            initiator_action = self._choose_red_action(initiator, target, self.bayes_wj, initiator_state)

            if target_action == 0:  # Normal Request rating
                target_reward = 0.0
            elif target_action == 1:  # Boost only Z[target, initiator]
                request_trust_override = 3.0
                target_reward = 0.0
            else:
                raise ValueError(f"Invalid Red-vs-Red action: {target_action}")

            if initiator_action == 0:  # Normal Generation rating
                initiator_reward = 0.0
            elif initiator_action == 1:  # Boost only Z[initiator, target]
                generation_trust_override = 3.0
                initiator_reward = 0.0
            else:
                raise ValueError(f"Invalid Red-vs-Red action: {initiator_action}")

            self._store_pending_transition(
                target, target_state, target_action, target_reward
            )
            self._store_pending_transition(
                initiator, initiator_state, initiator_action, initiator_reward
            )

        try:
            if wj_uniform is not None:
                self.bayes_wj._forced_uniform = wj_uniform
            if fa_uniform is not None:
                self.bayes_fa._forced_uniform = fa_uniform

            # Target evaluates the input/request using the WJ reference.
            self._evaluate_and_update(
                epoch=epoch,
                evaluator_id=target,
                evaluated_id=initiator,
                bayes_model=self.bayes_wj,
                reference_kind="input",
                actual_action_is_attack=request_is_attack,
                trust_override=request_trust_override,
            )

            # Initiator evaluates the output/generation using the FA reference.
            self._evaluate_and_update(
                epoch=epoch,
                evaluator_id=initiator,
                evaluated_id=target,
                bayes_model=self.bayes_fa,
                reference_kind="output",
                actual_action_is_attack=generation_is_attack,
                trust_override=generation_trust_override,
            )
        finally:
            if wj_uniform is not None:
                self.bayes_wj._forced_uniform = None
            if fa_uniform is not None:
                self.bayes_fa._forced_uniform = None

        return completed_transitions

    def _emit(self, kind, record):
        sink = getattr(self, 'event_sink', None)
        if sink:
            sink(kind, record)
        if getattr(self, 'retain_logs', True) and kind == 'feedback':
            self.feedback_log.append(record)
        if not hasattr(self, 'counts'):
            self.counts = dict(decisions=0, attacks=0, eligible=0, transitions=0, feedback=0)
            self.total_return = 0.
            self.discounted_returns = {}
            self.red_steps = {}
        if kind == 'decision':
            self.counts['decisions'] += 1
            self.counts['eligible'] += int(record['eligible'])
            self.counts['attacks'] += int(record['eligible'] and record['action']==1)
        elif kind == 'transition':
            self.counts['transitions'] += 1
            self.total_return += record['reward']
            red = record['red_id']
            step = self.red_steps.get(red, 0)
            self.discounted_returns[red] = self.discounted_returns.get(red, 0.) + self.agent.gamma**step*record['reward']
            self.red_steps[red] = step+1
        elif kind == 'feedback':
            self.counts['feedback'] += 1

    def _commit(self, batch, truncated=False, collapse=False):
        for red, state, action, reward, nxt, done in batch:
            self._emit('transition', dict(epoch=self.current_epoch, red_id=int(red),
                       state=state, action=int(action), reward=float(reward), next_state=nxt,
                       terminated=bool(done), truncated=bool(truncated), network_collapse=collapse))
        if self.retain_logs:
            self.transition_log.extend(batch)
        if self.training and batch:
            self.agent.learn_batch([row[1:] for row in batch])

    def _truncate_pending(self):
        if not self.pending_red_transitions:
            return
        if self.env.A.sum() <= 1:
            batch = [(r,p['state'],p['action'],p['reward'],p['state'],True)
                     for r,p in self.pending_red_transitions.items()]
            self.pending_red_transitions.clear()
            self._commit(batch, collapse=True)
            return
        # An independent continuation supplies actual next decision states.
        # It never updates Q or contributes rewards/events to the measured horizon.
        shadow = copy.copy(self)
        shadow.env = copy.deepcopy(self.env)
        shadow.rng = copy.deepcopy(self.rng)
        shadow.bayes_wj = copy.deepcopy(self.bayes_wj)
        shadow.bayes_fa = copy.deepcopy(self.bayes_fa)
        shadow.agent = copy.deepcopy(self.agent)
        shadow.pending_red_transitions = {}
        shadow.event_sink = None
        shadow.retain_logs = False
        shadow.counts = dict.fromkeys(self.counts, 0)
        wanted = set(self.pending_red_transitions)
        states = {}
        original = shadow._close_pending_transition
        def capture(red, state, completed):
            if red in wanted and red not in states:
                states[red] = state
            original(red, state, completed)
        shadow._close_pending_transition = capture
        for initiator, target, wj_uniform, fa_uniform in shadow._epoch_schedule():
            shadow._run_interaction(self.current_epoch+1, initiator, target, wj_uniform, fa_uniform)
            if wanted <= states.keys():
                break
        if wanted != states.keys():
            raise RuntimeError('Missing truncation bootstrap states')
        batch = [(r,p['state'],p['action'],p['reward'],states[r],False)
                 for r,p in self.pending_red_transitions.items()]
        self.pending_red_transitions.clear()
        self._commit(batch, truncated=True)

    def run(self):
        self.current_epoch = -1
        self._emit('start', dict(seed=self.seed, policy_seed=self.policy_seed, training=self.training,
                                identities=self.env.identities.tolist(), pretrusted=self.env.pretrusted.tolist()))
        isolation = [None]*self.env.num_nodes
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            schedule = self._epoch_schedule()
            if len(schedule) <= 1:
                break
            epoch_batch = []
            for initiator, target, wj_uniform, fa_uniform in schedule:
                epoch_batch.extend(self._run_interaction(epoch, initiator, target, wj_uniform, fa_uniform))
            diagnostics = self.env.update_global_trust()
            if not diagnostics['converged']:
                raise RuntimeError('EigenTrust nonconvergence')
            newly_dead = self.env.apply_eradication(self.n_std)
            for node in newly_dead:
                isolation[node] = epoch+1
            epoch_batch.extend(self._finalize_dead_red_transitions(newly_dead))
            self._commit(epoch_batch)
            arrays = [self.env.Z_latent,self.env.Red_Est_Z,self.env.C_rel,self.env.t,self.agent.q_table]
            if not all(np.isfinite(x).all() for x in arrays):
                raise FloatingPointError('Nonfinite simulation state')
            reds = np.flatnonzero((self.env.identities==1) & (self.env.A==1))
            errors = [abs(self.env.Red_Est_Z[r,j]-self.env.Z_latent[j,r])
                      for r in reds for j in np.flatnonzero(self.env.A) if j!=r]
            self._emit('epoch', dict(epoch=epoch, alive=self.env.A.tolist(), trust=self.env.t.tolist(),
                       estimate_mae=float(np.mean(errors)) if errors else None,
                       eradication=self.env.eradication_diagnostics, **diagnostics))
            # Stop only after all red terminal transitions were committed.
            if not len(reds):
                break
        self._truncate_pending()
        if self.counts['decisions'] != self.counts['transitions']:
            raise RuntimeError('Lost transition')
        if self.training:
            self.agent.decay_epsilon()
        summary = dict(**self.counts, total_return=self.total_return,
                       discounted_return=sum(self.discounted_returns.values()),
                       epochs=max(0,self.current_epoch+1), isolation_times=isolation,
                       alive=self.env.A.tolist(), identities=self.env.identities.tolist(),
                       attack_rate=self.counts['attacks']/self.counts['eligible'] if self.counts['eligible'] else None)
        self._emit('episode', summary)
        return summary

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the federated-trust simulation."
    )
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--malign", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--reference-version",
        default="qwen3.7-plus",
        help=(
            "Reference dataset suffix shared by the WJ/FA distribution and "
            "delta-V files (default: qwen3.7-plus)."
        ),
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prior",
        help="Directory containing the reference CSV files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sim = Simulation(
        num_nodes=args.num_nodes,
        malign=args.malign,
        epochs=args.epochs,
        reference_version=args.reference_version,
        prior_dir=args.prior_dir,
    )
    sim.run()
