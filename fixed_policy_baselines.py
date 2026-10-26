"""Frozen Red policies for the post-round-2 baseline extension.

These policies are deliberately not learned.  Their decisions use only the
Red agent's locally available estimate of counterpart credence and the fixed
monitor PMF moments supplied by ``Simulation``; they never inspect latent
trust, monitor draws, or pretrusted-anchor identities.
"""

from __future__ import annotations

from Agent import SharedRedSyndicateAgent


LEGACY_BASELINES = ("farm", "attack", "bernoulli", "threshold")
FROZEN_BASELINES = ("bernoulli-r0-anchor", "postattack-c90", "postattack-c95")
SUPPORTED_BASELINES = LEGACY_BASELINES + FROZEN_BASELINES


def baseline_names(config):
    """Return the explicitly configured baselines, retaining legacy defaults."""
    raw_names = config.get("baselines")
    if raw_names is None:
        names = LEGACY_BASELINES
    elif isinstance(raw_names, list):
        names = tuple(raw_names)
    else:
        raise ValueError("baselines must be a list")
    if len(names) != len(set(names)):
        raise ValueError("baselines must not contain duplicates")
    unknown = set(names) - set(SUPPORTED_BASELINES)
    if unknown:
        raise ValueError(f"Unsupported baselines: {sorted(unknown)}")
    return tuple(names)


class FrozenBaseline(SharedRedSyndicateAgent):
    """A deterministic or seeded-random fixed policy with no Q-learning."""

    def __init__(self, name, seed, include_estimated_postattack_credence=True):
        if name not in SUPPORTED_BASELINES:
            raise ValueError(f"Unsupported baseline: {name}")
        super().__init__(seed=seed, training=False, manipulation=False,
                         include_estimated_postattack_credence=include_estimated_postattack_credence)
        self.name = name

    def choose_action(self, state, context=None):
        """Choose Normal/Farming (0) or Attack/legacy Boosting (1).

        ``context`` is supplied by the simulator at the decision point.  The
        post-attack rules require the Red-side prediction of its counterpart's
        credence after one malicious action, not any hidden environment state.
        """
        if state[2]:  # Red teammate: new baselines are always Normal.
            return 1 if self.name == "threshold" else 0
        if self.name == "farm":
            return 0
        if self.name == "attack":
            return 1
        if self.name == "bernoulli":
            return int(self.rng.random() < 0.5)
        if self.name == "threshold":
            # Legacy threshold is defined by the same safe (>= .90) Red-side
            # post-attack prediction as the historical fifth state bin.  The
            # reduced learned-state ablation must not remove that simulator
            # calculation from fixed-policy baselines.
            if len(state) == 5:
                return int(state[4] == 2)
            if context is None or "post_attack_credence" not in context:
                raise ValueError("threshold requires Red-side post-attack credence context")
            return int(context["post_attack_credence"] >= 0.90)
        if self.name == "bernoulli-r0-anchor":
            return int(self.rng.random() < 0.1)
        if context is None or "post_attack_credence" not in context:
            raise ValueError(f"{self.name} requires Red-side post-attack credence context")
        floor = 0.90 if self.name == "postattack-c90" else 0.95
        return int(context["post_attack_credence"] >= floor)
