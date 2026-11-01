# AutoTrust Simulation Code

The manuscript, **Automated Trust Management under LLM Monitoring Uncertainty: Adversarial Risk-Aware Deployment in a Decentralized Multi-Agent Network**, is accepted by TAAI 2026.

This repository is the clean GitHub delivery mirror for the paper code. It contains the simulator, 
formal experiment configurations, and the small derived monitor-reference inputs required by the unchanged simulator.

## Research context

This research was conducted at Nanyang University for the Global Connect Fellowship, with daily guidance from Dr. [Weiyang He](https://scholar.google.com/citations?user=XwVpc4MAAAAJ&hl=en) and supervision by Prof. [Chip Hong Chang](https://scholar.google.com/citations?user=sovM_aMAAAAJ&hl=en).

## Layout

- [`Li_He_Chang_2026_AutoTrust_Management_Manuscript_v1.pdf`](Li_He_Chang_2026_AutoTrust_Management_Manuscript_v1.pdf): manuscript accepted by TAAI 2026.
- `Agent.py`, `Environment.py`, `Simulation.py`, `TrustBayes.py`: core simulator.
- `fixed_policy_baselines.py`: fixed-policy comparators used by the paper.
- `experiment_runner.py`: configuration-driven experiment runner.
- `frozen_baseline_runner.py`: runner for baseline-only configurations.
- `readiness_check.py`: formal-run preflight check based on a local pilot output.
- `prior/`: binned score distributions and matching Delta-V validation tables.
- `configs/`: the fourteen formal configurations represented in the submitted
  paper's reported regimes. Pilot configurations are intentionally excluded.

## Monitor-reference CSV files

Read `data/README.md`.

The eight files in `prior/` are small derived numeric inputs. They do not
contain original prompts, raw dataset rows, API credentials, monitor outputs,
or experimental event logs.

| File pattern | Columns | Role in the simulator |
|---|---|---|
| `distribution_formai_*.csv` | `Score_Interval`, `Count_Benign`, `Count_Malicious`, and derived percentage/probability columns | The FormAI score-bin distribution. The simulator uses the two count columns to construct the distributions of benign and malicious scores. |
| `distribution_wildjailbreak_*.csv` | `Score_Interval`, `Count_Benign`, `Count_Malicious`, and derived percentage/probability columns | The WildJailbreak score-bin distribution. The simulator uses the two count columns in the same way. |
| `deltaV_lookup_table_formai_*.csv` | `Prior_c_B`, `Prior_c_M`, `Delta_V` | Validation grid for the corresponding FormAI distribution and monitor version. |
| `deltaV_lookup_table_wildjailbreak_*.csv` | `Prior_c_B`, `Prior_c_M`, `Delta_V` | Validation grid for the corresponding WildJailbreak distribution and monitor version. |

`TrustBayes.py` recomputes Delta-V from the distributions. `Simulation.py`
still loads the matching lookup table during initialization to verify that its
stored values agree with that calculation; therefore the distribution and
lookup files must travel as matching pairs.

The original FormAI and WildJailbreak records, prompts, monitor-score JSONL
files, and access credentials are not redistributed. The derived inputs needed
by this simulator are already in `prior/`; users do not need to obtain another
file merely to run the included simulation.

## Requirements and execution platform

Read `requirements.txt`.

Third-party packages: `numpy` and `pandas` are required.  
Platform requirements: Linux or WSL for `resource` package,
which is not available in native Windows Python.

## Paper Abstract

Decentralized multi-agent systems increasingly rely on shared information, yet repeated interactions allow locally plausible actions to accumulate into distributed harm. We propose an automated trust-management framework that converts noisy monitor scores into directed local trust, aggregates them into a global reputation via an EigenTrust-style process, and removes persistently low-reputation agents. 

Building on this framework, we study an adaptive Red syndicate using shared Q-learning to farm trust, attack benign agents, smear reputations, or boost malicious teammates. We evaluate pretrust, removal threshold population, defense calibration, and Red observation regimes on attacks, Red survival and eradication, and Blue false isolation.

In evaluated regimes, including pretrust anchors in threshold computation can prevent removal; excluding them restores operation but reveals an attack-false isolation trade-off. Full manipulation generally increases attacks and Red survival, while post-attack credence has limited aggregate impact despite affecting greedy action preferences. Results characterize adversarial trust dynamics under uncertain monitoring.
