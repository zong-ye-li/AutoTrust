"""Validate a completed pilot run and write formal-run readiness evidence."""
import argparse
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from experiment_runner import condition_specs, config_fingerprint, fingerprints, load_config
from fixed_policy_baselines import baseline_names


def load_json(path):
    return json.loads(path.read_text())


def formal_episode_count(config=None):
    if config is None:
        config = dict(
            pretrust_values=[0, 3], variants=['full', 'no-manipulation'],
            training_seeds=5, train_episodes=200, validation_every=20,
            validation_seeds=3, evaluation_seeds=30,
        )
    conditions = len(condition_specs(config))
    if config.get('execution_kind') == 'baseline-only':
        return conditions * len(baseline_names(config)) * config['evaluation_seeds']
    variants = len(config['variants'])
    training_seeds = config['training_seeds']
    train_episodes = config['train_episodes']
    validation_points = train_episodes // config['validation_every']
    validation_seeds = config['validation_seeds']
    evaluation_seeds = config['evaluation_seeds']
    baselines = len(baseline_names(config))
    learned = conditions * variants * training_seeds * (
        train_episodes + validation_points * validation_seeds + evaluation_seeds
    )
    baseline = conditions * baselines * evaluation_seeds
    return learned + baseline


def formal_max_epoch_count(config):
    """Conservative bound: every planned episode reaches its horizon."""
    return formal_episode_count(config) * config.get('horizon', 1000)


def iter_events(run_dir):
    for path in run_dir.glob('**/events.jsonl.gz'):
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            for line_no, line in enumerate(stream, 1):
                yield path, line_no, json.loads(line)


def validate_summary(path, summary):
    problems = []
    if summary['decisions'] != summary['transitions']:
        problems.append(f'{path}: decisions/transitions mismatch')
    for key in ('seconds', 'q_max_change', 'greedy_agreement'):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f'{path}: nonfinite {key}')
    attack_rate = summary.get('attack_rate')
    if attack_rate is not None and not (isinstance(attack_rate, (int, float)) and 0 <= attack_rate <= 1):
        problems.append(f'{path}: invalid attack_rate')
    for key in ('alive', 'identities', 'isolation_times'):
        if key not in summary:
            problems.append(f'{path}: missing {key}')
    return problems


def validate_event(path, line_no, event, include_estimated_postattack_credence=True):
    problems = []
    kind = event.get('kind')
    if kind in ('decision', 'transition'):
        action = event.get('action')
        state = event.get('state')
        if action not in (0, 1, 2):
            problems.append(f'{path}:{line_no}: invalid action {action}')
        expected_state_length = 5 if include_estimated_postattack_credence else 4
        if not isinstance(state, list) or len(state) != expected_state_length:
            problems.append(f'{path}:{line_no}: invalid state')
        elif state[2] == 1 and action == 2:
            problems.append(f'{path}:{line_no}: smear action against Red target')
    if kind == 'transition':
        if 'terminated' not in event or 'truncated' not in event:
            problems.append(f'{path}:{line_no}: transition missing terminal flags')
    if kind == 'feedback':
        for key in ('observed_score', 'observed_v'):
            value = event.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f'{path}:{line_no}: nonfinite {key}')
    return problems


def check(run_dir, formal_config=None):
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / 'manifest.json')
    status = load_json(run_dir / 'status.json')
    config = manifest['config']
    problems = []

    if status.get('status') != 'complete':
        problems.append('pilot status is not complete')
    if config.get('mode') != 'pilot':
        problems.append('manifest is not a pilot run')
    if config.get('training_seeds', 0) < 5:
        problems.append('pilot used fewer than five training seeds')
    if status.get('decisions') != status.get('transitions'):
        problems.append('aggregate decisions/transitions mismatch')
    if manifest.get('sha256') != fingerprints():
        problems.append('source or PMF fingerprint is stale')

    summaries = list(run_dir.glob('**/summary.json'))
    if not summaries:
        problems.append('no episode summaries found')
    for path in summaries:
        if '.partial' in str(path):
            problems.append(f'incomplete partial summary present: {path}')
            continue
        problems.extend(validate_summary(path, load_json(path)))

    event_count = 0
    for path, line_no, event in iter_events(run_dir):
        event_count += 1
        problems.extend(validate_event(
            path, line_no, event,
            config.get('include_estimated_postattack_credence', True),
        ))
    if event_count == 0:
        problems.append('no structured events found')

    pilot_episodes = max(1, status.get('episodes', len(summaries)))
    bytes_per_episode = status.get('bytes', 0) / pilot_episodes
    bytes_per_epoch = status.get('bytes', 0) / max(1, status.get('epochs', 0))
    formal_episodes = formal_episode_count(formal_config)
    formal_max_epochs = formal_max_epoch_count(formal_config or {})
    formal_bytes_estimate = int(bytes_per_episode * formal_episodes * 1.5)
    formal_max_bytes_estimate = int(bytes_per_epoch * formal_max_epochs * 1.5)
    free_bytes = status.get('free_bytes', 0)
    if free_bytes <= formal_max_bytes_estimate:
        problems.append('free disk does not clear 1.5x horizon-safe formal-output extrapolation')

    result = dict(
        passed=not problems,
        checked_at=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run_dir),
        sha256=fingerprints(),
        pilot=dict(status=status, config=config, summaries=len(summaries), events=event_count),
        extrapolation=dict(
            formal_episodes=formal_episodes,
            formal_bytes_estimate=formal_bytes_estimate,
            formal_max_epochs=formal_max_epochs,
            formal_max_bytes_estimate=formal_max_bytes_estimate,
            free_bytes=free_bytes,
            seconds_per_epoch=status.get('seconds_per_epoch'),
        ),
        problems=problems,
    )
    if formal_config is not None:
        result['formal_config_sha256'] = config_fingerprint(formal_config)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--formal-config', type=Path,
                        help='Formal JSON config whose output size and fingerprint are validated')
    args = parser.parse_args()
    formal_config = load_config(args.formal_config) if args.formal_config else None
    if formal_config is not None and formal_config['mode'] != 'formal':
        parser.error('--formal-config must have mode="formal"')
    result = check(args.run_dir, formal_config)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
