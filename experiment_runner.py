"""Streaming, transactional episode runner. Resume only at committed episode boundaries."""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback
import numpy as np
from Agent import SharedRedSyndicateAgent
from fixed_policy_baselines import FrozenBaseline, baseline_names
from Simulation import Simulation

ROOT = Path(__file__).resolve().parent
THRESHOLD_POPULATIONS = ('all-alive', 'non-anchor-alive')


def config_fingerprint(config):
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf8')).hexdigest()


def validate_config(config):
    """Reject malformed experiment matrices before any output is created."""
    required = (
        'mode', 'num_nodes', 'malign', 'n_std', 'reference_version', 'smoothing',
        'variants', 'training_seeds', 'train_episodes', 'horizon',
        'evaluation_seeds', 'validation_seeds', 'validation_every',
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Missing config keys: {missing}')
    if config['mode'] not in ('pilot', 'formal'):
        raise ValueError('Invalid config mode')
    if config.get('execution_kind') not in (None, 'baseline-only'):
        raise ValueError('Invalid execution_kind')
    if not isinstance(config.get('include_estimated_postattack_credence', True), bool):
        raise ValueError('include_estimated_postattack_credence must be boolean')
    if not isinstance(config['variants'], list) or not config['variants']:
        raise ValueError('Config must include variants')
    if any(variant not in ('full', 'no-manipulation') for variant in config['variants']):
        raise ValueError('Invalid variant')
    baseline_names(config)

    conditions = config.get('conditions')
    if conditions is None:
        values = config.get('pretrust_values')
        if not isinstance(values, list) or not values:
            raise ValueError('Config must include pretrust_values or conditions')
        return
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('Conditions must be a non-empty list')
    identifiers = set()
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError('Each condition must be an object')
        required_condition = ('id', 'pretrust', 'n_std', 'threshold_population')
        missing = [key for key in required_condition if key not in condition]
        if missing:
            raise ValueError(f'Condition missing keys: {missing}')
        identifier = condition['id']
        if not isinstance(identifier, str) or not identifier or '/' in identifier or '\\' in identifier:
            raise ValueError('Invalid condition id')
        if identifier in identifiers:
            raise ValueError(f'Duplicate condition id: {identifier}')
        identifiers.add(identifier)
        if not isinstance(condition['pretrust'], int) or condition['pretrust'] < 0:
            raise ValueError('Invalid condition pretrust')
        if not np.isfinite(condition['n_std']) or condition['n_std'] < 0:
            raise ValueError('Invalid condition n_std')
        if condition['threshold_population'] not in THRESHOLD_POPULATIONS:
            raise ValueError('Invalid condition threshold_population')


def condition_specs(config):
    """Expand either the legacy pretrust list or declarative condition matrix."""
    if config.get('conditions') is not None:
        return [dict(condition) for condition in config['conditions']]
    population = config.get('threshold_population', 'all-alive')
    if population not in THRESHOLD_POPULATIONS:
        raise ValueError('Invalid threshold_population')
    return [
        dict(id=f'pretrust-{pretrust}', pretrust=pretrust,
             n_std=config['n_std'], threshold_population=population)
        for pretrust in config['pretrust_values']
    ]


def load_config(path):
    path = Path(path)
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON config: {path}') from exc
    if not isinstance(config, dict):
        raise ValueError('Config root must be an object')
    validate_config(config)
    return config

def dump(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def fingerprints():
    files = [ROOT/n for n in (
        'Agent.py', 'Environment.py', 'TrustBayes.py', 'Simulation.py',
        'experiment_runner.py', 'readiness_check.py',
        'fixed_policy_baselines.py', 'frozen_baseline_runner.py',
    ) if (ROOT/n).is_file()]
    files += sorted((ROOT/'prior').glob('distribution_*.csv'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

class EventWriter:
    def __init__(self, directory, detail):
        self.file = gzip.open(directory/'events.jsonl.gz', 'wt', encoding='utf8')
        self.detail = detail
    def __call__(self, kind, data):
        if self.detail or kind not in ('feedback','decision','transition'):
            self.file.write(json.dumps(dict(kind=kind, **data), default=lambda x: x.item(), allow_nan=False)+'\n')
    def close(self): self.file.close()

def masked_greedy_actions(agent, q_table):
    actions = np.zeros(q_table.shape[:-1], dtype=int)
    for state in np.ndindex(actions.shape):
        valid = agent._get_valid_actions(state[2])
        values = [q_table[state + (action,)] for action in valid]
        actions[state] = valid[int(np.argmax(values))]
    return actions

def episode(directory, config, env_seed, policy_seed, agent, training):
    if directory.exists():
        result = json.loads((directory/'summary.json').read_text())
        if training:
            agent, _ = SharedRedSyndicateAgent.load(directory/'policy.npz')
        return agent, result
    staging = directory.with_name(directory.name+'.partial')
    # Keep failed partial outputs; a retry gets a unique staging directory.
    if staging.exists():
        staging.rename(staging.with_name(staging.name+'.failed-'+str(time.time_ns())))
    staging.mkdir(parents=True)
    writer = EventWriter(staging, detail=not training or config.get('training_detail', False))
    started = time.perf_counter()
    before = agent.q_table.copy()
    try:
        sim = Simulation(num_nodes=config['num_nodes'], malign=config['malign'], epochs=config['horizon'],
                         seed=env_seed, policy_seed=policy_seed, pretrust=config['pretrust'],
                         n_std=config['n_std'], threshold_population=config.get('threshold_population', 'all-alive'),
                         reference_version=config['reference_version'],
                         include_estimated_postattack_credence=config.get('include_estimated_postattack_credence', True),
                         smoothing=config['smoothing'], training=training, agent=agent,
                         event_sink=writer, retain_logs=False)
        result = sim.run()
        if not training and not np.array_equal(before, agent.q_table):
            raise RuntimeError('Evaluation mutated Q')
        result.update(seconds=time.perf_counter()-started, env_seed=env_seed, policy_seed=policy_seed,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      q_max_change=float(np.max(np.abs(agent.q_table-before))),
                      greedy_agreement=float(np.mean(masked_greedy_actions(agent, before)==masked_greedy_actions(agent, agent.q_table))),
                      training=training, epsilon=agent.epsilon if training else 0.)
        writer.close()
        agent.save(staging/'policy.npz', dict(env_seed=env_seed, policy_seed=policy_seed))
        dump(staging/'summary.json', result)
        os.replace(staging, directory)
        return agent, result
    except BaseException:
        writer.close()
        dump(staging/'failure.json', dict(traceback=traceback.format_exc()))
        raise

def execute(out, config, stop_after=None):
    validate_config(config)
    if config.get('execution_kind') == 'baseline-only':
        raise ValueError('Use frozen_baseline_runner.py for baseline-only configs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock is released automatically on crash; no stale PID lock.
    import fcntl
    lock = (out/'runner.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Output directory is already in use: {out}') from None
    except BaseException:
        lock.close()
        raise
    try:
        manifest_path = out/'manifest.json'
        signature = fingerprints()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['config'] != config or manifest['sha256'] != signature:
                raise ValueError('Resume config/source/PMF mismatch')
        else:
            dump(manifest_path, dict(config=config, config_sha256=config_fingerprint(config), sha256=signature, python=sys.version,
                 numpy=np.__version__, platform=platform.platform(), created=datetime.now(timezone.utc).isoformat(),
                 seed_scheme='train=100000+10000*training_index+episode; validation=2000000+j; eval=3000000+j; policy=4000000+index',
                 resume='atomic episode boundary; incomplete episode is replayed from last committed checkpoint',
                 convergence='not established; inspect repeated validation and checkpoint diagnostics'))
        completed = 0
        started = time.perf_counter()
        for spec in condition_specs(config):
            cfg = dict(config, **spec)
            condition = out/spec['id']
            for variant in config['variants']:
                for index in range(config['training_seeds']):
                    policy_seed = 4000000+index
                    agent = SharedRedSyndicateAgent(
                        seed=policy_seed, manipulation=variant=='full',
                        include_estimated_postattack_credence=config.get('include_estimated_postattack_credence', True),
                    )
                    for ep in range(config['train_episodes']):
                        directory = condition/variant/f'seed-{index}'/f'train-{ep:05d}'
                        agent, result = episode(directory, cfg, 100000+10000*index+ep, policy_seed, agent, True)
                        completed += 1
                        if stop_after is not None and completed >= stop_after:
                            dump(out/'status.json', dict(status='paused', completed=completed))
                            return
                        if (ep+1)%config['validation_every']==0 or ep+1==config['train_episodes']:
                            for val in range(config['validation_seeds']):
                                frozen = copy.deepcopy(agent)
                                frozen.training=False
                                frozen.epsilon=0.
                                episode(condition/variant/f'seed-{index}'/f'validation-{ep:05d}-{val}', cfg,
                                        2000000+val, policy_seed, frozen, False)
                    for ev in range(config['evaluation_seeds']):
                        frozen = copy.deepcopy(agent)
                        frozen.training=False
                        frozen.epsilon=0.
                        episode(condition/variant/f'seed-{index}'/f'eval-{ev:05d}', cfg,
                                3000000+ev, policy_seed, frozen, False)
                    print(f'Completed condition={spec["id"]} {variant} seed={index}', flush=True)
            for baseline_index, name in enumerate(baseline_names(config)):
                for ev in range(config['evaluation_seeds']):
                    episode(condition/name/f'eval-{ev:05d}', cfg, 3000000+ev,
                            5000000 + 10000 * baseline_index + ev,
                            FrozenBaseline(
                                name, 5000000 + 10000 * baseline_index + ev,
                                include_estimated_postattack_credence=config.get('include_estimated_postattack_credence', True),
                            ), False)
        summaries = [json.loads(p.read_text()) for p in out.glob('**/summary.json') if '.partial' not in str(p)]
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        total_epochs = sum(x['epochs'] for x in summaries)
        dump(out/'status.json', dict(status='complete', episodes=len(summaries),
             seconds_this_invocation=time.perf_counter()-started, total_simulation_seconds=sum(x['seconds'] for x in summaries),
             epochs=total_epochs, bytes=size, free_bytes=shutil.disk_usage(out).free,
             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             seconds_per_epoch=sum(x['seconds'] for x in summaries)/max(1,total_epochs),
             decisions=sum(x['decisions'] for x in summaries), transitions=sum(x['transitions'] for x in summaries)))
    except BaseException:
        dump(out/'status.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
    finally:
        lock.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot','formal'], default='pilot')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--readiness', type=Path)
    parser.add_argument('--config', type=Path,
                        help='JSON experiment config; cannot be combined with --resume')
    args=parser.parse_args()
    if args.resume and args.config:
        parser.error('--config cannot be combined with --resume')
    if args.resume:
        config=json.loads((args.resume/'manifest.json').read_text())['config']
        out=args.resume
    elif args.config:
        config = load_config(args.config)
        study_id = config.get('study_id', args.config.stem)
        out = args.output or ROOT/'runs'/(study_id+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    else:
        pilot=args.mode=='pilot'
        config=dict(mode=args.mode, num_nodes=50, malign=.1, n_std=2.5,
                    reference_version='qwen3.7-plus', smoothing=0., pretrust_values=[0,3],
                    variants=['full','no-manipulation'], training_seeds=5,
                    train_episodes=2 if pilot else 200, horizon=40 if pilot else 1000,
                    evaluation_seeds=1 if pilot else 30, validation_seeds=1 if pilot else 3,
                    validation_every=1 if pilot else 20)
        out=args.output or ROOT/'runs'/(args.mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        if out.exists():
            raise ValueError('New run output already exists; use --resume')
    validate_config(config)
    if config['mode']=='formal':
        if not args.readiness:
            raise ValueError('Formal run requires readiness evidence')
        ready=json.loads(args.readiness.read_text())
        if not ready.get('passed') or ready['sha256']!=fingerprints():
            raise ValueError('Readiness gate missing or stale')
        expected_config_hash = ready.get('formal_config_sha256')
        if expected_config_hash is not None and expected_config_hash != config_fingerprint(config):
            raise ValueError('Readiness config fingerprint is stale')
    print(out, flush=True)
    execute(out,config)

if __name__=='__main__': main()
