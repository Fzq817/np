import csv
import copy
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from env import FastNPFSEnvironment
from evaluation import (
    rollout_policy_once, create_fixed_eval_realization,
    build_realization_from_json,
)

def compute_pgps_score(
    makespan: float, energy: float, delay: float,
    pref: np.ndarray, e_ref: float, beta: float,
    norm_stats: dict = None,
) -> tuple:
    if norm_stats is not None:
        ms_norm = _safe_normalize(makespan, norm_stats['ms_min'], norm_stats['ms_max'])
        en_norm = _safe_normalize(energy, norm_stats['en_min'], norm_stats['en_max'])
        dl_norm = _safe_normalize(delay, norm_stats['dl_min'], norm_stats['dl_max'])
    else:
        ms_norm, en_norm, dl_norm = makespan, energy, delay

    phi = -(pref[0] * ms_norm + pref[1] * en_norm + pref[2] * dl_norm)

    if norm_stats is not None:
        en_range = norm_stats['en_max'] - norm_stats['en_min']
        if en_range < 1e-8:
            penalty = 0.0
        else:
            penalty = beta * abs(energy - e_ref) / (en_range + 1e-8)
    else:
        penalty = beta * abs(energy - e_ref)

    score = phi - penalty
    return phi, penalty, score

def _safe_normalize(value, v_min, v_max):
    if abs(v_max - v_min) < 1e-8:
        return 0.0
    return (value - v_min) / (v_max - v_min + 1e-8)

def _maybe_cuda_sync(*modules):
    if not torch.cuda.is_available():
        return
    for m in modules:
        if m is None:
            continue
        try:
            p = next(m.parameters(), None)
        except Exception:
            p = None
        if p is not None and p.is_cuda:
            torch.cuda.synchronize()
            return

def normalize_candidate_objectives(results: list) -> tuple:
    if len(results) == 0:
        return results, {}

    ms_vals = [r['makespan'] for r in results]
    en_vals = [r['energy'] for r in results]
    dl_vals = [r['delay'] for r in results]

    norm_stats = {
        'ms_min': min(ms_vals), 'ms_max': max(ms_vals),
        'en_min': min(en_vals), 'en_max': max(en_vals),
        'dl_min': min(dl_vals), 'dl_max': max(dl_vals),
    }

    for r in results:
        r['ms_norm'] = _safe_normalize(r['makespan'], norm_stats['ms_min'], norm_stats['ms_max'])
        r['en_norm'] = _safe_normalize(r['energy'], norm_stats['en_min'], norm_stats['en_max'])
        r['dl_norm'] = _safe_normalize(r['delay'], norm_stats['dl_min'], norm_stats['dl_max'])

    return results, norm_stats

class BatchedPGPSRunner:
    def __init__(self, encoder, actor_critic, env_builder_fn, fixed_realization,
                 pref, num_samples=16, max_steps=400, residual_snapshot=None):
        self.encoder = encoder
        self.actor_critic = actor_critic
        self.env_builder_fn = env_builder_fn
        self.fixed_realization = fixed_realization
        self.pref = pref
        self.pref_tensor = torch.FloatTensor(pref)
        self.num_samples = num_samples
        self.max_steps = max_steps
        self.residual_snapshot = residual_snapshot

    def _collate_active_states(self, states, active_indices):
        features_list = [states[i]['node_features'].squeeze(0) for i in active_indices]
        adj_list = [states[i]['adj_matrix'].squeeze(0) for i in active_indices]
        types_list = [states[i]['node_types'].squeeze(0) for i in active_indices]

        max_nodes = max(f.size(0) for f in features_list)

        padded_f, padded_a, padded_t = [], [], []
        for f, a, t in zip(features_list, adj_list, types_list):
            pad_n = max_nodes - f.size(0)
            if pad_n > 0:
                f = F.pad(f, (0, 0, 0, pad_n))
                a = F.pad(a, (0, pad_n, 0, pad_n))
                t = F.pad(t, (0, pad_n))
            padded_f.append(f.unsqueeze(0))
            padded_a.append(a.unsqueeze(0))
            padded_t.append(t.unsqueeze(0))

        batch_f = torch.cat(padded_f, dim=0)
        batch_a = torch.cat(padded_a, dim=0)
        batch_t = torch.cat(padded_t, dim=0)

        B_active = len(active_indices)
        batch_pref = self.pref_tensor.unsqueeze(0).expand(B_active, -1)

        return batch_f, batch_a, batch_t, batch_pref

    def run_parallel(self):
        envs = []
        init_mode = "residual_snapshot" if self.residual_snapshot is not None else "episode_start"
        for i in range(self.num_samples):
            env = self.env_builder_fn(self.fixed_realization)
            if self.residual_snapshot is not None:
                env.load_residual_state_snapshot(copy.deepcopy(self.residual_snapshot))
            else:
                env.reset()
            envs.append(env)

        states = [env._get_state() for env in envs]
        dones = [False] * self.num_samples
        steps = [0] * self.num_samples
        infos = [{}] * self.num_samples

        with torch.no_grad():
            for _ in range(self.max_steps):
                active_indices = [i for i in range(self.num_samples) if not dones[i]]
                if not active_indices:
                    break

                valid_actions_map = {}
                still_active = []
                for i in active_indices:
                    va = envs[i].get_valid_actions()
                    if va is None:
                        dones[i] = True
                    else:
                        valid_actions_map[i] = va
                        still_active.append(i)

                if not still_active:
                    break

                batch_f, batch_a, batch_t, batch_pref = self._collate_active_states(
                    states, still_active)

                _, batch_embed = self.encoder(batch_f, batch_a, batch_t, batch_pref)

                va_list = [valid_actions_map[i] for i in still_active]
                batch_actions = self.actor_critic.get_action_batch(
                    batch_embed, batch_pref, va_list, deterministic=False)

                for idx_in_batch, env_idx in enumerate(still_active):
                    action = batch_actions[idx_in_batch]
                    state, _, done, info = envs[env_idx].step(action)
                    states[env_idx] = state
                    dones[env_idx] = done
                    steps[env_idx] += 1
                    infos[env_idx] = info

        results = []
        for i in range(self.num_samples):
            info = infos[i]
            ms = info.get('makespan', 0)
            en = info.get('energy', 0)
            dl = info.get('delay_cost', 0)
            results.append({
                'makespan': ms,
                'energy': en,
                'delay': dl,
                'delay_cost': dl,
                'objectives': np.array([ms, en, dl]),
                'steps': steps[i],
                'done': dones[i],
                'schedule_info': envs[i].get_schedule_info() if envs[i].record_details else None,
                'info': info,
                'sample_id': i,
                'init_mode': init_mode,
            })

        return results

def run_pgps_inference(
    encoder, actor_critic, env_builder_fn,
    pref: np.ndarray, e_ref: float, beta: float = 1.0,
    num_samples: int = 16, max_steps: int = 400,
    fixed_realization: dict = None, residual_snapshot: dict = None,
):
    encoder.eval()
    actor_critic.eval()

    runner = BatchedPGPSRunner(
        encoder=encoder,
        actor_critic=actor_critic,
        env_builder_fn=env_builder_fn,
        fixed_realization=fixed_realization,
        pref=pref,
        num_samples=num_samples,
        max_steps=max_steps,
        residual_snapshot=residual_snapshot,
    )

    _maybe_cuda_sync(encoder, actor_critic)
    t0_total = time.perf_counter()

    all_trajectory_results = runner.run_parallel()

    _maybe_cuda_sync(encoder, actor_critic)
    t1_rollout_end = time.perf_counter()

    all_trajectory_results, norm_stats = normalize_candidate_objectives(all_trajectory_results)

    scores = []
    for r in all_trajectory_results:
        phi, penalty, score = compute_pgps_score(
            makespan=r['makespan'],
            energy=r['energy'],
            delay=r['delay'],
            pref=pref,
            e_ref=e_ref,
            beta=beta,
            norm_stats=norm_stats,
        )
        r['phi'] = phi
        r['proximity_penalty'] = penalty
        r['pgps_score'] = score
        scores.append(score)

    best_index = int(np.argmax(scores))

    _maybe_cuda_sync(encoder, actor_critic)
    t2_scoring_end = time.perf_counter()

    rollout_wall_time_sec = t1_rollout_end - t0_total
    rerank_wall_time_sec = t2_scoring_end - t1_rollout_end
    total_pgps_wall_time_sec = t2_scoring_end - t0_total

    steps_list = [int(r.get('steps', 0)) for r in all_trajectory_results]
    if len(steps_list) > 0:
        steps_arr = np.asarray(steps_list, dtype=np.float64)
        mean_steps = float(steps_arr.mean())
        std_steps = float(steps_arr.std(ddof=0))
        min_steps = int(steps_arr.min())
        max_steps_seen = int(steps_arr.max())
        total_steps = int(steps_arr.sum())
    else:
        mean_steps = std_steps = 0.0
        min_steps = max_steps_seen = total_steps = 0

    encoder.train()
    actor_critic.train()

    return {
        'best_trajectory_result': all_trajectory_results[best_index],
        'all_trajectory_results': all_trajectory_results,
        'best_index': best_index,
        'scores': scores,
        'e_ref': e_ref,
        'beta': beta,
        'pref': pref,
        'norm_stats': norm_stats,
        'from_residual': residual_snapshot is not None,
        'init_mode': "residual_snapshot" if residual_snapshot is not None else "episode_start",
        'rollout_wall_time_sec': rollout_wall_time_sec,
        'rerank_wall_time_sec': rerank_wall_time_sec,
        'total_pgps_wall_time_sec': total_pgps_wall_time_sec,
        'steps_list': steps_list,
        'mean_steps': mean_steps,
        'std_steps': std_steps,
        'min_steps': min_steps,
        'max_steps': max_steps_seen,
        'total_steps': total_steps,
    }

def save_pgps_candidates_csv(results: list, csv_path: str, best_index: int = -1) -> None:
    import os
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    fieldnames = [
        'sample_id', 'makespan', 'energy', 'delay',
        'ms_norm', 'en_norm', 'dl_norm',
        'phi', 'proximity_penalty', 'pgps_score',
        'steps', 'done', 'is_best', 'init_mode',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results):
            row = {
                'sample_id': r.get('sample_id', i),
                'makespan': f"{r['makespan']:.4f}",
                'energy': f"{r['energy']:.4f}",
                'delay': f"{r['delay']:.4f}",
                'ms_norm': f"{r.get('ms_norm', 0):.6f}",
                'en_norm': f"{r.get('en_norm', 0):.6f}",
                'dl_norm': f"{r.get('dl_norm', 0):.6f}",
                'phi': f"{r.get('phi', 0):.6f}",
                'proximity_penalty': f"{r.get('proximity_penalty', 0):.6f}",
                'pgps_score': f"{r.get('pgps_score', 0):.6f}",
                'steps': int(r.get('steps', 0)),
                'done': 1 if r.get('done', False) else 0,
                'is_best': 1 if i == best_index else 0,
                'init_mode': r.get('init_mode', 'episode_start'),
            }
            writer.writerow(row)

    print(f"[PGPS] Candidate trajectory results saved to: {csv_path}")

def evaluate_with_pgps(
    encoder, actor_critic, data_array, sequence,
    pref: np.ndarray, e_ref: float, beta: float = 1.0,
    num_samples: int = 16, seed: int = 12345, max_steps: int = 400,
    delay_reward_scale: float = 1.0,
    objective_ranges: dict = None, csv_path: str = None,
    dynamic_config: dict = None, residual_snapshot: dict = None,
):
    init_mode = "residual_snapshot" if residual_snapshot is not None else "episode_start"

    print(f"\n{'='*80}")
    print(f"PGPS Inference (Post-hoc Guided Policy Search)")
    print(f"{'='*80}")
    print(f"  Preference omega = {pref}")
    print(f"  Reference energy E_ref = {e_ref}")
    print(f"  Steering beta  = {beta}")
    print(f"  Candidate trajectories N = {num_samples}")
    print(f"  Init mode      = {init_mode}")
    print(f"  (offline paper experiments use init_mode='episode_start')")
    print(f"{'='*80}")

    dyn_cfg = dynamic_config or {}

    realization = create_fixed_eval_realization(
        data_array, sequence,
        delay_reward_scale=delay_reward_scale,
        seed=seed, dynamic_config=dyn_cfg,
    )

    torch.manual_seed(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)

    if objective_ranges is None:
        objective_ranges = {'makespan': 100.0, 'energy': 100.0, 'delay': 50.0}

    def env_builder_fn(fixed_real):
        return FastNPFSEnvironment(
            data_array, sequence,
            objective_ranges=objective_ranges,
            record_details=True,
            delay_reward_scale=delay_reward_scale,
            deterministic_eval=True,
            instance_realization=fixed_real,
            dynamic_config=dyn_cfg,
        )

    pgps_result = run_pgps_inference(
        encoder=encoder, actor_critic=actor_critic,
        env_builder_fn=env_builder_fn,
        pref=pref, e_ref=e_ref, beta=beta,
        num_samples=num_samples, max_steps=max_steps,
        fixed_realization=realization,
        residual_snapshot=residual_snapshot,
    )

    best = pgps_result['best_trajectory_result']
    print(f"\n  PGPS Best: MS={best['makespan']:.2f}, "
          f"EN={best['energy']:.2f}, DL={best['delay']:.2f}, "
          f"score={best['pgps_score']:.6f}")

    print(f"\n  [PGPS Runtime]")
    print(f"    Total PGPS wall time : {pgps_result['total_pgps_wall_time_sec']:.4f} s")
    print(f"    Rollout  wall time   : {pgps_result['rollout_wall_time_sec']:.4f} s")
    print(f"    Rerank   wall time   : {pgps_result['rerank_wall_time_sec']:.4f} s")
    print(f"    Mean candidate steps : {pgps_result['mean_steps']:.2f} "
          f"(std={pgps_result['std_steps']:.2f}, "
          f"min={pgps_result['min_steps']}, max={pgps_result['max_steps']})")
    print(f"    Total candidate steps: {pgps_result['total_steps']}")

    if csv_path is not None:
        save_pgps_candidates_csv(
            pgps_result['all_trajectory_results'],
            csv_path, best_index=pgps_result['best_index'],
        )

    print(f"{'='*80}")
    return pgps_result

def evaluate_with_pgps_from_checkpoint(
    checkpoint_path: str, instance_path: str,
    pref: np.ndarray, e_ref: float = None, beta: float = 1.0,
    num_samples: int = 16, seed: int = 12345, max_steps: int = 400,
    device: str = "cpu", csv_path: str = None,
) -> dict:
    from checkpoint_utils import restore_models_from_checkpoint
    from utils import load_instance_json

    encoder, actor_critic, ckpt = restore_models_from_checkpoint(
        checkpoint_path, device=device,
    )
    dynamic_config = ckpt.get('dynamic_config', {})

    instance = load_instance_json(instance_path)
    data_array = instance['base_processing_times']

    sequence = ckpt.get('sequence', None)
    if sequence is None:
        from trainer import WorkloadBasedMachineOrdering
        opt = WorkloadBasedMachineOrdering(data_array, num_restarts=2)
        sequence, _ = opt.optimize()

    if e_ref is None:
        realization = build_realization_from_json(instance)
        env_temp = FastNPFSEnvironment(
            data_array, sequence,
            delay_reward_scale=1.0,
            deterministic_eval=True,
            instance_realization=realization,
            dynamic_config=dynamic_config,
        )
        temp_state = env_temp.reset()
        done = False
        steps = 0
        info = {}
        while not done and steps < max_steps:
            va = env_temp.get_valid_actions()
            if va:
                action = {
                    'job': va['jobs'][random.randint(0, len(va['jobs']) - 1)],
                    'factory': random.choice(va['factories']),
                    'speed': random.choice(va['speeds']),
                }
            else:
                action = {'job': 0, 'factory': 0, 'speed': 2}
            _, _, done, info = env_temp.step(action)
            steps += 1
        e_ref = info.get('energy', 15000.0)
        print(f"[PGPS] Auto-estimated E_ref = {e_ref:.2f}")

    objective_ranges = ckpt.get('objective_ranges', None)
    if not objective_ranges:
        objective_ranges = {'makespan': 100.0, 'energy': 100.0, 'delay': 50.0}

    return evaluate_with_pgps(
        encoder=encoder,
        actor_critic=actor_critic,
        data_array=data_array,
        sequence=sequence,
        pref=pref,
        e_ref=e_ref,
        beta=beta,
        num_samples=num_samples,
        seed=seed,
        max_steps=max_steps,
        delay_reward_scale=ckpt.get('config', {}).get('delay_reward_scale', 1.0),
        objective_ranges=objective_ranges,
        csv_path=csv_path,
        dynamic_config=dynamic_config,
        residual_snapshot=None,
    )
