import os
import csv
import random
import numpy as np
import torch

from env import FastNPFSEnvironment
from utils import load_instance_json
from models import HeteroPreferenceGATEncoder, SimpleActorCritic
from data_splits import get_instance_id

def create_fixed_eval_realization(data_array, sequence, delay_reward_scale=1.0,
                                  seed=12345, dynamic_config=None):
    rng_state_np = np.random.get_state()
    rng_state_py = random.getstate()

    np.random.seed(seed)
    random.seed(seed)

    env_temp = FastNPFSEnvironment(
        data_array, sequence,
        delay_reward_scale=delay_reward_scale,
        dynamic_config=dynamic_config or {},
    )
    realization = env_temp.build_instance_realization()

    np.random.set_state(rng_state_np)
    random.setstate(rng_state_py)

    return realization

def build_realization_from_json(instance: dict) -> dict:
    if 'factory_times' in instance:
        factory_times = instance['factory_times']
        if not isinstance(factory_times[0], np.ndarray):
            factory_times = [np.array(ft) for ft in factory_times]
    else:
        base = np.array(instance['base_processing_times'])
        multipliers = np.array(instance['factory_time_multipliers'])
        factory_times = []
        for f in range(instance['num_factories']):
            factory_times.append(base * multipliers[f])

    priority_indices = instance.get('priority_indices', [])

    if 'job_meta' in instance:
        job_meta = instance['job_meta']
    else:
        num_jobs = instance['num_jobs']
        due_dates = np.array(instance['due_dates'])
        priority_weights = np.array(instance['priority_weights'])
        job_meta = []
        for i in range(num_jobs):
            is_hp = i in priority_indices
            job_meta.append({
                'id': i,
                'is_high_priority': is_hp,
                'priority': int(priority_weights[i]) if is_hp else 0,
                'due_date': float(due_dates[i]),
                'estimated_processing_time': float(due_dates[i] / 1.3),
                'buffer_factor': 1.3,
                'delay_cost': float(priority_weights[i]),
            })

    if 'machine_idle_power' in instance:
        machine_idle_power = np.array(instance['machine_idle_power'])
    else:
        rng = np.random.RandomState(instance.get('seed', 42))
        machine_idle_power = rng.uniform(
            0.2, 1.2,
            size=(instance['num_factories'], instance['num_machines'])
        )

    if 'machine_startup_energy' in instance:
        machine_startup_energy = np.array(instance['machine_startup_energy'])
    else:
        rng2 = np.random.RandomState(instance.get('seed', 42) + 1)
        machine_startup_energy = rng2.uniform(0.5, 1.5,
            size=(instance['num_factories'], instance['num_machines']))

    realization = {
        'factory_times': factory_times,
        'priority_indices': priority_indices,
        'job_meta': job_meta,
        'machine_idle_power': machine_idle_power,
        'machine_startup_energy': machine_startup_energy,
    }

    if 'arrival_times' in instance:
        realization['arrival_times'] = instance['arrival_times']

    if 'initial_job_mask' in instance:
        initial_ids = [i for i, m in enumerate(instance['initial_job_mask']) if m]
        dynamic_ids = [i for i, m in enumerate(instance['initial_job_mask']) if not m]
        realization['initial_job_ids'] = initial_ids
        realization['dynamic_job_ids'] = dynamic_ids
    elif 'arrival_times' in instance:
        arr = instance['arrival_times']
        realization['initial_job_ids'] = [jid for jid, t in arr.items() if t <= 0.0]
        realization['dynamic_job_ids'] = [jid for jid, t in arr.items() if t > 0.0]

    if 'arrival_times' in realization:
        arr = realization['arrival_times']
        initial_set = set(realization.get('initial_job_ids', []))
        for meta in realization['job_meta']:
            jid = meta['id']
            meta['arrival_time'] = arr.get(jid, arr.get(str(jid), 0.0))
            meta['is_dynamic'] = jid not in initial_set
            meta['arr_flag'] = 0 if jid in initial_set else 1

    return realization

def rollout_policy_once(
    encoder, actor_critic, env, pref,
    deterministic_action=False, collect_schedule=False, max_steps=None,
):
    if max_steps is None:
        max_steps = 400

    state = env.reset()
    done = False
    steps = 0
    pref_tensor = torch.FloatTensor(pref)
    info = {}

    with torch.no_grad():
        while not done and steps < max_steps:
            va = env.get_valid_actions()
            if va is None:
                break

            model_device = next(encoder.parameters()).device
            node_features = state['node_features'].to(model_device)
            adj_matrix = state['adj_matrix'].to(model_device)
            node_types = state['node_types'].to(model_device)
            pref_tensor = pref_tensor.to(model_device)

            _, embed = encoder(node_features, adj_matrix, node_types, pref_tensor)

            if embed.dim() == 1:
                embed = embed.unsqueeze(0)

            action, _, _, _ = actor_critic.get_action(
                embed, pref_tensor,
                valid_actions=[va] if va else None,
                deterministic=deterministic_action,
            )

            state, _, done, info = env.step(action)
            steps += 1

    ms = info.get('makespan', 0)
    en = info.get('energy', 0)
    dl = info.get('delay_cost', 0)

    result = {
        'makespan': ms,
        'energy': en,
        'delay': dl,
        'delay_cost': dl,
        'objectives': np.array([ms, en, dl]),
        'steps': steps,
        'done': done,
        'schedule_info': None,
        'info': info,
    }

    if collect_schedule or env.record_details:
        result['schedule_info'] = env.get_schedule_info()

    return result

DEFAULT_EVAL_PREFS = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.5, 0.5, 0.0]),
    np.array([0.5, 0.0, 0.5]),
    np.array([0.0, 0.5, 0.5]),
    np.array([0.33, 0.33, 0.34]),
]

def deterministic_multi_pref_eval(
    encoder, actor_critic, data_array, sequence, ranges,
    realization, max_steps=400, delay_reward_scale=1.0,
    eval_prefs=None, dynamic_config=None,
):
    if eval_prefs is None:
        eval_prefs = DEFAULT_EVAL_PREFS

    encoder.eval()
    actor_critic.eval()

    all_objectives = []

    for pref in eval_prefs:
        env = FastNPFSEnvironment(
            data_array, sequence,
            objective_ranges=ranges,
            record_details=False,
            delay_reward_scale=delay_reward_scale,
            deterministic_eval=True,
            instance_realization=realization,
            dynamic_config=dynamic_config or {},
        )
        result = rollout_policy_once(
            encoder, actor_critic, env, pref,
            deterministic_action=True, max_steps=max_steps,
        )
        ms = result['makespan']
        en = result['energy']
        dl = result['delay_cost']
        if ms > 0 and en > 0:
            all_objectives.append([ms, en, dl])

    if len(all_objectives) == 0:
        return [], 0.0, 0, 0, 0

    obj_array = np.array(all_objectives)
    avg_ms = float(np.mean(obj_array[:, 0]))
    avg_en = float(np.mean(obj_array[:, 1]))
    avg_dl = float(np.mean(obj_array[:, 2]))

    ref_point = obj_array.max(axis=0) * 1.1
    eval_hv = _hypervolume_from_points(obj_array, ref_point)

    return all_objectives, eval_hv, avg_ms, avg_en, avg_dl

def _hypervolume_from_points(points_array, ref_point=None):
    if len(points_array) == 0:
        return 0.0
    points = np.array(points_array)
    if ref_point is None:
        ref_point = points.max(axis=0) * 1.1
    ref_point = np.maximum(ref_point, points.max(axis=0) * 1.01)
    dominated = np.maximum(0, ref_point - points)
    if dominated.shape[1] == 3:
        hv = _hv_3d(dominated)
        max_hv = np.prod(ref_point)
        return hv / max_hv if max_hv > 0 else 0.0
    return 0.0

def _hv_3d(points):
    if len(points) == 0:
        return 0.0
    sorted_idx = np.argsort(points[:, 0])
    sorted_pts = points[sorted_idx]
    hv = 0.0
    for i in range(len(sorted_pts)):
        width = sorted_pts[i, 0] if i == 0 else sorted_pts[i, 0] - sorted_pts[i - 1, 0]
        remaining = sorted_pts[i:, 1:]
        hv_2d = _hv_2d(remaining)
        hv += width * hv_2d
    return hv

def _hv_2d(points):
    if len(points) == 0:
        return 0.0
    sorted_idx = np.argsort(points[:, 0])
    sorted_pts = points[sorted_idx]
    hv = 0.0
    for i in range(len(sorted_pts)):
        width = sorted_pts[i, 0] if i == 0 else sorted_pts[i, 0] - sorted_pts[i - 1, 0]
        height = sorted_pts[i, 1]
        hv += width * height
    return hv

def filter_nondominated(obj_array):
    if len(obj_array) == 0:
        return np.zeros((0, 3))
    arr = np.asarray(obj_array, dtype=float)
    n = len(arr)
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if i == j or not is_nd[j]:
                continue
            if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                is_nd[i] = False
                break
    return arr[is_nd]

def evaluate_checkpoint_on_instance(
    encoder, actor_critic,
    instance_path: str,
    config: dict,
    eval_prefs=None,
    realization_seed: int = 12345,
    max_steps: int = 400,
    dynamic_config: dict = None,
):
    instance = load_instance_json(instance_path)
    data_array = instance['base_processing_times']
    instance_id = get_instance_id(instance_path)

    from env import WorkloadBasedMachineOrdering
    optimizer = WorkloadBasedMachineOrdering(data_array, num_restarts=2)
    sequence, _ = optimizer.optimize()

    realization = build_realization_from_json(instance)

    delay_reward_scale = config.get('delay_reward_scale', 1.0)
    dyn_cfg = dynamic_config or {}

    warmup_env = FastNPFSEnvironment(
        data_array, sequence,
        delay_reward_scale=delay_reward_scale,
        deterministic_eval=True,
        instance_realization=realization,
        dynamic_config=dyn_cfg,
    )
    warmup_state = warmup_env.reset()
    import random as rnd
    done = False
    steps = 0
    info = {}
    while not done and steps < max_steps:
        va = warmup_env.get_valid_actions()
        if va:
            action = {
                'job': va['jobs'][rnd.randint(0, len(va['jobs']) - 1)],
                'factory': rnd.choice(va['factories']),
                'speed': rnd.choice(va['speeds']),
            }
        else:
            action = {'job': 0, 'factory': 0, 'speed': 2}
        _, _, done, info = warmup_env.step(action)
        steps += 1

    ranges = {
        'makespan': max(info.get('makespan', 100.0), 1.0),
        'energy': max(info.get('energy', 100.0), 1.0),
        'delay': max(info.get('delay_cost', 50.0), 1.0),
    }

    all_objectives, eval_hv, avg_ms, avg_en, avg_dl = deterministic_multi_pref_eval(
        encoder, actor_critic, data_array, sequence, ranges,
        realization, max_steps=max_steps,
        delay_reward_scale=delay_reward_scale,
        eval_prefs=eval_prefs, dynamic_config=dyn_cfg,
    )

    nd_objs = filter_nondominated(np.array(all_objectives)) if all_objectives else np.zeros((0, 3))

    return {
        'objectives_list': all_objectives,
        'nd_objectives': nd_objs,
        'instance_id': instance_id,
        'eval_hv': eval_hv,
        'avg_ms': avg_ms,
        'avg_en': avg_en,
        'avg_dl': avg_dl,
        'sequence': sequence,
        'ranges': ranges,
    }

def evaluate_checkpoint_on_testset(
    checkpoint_path: str,
    test_instance_paths: list,
    config: dict,
    seed: int,
    run_id: str,
    output_dir: str,
    eval_prefs: list = None,
    realization_seed: int = 12345,
    device: str = "cpu",
) -> dict:
    from checkpoint_utils import restore_models_from_checkpoint

    print(f"\n{'='*70}")
    print(f"  Test Evaluation: {run_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Test instances: {len(test_instance_paths)}")
    print(f"{'='*70}")

    encoder, actor_critic, ckpt = restore_models_from_checkpoint(
        checkpoint_path, device=device,
    )
    dynamic_config = ckpt.get('dynamic_config', {})

    results = {}
    exported_paths = []

    for inst_path in test_instance_paths:
        instance_id = get_instance_id(inst_path)
        print(f"\n  Testing on: {instance_id}")

        inst_result = evaluate_checkpoint_on_instance(
            encoder, actor_critic,
            inst_path, config,
            eval_prefs=eval_prefs,
            realization_seed=realization_seed,
            dynamic_config=dynamic_config,
        )

        nd_objs = inst_result['nd_objectives']
        if len(nd_objs) > 0:
            front_path = export_front_csv(
                nd_objs, instance_id, run_id,
                output_dir=os.path.join(output_dir, "GAT_PPO"),
            )
            exported_paths.append(front_path)

        results[instance_id] = inst_result
        print(f"    ND solutions: {len(nd_objs)}, "
              f"avg MS={inst_result['avg_ms']:.2f}, "
              f"EN={inst_result['avg_en']:.2f}, "
              f"DL={inst_result['avg_dl']:.2f}")

    print(f"\n  Test evaluation complete. Exported {len(exported_paths)} fronts.")
    return {
        'per_instance': results,
        'exported_paths': exported_paths,
        'run_id': run_id,
        'seed': seed,
        'checkpoint_path': checkpoint_path,
    }

def export_front_csv(
    nd_objectives, instance_id: str, run_id: str,
    output_dir: str = "results/fronts/GAT_PPO", algorithm: str = "GAT_PPO",
) -> str:
    inst_dir = os.path.join(output_dir, instance_id)
    os.makedirs(inst_dir, exist_ok=True)

    filepath = os.path.join(inst_dir, f"{run_id}_final_front.csv")

    fieldnames = ['instance_id', 'run_id', 'algorithm', 'makespan', 'energy', 'delay']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obj in nd_objectives:
            writer.writerow({
                'instance_id': instance_id,
                'run_id': run_id,
                'algorithm': algorithm,
                'makespan': f"{obj[0]:.6f}",
                'energy': f"{obj[1]:.6f}",
                'delay': f"{obj[2]:.6f}",
            })

    print(f"    Exported {len(nd_objectives)} ND solutions → {filepath}")
    return filepath
