import argparse
import json
import os
import random
import numpy as np
from typing import List, Tuple, Dict, Any

DEFAULT_NUM_FACTORIES = 2
DEFAULT_SPEED_MULTIPLIERS = [1.0, 0.7, 0.5, 0.3, 0.15]
DEFAULT_POWER_PER_SPEED = [1, 3, 4, 6, 8]
DEFAULT_HETEROGENEITY_RANGE = (0.8, 1.2)
DEFAULT_HETEROGENEITY_MODE = "profile"
DEFAULT_DUE_DATE_TIGHTNESS = 1.5
DEFAULT_PRIORITY_WEIGHT_RANGE = (1.0, 5.0)

DEFAULT_IDLE_POWER_RANGE = (0.2, 1.2)
DEFAULT_STARTUP_ENERGY_RANGE = (0.5, 1.5)

DEFAULT_DYNAMIC_JOB_RATIO = 0.3
DEFAULT_INITIAL_JOB_RATIO = 0.7
DEFAULT_ARRIVAL_TIME_MODE = "uniform"
DEFAULT_ARRIVAL_TIME_RANGE_LOW = 0.1
DEFAULT_ARRIVAL_TIME_RANGE_HIGH = 0.8
DEFAULT_DYNAMIC_PRIORITY_JOB_RATIO = 0.7
DEFAULT_MAX_ARRIVALS_PER_EVENT = 3

PROFILE_TEMPLATE = [1.00, 0.90, 1.10, 0.85, 1.20, 0.95, 1.15, 0.80, 1.25]

def load_base_processing_times(csv_path: str, skip_header: bool = True,
                                skip_first_col: bool = True) -> np.ndarray:
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if skip_header and i == 0:
                continue
            parts = line.split(',')
            if skip_first_col:
                parts = parts[1:]
            row = [float(x.strip()) for x in parts if x.strip()]
            data.append(row)
    return np.array(data)

def _get_profile_multiplier(factory_index: int) -> float:
    if factory_index < len(PROFILE_TEMPLATE):
        return PROFILE_TEMPLATE[factory_index]
    return 0.8 + 0.4 * ((factory_index % 5) / 4.0)

def generate_factory_times_and_multipliers(
    base_processing_times: np.ndarray,
    num_factories: int,
    heterogeneity_mode: str,
    heterogeneity_range: Tuple[float, float],
    rng: np.random.RandomState,
) -> Tuple[List[np.ndarray], np.ndarray]:
    base = base_processing_times
    num_machines = base.shape[1]
    factory_times: List[np.ndarray] = [base.copy()]

    for f in range(1, num_factories):
        if heterogeneity_mode == "random_weak":
            mult_matrix = rng.uniform(0.9, 1.1, base.shape)
            factory_times.append(base * mult_matrix)
        elif heterogeneity_mode == "random_strong":
            mult_matrix = rng.uniform(0.7, 1.3, base.shape)
            factory_times.append(base * mult_matrix)
        elif heterogeneity_mode == "profile":
            mult_scalar = _get_profile_multiplier(f)
            noise = rng.uniform(0.98, 1.02, base.shape)
            factory_times.append(base * mult_scalar * noise)
        else:
            lo, hi = heterogeneity_range
            mult_matrix = rng.uniform(lo, hi, base.shape)
            factory_times.append(base * mult_matrix)

    factory_time_multipliers = np.ones((num_factories, num_machines), dtype=float)
    for f in range(num_factories):
        ratio = factory_times[f] / np.maximum(base, 1e-9)
        factory_time_multipliers[f, :] = ratio.mean(axis=0)

    return factory_times, factory_time_multipliers

def estimate_horizon(
    factory_times: List[np.ndarray],
    num_factories: int,
    num_jobs: int,
    num_machines: int,
    speed_factors_array: np.ndarray,
) -> float:
    speed_mid = speed_factors_array[2] if len(speed_factors_array) > 2 else speed_factors_array[0]
    total = 0.0
    for j in range(num_jobs):
        job_time = 0.0
        for m in range(num_machines):
            base_min = min(factory_times[f][j, m] for f in range(num_factories))
            job_time += base_min * speed_mid
        total += job_time
    horizon = total / max(num_factories, 1)
    return float(max(horizon, 1.0))

def generate_arrival_times(
    num_dynamic: int,
    horizon: float,
    arrival_time_mode: str,
    arrival_time_range_low: float,
    arrival_time_range_high: float,
    max_arrivals_per_event: int,
    rng: np.random.RandomState,
    py_rng: random.Random,
) -> List[float]:
    low = arrival_time_range_low * horizon
    high = arrival_time_range_high * horizon

    if arrival_time_mode == "uniform":
        times = rng.uniform(low, high, num_dynamic)
    elif arrival_time_mode == "clustered":
        num_clusters = max(1, num_dynamic // max(max_arrivals_per_event, 1))
        cluster_centers = rng.uniform(low, high, num_clusters)
        times = []
        for _ in range(num_dynamic):
            center = py_rng.choice(cluster_centers.tolist())
            t = center + rng.normal(0, horizon * 0.02)
            times.append(float(np.clip(t, low, high)))
        times = np.array(times)
    elif arrival_time_mode == "front_loaded":
        times = rng.beta(2, 5, num_dynamic) * (high - low) + low
    elif arrival_time_mode == "tail_loaded":
        times = rng.beta(5, 2, num_dynamic) * (high - low) + low
    else:
        times = rng.uniform(low, high, num_dynamic)

    return sorted([float(t) for t in times])

def build_job_partitions(
    num_jobs: int,
    dynamic_job_ratio: float,
    py_rng: random.Random,
) -> Tuple[List[int], List[int]]:
    num_dynamic = max(1, int(num_jobs * dynamic_job_ratio))
    num_initial = num_jobs - num_dynamic
    all_ids = list(range(num_jobs))
    py_rng.shuffle(all_ids)
    initial_job_ids = sorted(all_ids[:num_initial])
    dynamic_job_ids = sorted(all_ids[num_initial:])
    return initial_job_ids, dynamic_job_ids

def build_masks_from_ids(
    num_jobs: int,
    initial_job_ids: List[int],
    dynamic_job_ids: List[int],
    high_priority_ids: List[int],
) -> Tuple[List[bool], List[bool], List[bool]]:
    initial_set = set(initial_job_ids)
    dynamic_set = set(dynamic_job_ids)
    hp_set = set(high_priority_ids)
    initial_mask = [bool(i in initial_set) for i in range(num_jobs)]
    dynamic_mask = [bool(i in dynamic_set) for i in range(num_jobs)]
    priority_mask = [bool(i in hp_set) for i in range(num_jobs)]
    return initial_mask, dynamic_mask, priority_mask

def build_arrival_batches(
    arrival_times: Dict[int, float],
    eps: float = 1e-6,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    dyn_items = [(jid, t) for jid, t in arrival_times.items() if t > eps]
    dyn_items.sort(key=lambda x: x[1])

    batches: List[Dict[str, Any]] = []
    for jid, t in dyn_items:
        if batches and abs(batches[-1]["time"] - t) < eps:
            batches[-1]["job_ids"].append(int(jid))
        else:
            batches.append({"time": float(t), "job_ids": [int(jid)]})

    event_times = [b["time"] for b in batches]
    return batches, event_times

def generate_single_instance(
    base_processing_times: np.ndarray,
    seed: int,
    num_factories: int = DEFAULT_NUM_FACTORIES,
    speed_multipliers: list = None,
    power_per_speed: list = None,
    heterogeneity_range: tuple = DEFAULT_HETEROGENEITY_RANGE,
    heterogeneity_mode: str = DEFAULT_HETEROGENEITY_MODE,
    due_date_tightness: float = DEFAULT_DUE_DATE_TIGHTNESS,
    priority_weight_range: tuple = DEFAULT_PRIORITY_WEIGHT_RANGE,
    num_priority_jobs: int = 5,
    idle_power_range: tuple = DEFAULT_IDLE_POWER_RANGE,
    startup_energy_range: tuple = DEFAULT_STARTUP_ENERGY_RANGE,
    dynamic_job_ratio: float = DEFAULT_DYNAMIC_JOB_RATIO,
    initial_job_ratio: float = DEFAULT_INITIAL_JOB_RATIO,
    arrival_time_mode: str = DEFAULT_ARRIVAL_TIME_MODE,
    arrival_time_range_low: float = DEFAULT_ARRIVAL_TIME_RANGE_LOW,
    arrival_time_range_high: float = DEFAULT_ARRIVAL_TIME_RANGE_HIGH,
    dynamic_priority_job_ratio: float = DEFAULT_DYNAMIC_PRIORITY_JOB_RATIO,
    max_arrivals_per_event: int = DEFAULT_MAX_ARRIVALS_PER_EVENT,
) -> dict:
    if speed_multipliers is None:
        speed_multipliers = list(DEFAULT_SPEED_MULTIPLIERS)
    if power_per_speed is None:
        power_per_speed = list(DEFAULT_POWER_PER_SPEED)

    if abs(initial_job_ratio + dynamic_job_ratio - 1.0) > 1e-3:
        print(f"  [warn] initial_job_ratio({initial_job_ratio}) + "
              f"dynamic_job_ratio({dynamic_job_ratio}) != 1.0, "
              f"以 dynamic_job_ratio 为准。")

    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    num_jobs = base_processing_times.shape[0]
    num_machines = base_processing_times.shape[1]
    speed_factors_array = np.array(speed_multipliers)

    factory_times, factory_time_multipliers = generate_factory_times_and_multipliers(
        base_processing_times=base_processing_times,
        num_factories=num_factories,
        heterogeneity_mode=heterogeneity_mode,
        heterogeneity_range=heterogeneity_range,
        rng=rng,
    )

    horizon = estimate_horizon(
        factory_times, num_factories, num_jobs, num_machines, speed_factors_array
    )

    num_priority_jobs = min(num_priority_jobs, num_jobs)
    priority_indices = sorted(py_rng.sample(range(num_jobs), num_priority_jobs))

    machine_idle_power = rng.uniform(
        idle_power_range[0], idle_power_range[1],
        size=(num_factories, num_machines),
    )
    machine_startup_energy = rng.uniform(
        startup_energy_range[0], startup_energy_range[1],
        size=(num_factories, num_machines),
    )

    initial_job_ids, dynamic_job_ids = build_job_partitions(
        num_jobs, dynamic_job_ratio, py_rng,
    )
    num_dynamic = len(dynamic_job_ids)

    arrival_times: Dict[int, float] = {}
    for jid in initial_job_ids:
        arrival_times[jid] = 0.0
    if num_dynamic > 0:
        dyn_times = generate_arrival_times(
            num_dynamic=num_dynamic,
            horizon=horizon,
            arrival_time_mode=arrival_time_mode,
            arrival_time_range_low=arrival_time_range_low,
            arrival_time_range_high=arrival_time_range_high,
            max_arrivals_per_event=max_arrivals_per_event,
            rng=rng,
            py_rng=py_rng,
        )
        for i, jid in enumerate(dynamic_job_ids):
            arrival_times[jid] = float(dyn_times[i])

    num_dyn_priority = max(1, int(num_dynamic * dynamic_priority_job_ratio)) if num_dynamic > 0 else 0
    shuffled_dyn = list(dynamic_job_ids)
    py_rng.shuffle(shuffled_dyn)
    dynamic_priority_ids = set(shuffled_dyn[:num_dyn_priority])

    high_priority_ids = sorted(set(priority_indices) | dynamic_priority_ids)

    initial_set = set(initial_job_ids)
    speed_mid = speed_factors_array[2] if len(speed_factors_array) > 2 else speed_factors_array[0]

    job_meta: List[Dict[str, Any]] = []
    due_dates: List[float] = []
    priority_weights: List[float] = []

    for i in range(num_jobs):
        is_initial = i in initial_set
        is_high_priority = i in set(high_priority_ids)

        factory_idx = py_rng.randint(0, num_factories - 1)
        total_time = float(np.sum(factory_times[factory_idx][i, :] * speed_mid))

        if is_high_priority:
            buffer_factor = py_rng.uniform(1.1, 1.3)
            priority = py_rng.randint(3, 5)
            delay_cost = py_rng.uniform(5, 10)
        else:
            buffer_factor = py_rng.uniform(1.2, 1.5)
            priority = 0
            delay_cost = py_rng.uniform(1, 5)

        arr_t = float(arrival_times[i])
        due_date = arr_t + total_time * buffer_factor

        meta = {
            'id': int(i),
            'is_high_priority': bool(is_high_priority),
            'priority': int(priority),
            'due_date': float(due_date),
            'estimated_processing_time': float(total_time),
            'buffer_factor': float(buffer_factor),
            'delay_cost': float(delay_cost),
            'arrival_time': float(arr_t),
            'is_dynamic': bool(not is_initial),
            'arr_flag': 0 if is_initial else 1,
        }
        job_meta.append(meta)
        due_dates.append(float(due_date))
        priority_weights.append(float(priority) if is_high_priority else float(delay_cost))

    initial_mask, dynamic_mask, priority_mask = build_masks_from_ids(
        num_jobs, initial_job_ids, dynamic_job_ids, high_priority_ids,
    )

    arrival_batches, event_times = build_arrival_batches(arrival_times)

    lo, hi = heterogeneity_range
    instance_data = {
        "seed": int(seed),
        "num_jobs": int(num_jobs),
        "num_machines": int(num_machines),
        "num_factories": int(num_factories),
        "num_stages": int(num_machines),
        "num_speed_levels": len(speed_multipliers),
        "num_priority_jobs": int(num_priority_jobs),
        "instance_class": f"h{num_factories}_n{num_jobs}_m{num_machines}",

        "speed_multipliers": [float(s) for s in speed_multipliers],
        "power_per_speed": [float(p) for p in power_per_speed],
        "heterogeneity_range": [float(lo), float(hi)],
        "heterogeneity_mode": str(heterogeneity_mode),
        "due_date_tightness": float(due_date_tightness),
        "priority_weight_range": [float(priority_weight_range[0]), float(priority_weight_range[1])],
        "idle_power_range": [float(idle_power_range[0]), float(idle_power_range[1])],
        "startup_energy_range": [float(startup_energy_range[0]), float(startup_energy_range[1])],

        "base_processing_times": base_processing_times.tolist(),
        "factory_times": [ft.tolist() for ft in factory_times],
        "factory_time_multipliers": factory_time_multipliers.tolist(),
        "priority_indices": [int(x) for x in priority_indices],
        "job_meta": job_meta,
        "machine_idle_power": machine_idle_power.tolist(),
        "machine_startup_energy": machine_startup_energy.tolist(),

        "horizon": float(horizon),
        "arrival_times": {str(k): float(v) for k, v in arrival_times.items()},
        "initial_job_ids": [int(x) for x in initial_job_ids],
        "dynamic_job_ids": [int(x) for x in dynamic_job_ids],
        "dynamic_priority_ids": sorted([int(x) for x in dynamic_priority_ids]),
        "high_priority_ids": [int(x) for x in high_priority_ids],
        "initial_job_mask": initial_mask,
        "dynamic_job_mask": dynamic_mask,
        "priority_job_mask": priority_mask,
        "arrival_batches": arrival_batches,
        "event_times": [float(t) for t in event_times],

        "due_dates": due_dates,
        "priority_weights": priority_weights,

        "generation_protocol": {
            "generator": "generate_eval_instances.py",
            "description": (
                "统一评估实例 (含动态插单)。所有算法 (GAT-PPO / CPLEX / 其他) "
                "必须使用此实例文件, 不允许各自随机生成工厂异构时间或动态到达。"
            ),
            "reproducible": True,
            "env_compatible": True,
        },
        "dynamic_generation_protocol": {
            "dynamic_job_ratio": float(dynamic_job_ratio),
            "initial_job_ratio": float(initial_job_ratio),
            "arrival_time_mode": str(arrival_time_mode),
            "arrival_time_range_low": float(arrival_time_range_low),
            "arrival_time_range_high": float(arrival_time_range_high),
            "dynamic_priority_job_ratio": float(dynamic_priority_job_ratio),
            "max_arrivals_per_event": int(max_arrivals_per_event),
        },
        "heterogeneity_protocol": {
            "heterogeneity_mode": str(heterogeneity_mode),
            "heterogeneity_range": [float(lo), float(hi)],
            "profile_template": list(PROFILE_TEMPLATE),
        },
    }

    return instance_data

def generate_and_save_instances(
    csv_path: str,
    seeds: List[int],
    output_dir: str = "instances",
    skip_header: bool = True,
    skip_first_col: bool = True,
    num_factories: int = DEFAULT_NUM_FACTORIES,
    **kwargs,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    base_times = load_base_processing_times(csv_path, skip_header, skip_first_col)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    num_jobs = base_times.shape[0]
    num_machines = base_times.shape[1]
    print(f"\n{'=' * 70}")
    print(f"  实例生成: {base_name}")
    print(f"  Jobs={num_jobs}, Machines={num_machines}, Factories={num_factories}")
    print(f"  Seeds: {seeds}")
    print(f"  Heterogeneity mode: {kwargs.get('heterogeneity_mode', DEFAULT_HETEROGENEITY_MODE)}")
    print(f"  Dynamic ratio: {kwargs.get('dynamic_job_ratio', DEFAULT_DYNAMIC_JOB_RATIO)}, "
          f"Arrival mode: {kwargs.get('arrival_time_mode', DEFAULT_ARRIVAL_TIME_MODE)}")
    print(f"{'=' * 70}")

    saved_paths = []
    for seed in seeds:
        instance = generate_single_instance(
            base_times,
            seed=seed,
            num_factories=num_factories,
            **kwargs,
        )
        filename = f"{base_name}_seed{seed}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)

        n_dyn = len(instance["dynamic_job_ids"])
        n_ini = len(instance["initial_job_ids"])
        n_hp = len(instance["high_priority_ids"])
        n_batch = len(instance["arrival_batches"])
        print(f"  [OK] {filepath}  "
              f"(initial={n_ini}, dynamic={n_dyn}, high_priority={n_hp}, "
              f"arrival_batches={n_batch}, horizon={instance['horizon']:.1f})")
        saved_paths.append(filepath)

    return saved_paths

def load_instance(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        instance = json.load(f)

    instance['base_processing_times'] = np.array(instance['base_processing_times'])
    instance['factory_time_multipliers'] = np.array(instance['factory_time_multipliers'])
    instance['due_dates'] = np.array(instance['due_dates'])
    instance['priority_weights'] = np.array(instance['priority_weights'])

    if 'factory_times' in instance:
        instance['factory_times'] = [np.array(ft) for ft in instance['factory_times']]
    if 'machine_idle_power' in instance:
        instance['machine_idle_power'] = np.array(instance['machine_idle_power'])
    if 'machine_startup_energy' in instance:
        instance['machine_startup_energy'] = np.array(instance['machine_startup_energy'])
    if 'arrival_times' in instance and isinstance(instance['arrival_times'], dict):
        instance['arrival_times'] = {int(k): float(v) for k, v in instance['arrival_times'].items()}

    return instance

def list_instances(instance_dir: str, prefix: str = None) -> List[str]:
    files = []
    for fname in sorted(os.listdir(instance_dir)):
        if not fname.endswith('.json'):
            continue
        if prefix and not fname.startswith(prefix):
            continue
        files.append(os.path.join(instance_dir, fname))
    return files

def get_instance_id(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def main():
    parser = argparse.ArgumentParser(description="统一评估实例生成器 (动态增强版)")

    parser.add_argument("--csv", type=str, required=True, help="基础加工时间 CSV 文件路径")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="随机种子列表")
    parser.add_argument("--output_dir", type=str, default="instances", help="输出目录")
    parser.add_argument("--num_factories", type=int, default=DEFAULT_NUM_FACTORIES)
    parser.add_argument("--num_priority_jobs", type=int, default=5)
    parser.add_argument("--skip_header", action="store_true", default=True)
    parser.add_argument("--skip_first_col", action="store_true", default=True)

    parser.add_argument("--heterogeneity_mode", type=str,
                        choices=["profile", "random_weak", "random_strong"],
                        default=DEFAULT_HETEROGENEITY_MODE)
    parser.add_argument("--heterogeneity_low", type=float, default=DEFAULT_HETEROGENEITY_RANGE[0])
    parser.add_argument("--heterogeneity_high", type=float, default=DEFAULT_HETEROGENEITY_RANGE[1])

    parser.add_argument("--dynamic_job_ratio", type=float, default=DEFAULT_DYNAMIC_JOB_RATIO)
    parser.add_argument("--initial_job_ratio", type=float, default=DEFAULT_INITIAL_JOB_RATIO)
    parser.add_argument("--arrival_time_mode", type=str,
                        choices=["uniform", "clustered", "front_loaded", "tail_loaded"],
                        default=DEFAULT_ARRIVAL_TIME_MODE)
    parser.add_argument("--arrival_time_range_low", type=float, default=DEFAULT_ARRIVAL_TIME_RANGE_LOW)
    parser.add_argument("--arrival_time_range_high", type=float, default=DEFAULT_ARRIVAL_TIME_RANGE_HIGH)
    parser.add_argument("--dynamic_priority_job_ratio", type=float,
                        default=DEFAULT_DYNAMIC_PRIORITY_JOB_RATIO)
    parser.add_argument("--max_arrivals_per_event", type=int, default=DEFAULT_MAX_ARRIVALS_PER_EVENT)

    parser.add_argument("--idle_power_low", type=float, default=DEFAULT_IDLE_POWER_RANGE[0])
    parser.add_argument("--idle_power_high", type=float, default=DEFAULT_IDLE_POWER_RANGE[1])
    parser.add_argument("--startup_energy_low", type=float, default=DEFAULT_STARTUP_ENERGY_RANGE[0])
    parser.add_argument("--startup_energy_high", type=float, default=DEFAULT_STARTUP_ENERGY_RANGE[1])

    parser.add_argument("--due_date_tightness", type=float, default=DEFAULT_DUE_DATE_TIGHTNESS)

    args = parser.parse_args()

    paths = generate_and_save_instances(
        csv_path=args.csv,
        seeds=args.seeds,
        output_dir=args.output_dir,
        skip_header=args.skip_header,
        skip_first_col=args.skip_first_col,
        num_factories=args.num_factories,
        num_priority_jobs=args.num_priority_jobs,
        heterogeneity_mode=args.heterogeneity_mode,
        heterogeneity_range=(args.heterogeneity_low, args.heterogeneity_high),
        due_date_tightness=args.due_date_tightness,
        idle_power_range=(args.idle_power_low, args.idle_power_high),
        startup_energy_range=(args.startup_energy_low, args.startup_energy_high),
        dynamic_job_ratio=args.dynamic_job_ratio,
        initial_job_ratio=args.initial_job_ratio,
        arrival_time_mode=args.arrival_time_mode,
        arrival_time_range_low=args.arrival_time_range_low,
        arrival_time_range_high=args.arrival_time_range_high,
        dynamic_priority_job_ratio=args.dynamic_priority_job_ratio,
        max_arrivals_per_event=args.max_arrivals_per_event,
    )

    print(f"\n  共生成 {len(paths)} 个实例文件 -> {args.output_dir}")

if __name__ == "__main__":
    main()
