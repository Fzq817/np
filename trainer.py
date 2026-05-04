import os
import time
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from config import TRAIN_CONFIG, DYNAMIC_ENV_CONFIG, build_runtime_config, set_global_seed
from utils import check_tensor_valid, load_instance_json, TimeTracker
from models import (
    HeteroPreferenceGATEncoder, SimpleActorCritic,
    masked_categorical, build_models,
)
from evaluation import (
    create_fixed_eval_realization, build_realization_from_json,
    rollout_policy_once, deterministic_multi_pref_eval, filter_nondominated,
    export_front_csv,
)
from checkpoint_utils import save_checkpoint, get_checkpoint_paths
from pareto_pref import (
    FastHVCalculator, FastPreferenceSampler,
    LightweightHistoricalFront, FastParetoFront,
)
from env import FastNPFSEnvironment

class WorkloadBasedMachineOrdering:
    def __init__(self, data_array, num_factories=2, num_restarts=2):
        self.num_factories = num_factories
        self.num_machines = data_array.shape[1]
        self.num_jobs = data_array.shape[0]
        self.factory_times = data_array
        self.num_restarts = num_restarts

    def optimize(self):
        best_sequence = None
        best_makespan = float('inf')
        for restart in range(self.num_restarts):
            scores = [(m, np.mean(self.factory_times[:, m]))
                      for m in range(self.num_machines)]
            scores.sort(key=lambda x: x[1], reverse=True)
            sequence = [m for m, _ in scores]
            makespan = self._evaluate(sequence)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = sequence
        return best_sequence, best_makespan

    def _evaluate(self, sequence):
        machine_times = np.zeros(self.num_machines)
        job_times = np.zeros((self.num_jobs, self.num_machines))
        for job_id in range(self.num_jobs):
            for stage_idx, machine_id in enumerate(sequence):
                base_time = self.factory_times[job_id, machine_id]
                if stage_idx == 0:
                    start = machine_times[machine_id]
                else:
                    start = max(machine_times[machine_id], job_times[job_id, stage_idx - 1])
                completion = start + base_time
                machine_times[machine_id] = completion
                job_times[job_id, stage_idx] = completion
        return machine_times.max()

class MetricsCalculator:

    @staticmethod
    def hypervolume(pf, ref=None, normalize=True):
        if pf.size() == 0:
            return 0.0
        objs = pf.get_objectives_array()
        if normalize:
            normed = pf.get_normalized_objectives_array()
            ref_pt = normed.max(axis=0) * 1.1
            ref_pt = np.maximum(ref_pt, np.ones(3) * 1.1)
        else:
            normed = objs
            ref_pt = objs.max(axis=0) * 1.1
        dominated = np.maximum(0, ref_pt - normed)
        if dominated.shape[1] == 3:
            hv = MetricsCalculator._hv_3d(dominated)
            max_hv = np.prod(ref_pt)
            return hv / max_hv if max_hv > 0 else 0.0
        return 0.0

    @staticmethod
    def hypervolume_from_points(points_array, ref_point=None):
        if len(points_array) == 0:
            return 0.0
        points = np.array(points_array)
        if ref_point is None:
            ref_point = points.max(axis=0) * 1.1
        ref_point = np.maximum(ref_point, points.max(axis=0) * 1.01)
        dominated = np.maximum(0, ref_point - points)
        if dominated.shape[1] == 3:
            hv = MetricsCalculator._hv_3d(dominated)
            max_hv = np.prod(ref_point)
            return hv / max_hv if max_hv > 0 else 0.0
        return 0.0

    @staticmethod
    def _hv_3d(points):
        if len(points) == 0:
            return 0.0
        sorted_idx = np.argsort(points[:, 0])
        sorted_pts = points[sorted_idx]
        hv = 0.0
        for i in range(len(sorted_pts)):
            width = sorted_pts[i, 0] if i == 0 else sorted_pts[i, 0] - sorted_pts[i - 1, 0]
            remaining = sorted_pts[i:, 1:]
            hv_2d = MetricsCalculator._hv_2d(remaining)
            hv += width * hv_2d
        return hv

    @staticmethod
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

    @staticmethod
    def spacing(pf, normalize=True):
        if pf.size() <= 1:
            return 0.0
        objs = pf.get_normalized_objectives_array() if normalize else pf.get_objectives_array()
        dists = cdist(objs, objs)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        mean = min_dists.mean()
        sp = np.sqrt(((min_dists - mean) ** 2).mean())
        if normalize:
            return sp / np.sqrt(3)
        return sp

    @staticmethod
    def calculate_all_metrics(pf, ref_pf=None, normalize=True):
        metrics = {}
        metrics['HV'] = MetricsCalculator.hypervolume(pf, normalize=normalize)
        metrics['Spacing'] = MetricsCalculator.spacing(pf, normalize=normalize)
        if ref_pf and ref_pf.size() > 0 and pf.size() > 0:
            from scipy.spatial.distance import cdist as cdist_f
            pf_objs = pf.get_objectives_array()
            ref_objs = ref_pf.get_objectives_array()
            if normalize:
                all_objs = np.vstack([pf_objs, ref_objs])
                mins = all_objs.min(axis=0)
                maxs = all_objs.max(axis=0)
                rng = maxs - mins
                rng[rng == 0] = 1.0
                pf_norm = (pf_objs - mins) / rng
                ref_norm = (ref_objs - mins) / rng
            else:
                pf_norm = pf_objs
                ref_norm = ref_objs
            dists = cdist_f(ref_norm, pf_norm)
            min_dists = dists.min(axis=1)
            metrics['IGD'] = float(min_dists.mean())
        else:
            metrics['IGD'] = None
        return metrics

class FastPPOTrainer:
    def __init__(self, encoder, actor_critic, lr=3e-4, gamma=0.98,
                 actor_lr=6e-4, critic_lr=2e-3, clip_epsilon=0.1,
                 lr_decay_gamma=0.997, c1=0.5, c2=0.01):
        self.encoder = encoder
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(actor_critic.parameters()),
            lr=actor_lr,
        )
        self.memory = []
        self.actor_loss_ema = None
        self.value_loss_ema = None
        self.entropy_ema = None
        self.ema_alpha = 0.70

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay_gamma,
        )

    def store(self, state, action, log_prob, values, rewards, done, pref, mask_info=None):
        if not (np.isnan(rewards).any() or np.isinf(rewards).any()):
            rewards = np.clip(rewards, -10, 10)
            action_tuple = (action['job_idx'], action['factory'], action['speed'])
            self.memory.append((state, action_tuple, log_prob, values, rewards, done, pref, mask_info))

    def clear_memory(self):
        self.memory = []

    def update(self, batch_size=128, epochs=4, entropy_coef=0.01):
        if len(self.memory) < batch_size:
            return None

        all_states = [item[0] for item in self.memory]
        all_actions = torch.LongTensor(np.array([list(item[1]) for item in self.memory]))
        all_old_log_probs = torch.FloatTensor(np.array([item[2] for item in self.memory]))
        all_values = torch.FloatTensor(np.array([item[3] for item in self.memory])).squeeze()
        all_prefs = torch.FloatTensor(np.array([item[6] for item in self.memory]))
        all_mask_info = [item[7] for item in self.memory]

        all_rewards_raw = np.array([item[4] for item in self.memory])
        all_rewards = np.empty_like(all_rewards_raw)
        mean_3 = all_rewards_raw[:, :3].mean(axis=0)
        std_3 = all_rewards_raw[:, :3].std(axis=0) + 1e-8
        all_rewards[:, :3] = (all_rewards_raw[:, :3] - mean_3) / std_3 / 5.0
        all_rewards[:, 3] = all_rewards_raw[:, 3]

        returns = []
        discounted_sum = 0
        prefs_np = all_prefs.numpy()
        for i in reversed(range(len(all_rewards))):
            r = all_rewards[i]
            p = prefs_np[i]
            scalar_r = float(np.sum(r[:3] * p) + r[3])
            discounted_sum = scalar_r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(np.array(returns))
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        if all_values.dim() == 2:
            scalar_values = all_values.squeeze(-1)
        else:
            scalar_values = all_values
        advantages = returns - scalar_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(self.memory)
        indices = np.arange(dataset_size)

        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                batch_features = [all_states[i]['node_features'] for i in idx]
                batch_adj = [all_states[i]['adj_matrix'] for i in idx]
                batch_types = [all_states[i]['node_types'] for i in idx]

                mb_prefs = all_prefs[idx]
                mb_actions = all_actions[idx]
                mb_old_log_probs = all_old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                max_nodes = max(f.size(1) for f in batch_features)
                padded_f, padded_a, padded_t = [], [], []
                for f, a, t in zip(batch_features, batch_adj, batch_types):
                    pad = max_nodes - f.size(1)
                    if pad > 0:
                        f = F.pad(f, (0, 0, 0, pad))
                        a = F.pad(a, (0, pad, 0, pad))
                        t = F.pad(t, (0, pad))
                    padded_f.append(f)
                    padded_a.append(a)
                    padded_t.append(t)

                model_device = next(self.encoder.parameters()).device
                batch_f = torch.cat(padded_f, dim=0).to(model_device)
                batch_a = torch.cat(padded_a, dim=0).to(model_device)
                batch_t = torch.cat(padded_t, dim=0).to(model_device)
                mb_prefs = mb_prefs.to(model_device)
                mb_actions = mb_actions.to(model_device)
                mb_old_log_probs = mb_old_log_probs.to(model_device)
                mb_advantages = mb_advantages.to(model_device)
                mb_returns = mb_returns.to(model_device)

                _, embeds = self.encoder(batch_f, batch_a, batch_t, mb_prefs)
                shared_state, values_new = self.actor_critic.forward_base(embeds, mb_prefs)

                batch_size_actual = shared_state.size(0)
                new_log_probs_list = []
                entropy_list = []

                for b in range(batch_size_actual):
                    h_b = shared_state[b:b+1]
                    sample_idx = idx[b]
                    mi = all_mask_info[sample_idx]

                    job_idx_b = mb_actions[b, 0]
                    factory_b = mb_actions[b, 1]
                    speed_b = mb_actions[b, 2]

                    j_logit_b = self.actor_critic.job_head(h_b)
                    if mi is not None and 'num_candidates' in mi:
                        valid_job_indices = list(range(mi['num_candidates']))
                        j_dist_b = masked_categorical(j_logit_b, valid_job_indices)
                    else:
                        j_dist_b = torch.distributions.Categorical(logits=j_logit_b)

                    job_emb_b = self.actor_critic.job_embed(job_idx_b.unsqueeze(0))
                    f_logit_b = self.actor_critic.factory_head(
                        torch.cat([h_b, job_emb_b], dim=-1))
                    if mi is not None and 'valid_factories' in mi:
                        f_dist_b = masked_categorical(f_logit_b, mi['valid_factories'])
                    else:
                        f_dist_b = torch.distributions.Categorical(logits=f_logit_b)

                    fac_emb_b = self.actor_critic.factory_embed(factory_b.unsqueeze(0))
                    s_logit_b = self.actor_critic.speed_head(
                        torch.cat([h_b, job_emb_b, fac_emb_b], dim=-1))
                    s_dist_b = torch.distributions.Categorical(logits=s_logit_b)

                    lp = (j_dist_b.log_prob(job_idx_b)
                          + f_dist_b.log_prob(factory_b)
                          + s_dist_b.log_prob(speed_b))
                    new_log_probs_list.append(lp)

                    ent = j_dist_b.entropy() + f_dist_b.entropy() + s_dist_b.entropy()
                    entropy_list.append(ent)

                new_log_probs = torch.cat(new_log_probs_list)
                entropy = torch.cat(entropy_list).mean()

                check_tensor_valid('new_log_probs (update)', new_log_probs)
                check_tensor_valid('entropy (update)', entropy)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                check_tensor_valid('ratio', ratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values_scalar = values_new.squeeze(-1)
                value_loss = 0.5 * ((values_scalar - mb_returns) ** 2).mean()

                total_loss = policy_loss + self.c1 * value_loss - entropy_coef * entropy
                check_tensor_valid('total_loss', total_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor_critic.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                total_actor_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        avg_actor = total_actor_loss / update_count
        avg_value = total_value_loss / update_count
        avg_entropy = total_entropy / update_count

        if self.actor_loss_ema is None:
            self.actor_loss_ema = avg_actor
            self.value_loss_ema = avg_value
            self.entropy_ema = avg_entropy
        else:
            self.actor_loss_ema = self.ema_alpha * self.actor_loss_ema + (1 - self.ema_alpha) * avg_actor
            self.value_loss_ema = self.ema_alpha * self.value_loss_ema + (1 - self.ema_alpha) * avg_value
            self.entropy_ema = self.ema_alpha * self.entropy_ema + (1 - self.ema_alpha) * avg_entropy

        if len(self.memory) > batch_size:
            self.memory = self.memory[-batch_size:]

        self.scheduler.step()

        return {
            'actor_loss': self.actor_loss_ema,
            'value_loss': self.value_loss_ema,
            'entropy': self.entropy_ema,
            'entropy_coef': entropy_coef,
            'total_loss': self.actor_loss_ema + self.c1 * self.value_loss_ema - entropy_coef * self.entropy_ema,
        }

def evaluate_on_validation_set(
    encoder,
    actor_critic,
    val_instance_paths: list,
    sequence: list,
    ranges: dict,
    config: dict,
    dynamic_config: dict = None,
    max_steps: int = 400,
    delay_reward_scale: float = 1.0,
    device: str = "cpu",
) -> dict:
    if not val_instance_paths:
        return {"avg_hv": 0.0, "avg_ms": 0.0, "avg_en": 0.0, "avg_dl": 0.0,
                "num_val_instances": 0, "num_evaluated": 0}

    hvs, mss, ens, dls = [], [], [], []

    for vp in val_instance_paths:
        try:
            v_inst = load_instance_json(vp)
        except Exception as e:
            print(f"    [val] skip {vp}: {e}")
            continue
        v_data = v_inst['base_processing_times']
        v_real = build_realization_from_json(v_inst)

        try:
            v_opt = WorkloadBasedMachineOrdering(v_data, num_restarts=1)
            v_seq, _ = v_opt.optimize()
        except Exception:
            v_seq = sequence

        try:
            _, eval_hv, avg_ms, avg_en, avg_dl = deterministic_multi_pref_eval(
                encoder, actor_critic, v_data, v_seq, ranges,
                v_real, max_steps=max_steps,
                delay_reward_scale=delay_reward_scale,
                eval_prefs=None,
                dynamic_config=dynamic_config or {},
            )
        except Exception as e:
            import traceback
            print(f"    [val] eval failed on {os.path.basename(vp)}: {e}")
            traceback.print_exc()
            continue

        hvs.append(eval_hv)
        mss.append(avg_ms)
        ens.append(avg_en)
        dls.append(avg_dl)

    if not hvs:
        return {"avg_hv": 0.0, "avg_ms": 0.0, "avg_en": 0.0, "avg_dl": 0.0,
                "num_val_instances": len(val_instance_paths), "num_evaluated": 0}

    return {
        "avg_hv": float(np.mean(hvs)),
        "avg_ms": float(np.mean(mss)),
        "avg_en": float(np.mean(ens)),
        "avg_dl": float(np.mean(dls)),
        "num_val_instances": len(val_instance_paths),
        "num_evaluated": len(hvs),
    }

def train_one_seed(
    train_instance_paths: list,
    val_instance_paths: list,
    config: dict,
    seed: int,
    output_dir: str,
    device: str = "cpu",
    class_name: str = None,
    instance_sampling: str = "round_robin",
) -> dict:
    cfg = dict(TRAIN_CONFIG)
    if config:
        cfg.update(config)
    cfg['seed'] = seed

    dynamic_config = {k: cfg[k] for k in DYNAMIC_ENV_CONFIG if k in cfg}
    cfg = build_runtime_config(cfg)

    num_episodes = cfg['num_episodes']
    max_steps = cfg['max_steps_per_episode']
    eval_every = cfg['eval_every']
    save_best = cfg['save_best']
    delay_reward_scale = cfg.get('delay_reward_scale', 1.0)
    late_stage_start_episode = cfg['late_stage_start_episode']

    set_global_seed(seed)

    print(f"\n{'='*80}")
    print(f"  Training seed={seed}, episodes={num_episodes}")
    print(f"  class_name:      {class_name if class_name else '(not set)'}")
    print(f"  Train instances: {len(train_instance_paths)}")
    print(f"  Val instances:   {len(val_instance_paths)}")
    print(f"  Instance sampling: {instance_sampling}")
    print(f"{'='*80}")

    ckpt_paths = get_checkpoint_paths(output_dir, seed)
    os.makedirs(os.path.dirname(ckpt_paths['best']), exist_ok=True)

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_seed{seed}.log")
    summary_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"train_seed{seed}_summary.txt")

    train_instance = load_instance_json(train_instance_paths[0])
    data_array = train_instance['base_processing_times']

    optimizer = WorkloadBasedMachineOrdering(data_array, num_restarts=2)
    sequence, _ = optimizer.optimize()
    print(f"  Machine sequence: {sequence}")

    realization = build_realization_from_json(train_instance)
    warmup_n = cfg['warmup_episodes']
    makespans, energies, delays = [], [], []

    warmup_cache = []
    for tp in train_instance_paths:
        try:
            ti = load_instance_json(tp)
            warmup_cache.append({
                'path': tp,
                'data': ti['base_processing_times'],
                'realization': build_realization_from_json(ti),
            })
        except Exception as e:
            print(f"  [WARN] failed to load training instance {tp}: {e}")
    if not warmup_cache:
        raise RuntimeError("No training instances could be loaded for warmup.")

    for w in range(warmup_n):
        cached = warmup_cache[w % len(warmup_cache)]
        env = FastNPFSEnvironment(
            cached['data'], sequence, record_details=False,
            delay_reward_scale=delay_reward_scale,
            deterministic_eval=True,
            instance_realization=cached['realization'],
            dynamic_config=dynamic_config,
        )
        state = env.reset()
        done = False
        steps = 0
        info = {}
        while not done and steps < max_steps:
            va = env.get_valid_actions()
            if va:
                action = {
                    'job': va['jobs'][random.randint(0, len(va['jobs']) - 1)],
                    'factory': random.choice(va['factories']),
                    'speed': random.choice(va['speeds']),
                }
            else:
                action = {'job': 0, 'factory': 0, 'speed': 2}
            _, _, done, info = env.step(action)
            steps += 1
        makespans.append(info.get('makespan', 0))
        energies.append(info.get('energy', 0))
        delays.append(info.get('delay_cost', 0))

    ranges = {
        'makespan': max(np.max(makespans), 1.0),
        'energy': max(np.max(energies), 1.0),
        'delay': max(np.max(delays), 1.0),
    }
    running_ranges = dict(ranges)
    print(f"  Ranges (over {len(warmup_cache)} train instances, "
          f"{warmup_n} rollouts): "
          f"MS={ranges['makespan']:.1f}, EN={ranges['energy']:.1f}, "
          f"DL={ranges['delay']:.1f}")

    probe_env = FastNPFSEnvironment(
        data_array, sequence, objective_ranges=ranges,
        record_details=False, delay_reward_scale=delay_reward_scale,
        deterministic_eval=True, instance_realization=realization,
        dynamic_config=dynamic_config,
    )
    probe_state = probe_env.reset()
    feat_dim = probe_state['node_features'].shape[-1]
    num_factories = probe_env.num_factories
    del probe_env, probe_state

    print(f"  feat_dim={feat_dim}, num_factories={num_factories}")

    dev = torch.device(device)
    encoder, actor_critic = build_models(cfg, feat_dim, num_factories, dev)

    pareto_front = FastParetoFront(max_size=1000)
    historical_front = LightweightHistoricalFront(max_size=5000)
    sampler = FastPreferenceSampler()

    trainer = FastPPOTrainer(
        encoder, actor_critic, lr=3e-4,
        actor_lr=cfg.get('actor_lr', 6e-4),
        critic_lr=cfg.get('critic_lr', 2e-3),
        clip_epsilon=cfg.get('clip_epsilon', 0.1),
        lr_decay_gamma=cfg.get('lr_decay_gamma', 0.997),
        c1=cfg.get('c1', 0.5),
        c2=cfg.get('c2', 0.01),
    )

    env = FastNPFSEnvironment(
        data_array, sequence, objective_ranges=ranges,
        record_details=False, delay_reward_scale=delay_reward_scale,
        deterministic_eval=True, instance_realization=realization,
        dynamic_config=dynamic_config,
    )

    best_hv = 0
    best_episode = -1
    best_val_hv = 0.0
    best_late_episode = -1
    best_late_val_hv = 0.0
    hv_history = []
    best_ma_window = cfg['best_model_ma_window']

    entropy_coef_start = cfg['entropy_coef_start']
    entropy_coef_end = cfg['entropy_coef_end']
    entropy_anneal_episodes = cfg['entropy_anneal_episodes']

    time_tracker = TimeTracker()
    training_start_time = time.time()
    log_lines = []

    train_inst_cache = warmup_cache
    print(f"  [Cache] using {len(train_inst_cache)} training instances "
          f"(shared with warmup cache)")

    current_inst_idx = -1

    for episode in range(num_episodes):
        if instance_sampling == "random":
            inst_idx = int(np.random.randint(len(train_inst_cache)))
        else:
            inst_idx = episode % len(train_inst_cache)

        if inst_idx != current_inst_idx:
            cached = train_inst_cache[inst_idx]
            env = FastNPFSEnvironment(
                cached['data'], sequence, objective_ranges=ranges,
                record_details=False, delay_reward_scale=delay_reward_scale,
                deterministic_eval=True, instance_realization=cached['realization'],
                dynamic_config=dynamic_config,
            )
            current_inst_idx = inst_idx

        current_inst_path = train_inst_cache[inst_idx]['path']
        current_inst_id = os.path.splitext(os.path.basename(current_inst_path))[0]

        early_phase_end = int(num_episodes * 0.3)
        update_interval = 2 if episode < early_phase_end else 4

        state = env.reset()
        done = False
        step_count = 0
        pref = sampler.sample(pareto_front, episode, total_episodes=num_episodes)
        pref_tensor = torch.FloatTensor(pref).to(dev)
        episode_return = np.zeros(3)

        while not done and step_count < max_steps:
            va = env.get_valid_actions()
            if va is None:
                break

            inference_start = time.time()
            with torch.no_grad():
                _, embed = encoder(
                    state['node_features'].to(dev), state['adj_matrix'].to(dev),
                    state['node_types'].to(dev), pref_tensor,
                )
                if embed.dim() == 1:
                    embed = embed.unsqueeze(0)
                action, log_prob, _, values = actor_critic.get_action(
                    embed, pref_tensor,
                    valid_actions=[va] if va else None,
                )
            time_tracker.network_inference_time += (time.time() - inference_start)

            env_start = time.time()
            next_state, rewards, done, info = env.step(action)
            time_tracker.env_step_time += (time.time() - env_start)

            episode_return += rewards

            if 'waiting' not in info and 'completed' not in info:
                mask_info = {
                    'num_candidates': len(va['jobs']),
                    'valid_factories': va['per_job_factories'].get(
                        action['job'], va['factories']),
                }
                trainer.store(state, action, log_prob.item(),
                              values.squeeze().detach().cpu().numpy(),
                              rewards, done, pref, mask_info)

            state = next_state
            step_count += 1

        ms = info.get('makespan', 0)
        en = info.get('energy', 0)
        dl = info.get('delay_cost', 0)

        if ms > 0: running_ranges['makespan'] = max(running_ranges['makespan'], ms)
        if en > 0: running_ranges['energy'] = max(running_ranges['energy'], en)
        if dl > 0: running_ranges['delay'] = max(running_ranges['delay'], dl)
        if episode % 20 == 0 and episode > 0:
            env.objective_ranges = dict(running_ranges)

        if ms > 0 and en > 0:
            pareto_front.add_solution([ms, en, dl], schedule_info=None)
            historical_front.add_solution([ms, en, dl])

        anneal_progress = min(episode, entropy_anneal_episodes) / max(entropy_anneal_episodes, 1)
        entropy_anneal_mode = cfg.get('entropy_anneal_mode', 'cosine')
        if entropy_anneal_mode == 'cosine':
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
            current_entropy_coef = entropy_coef_end + (entropy_coef_start - entropy_coef_end) * cosine_factor
        else:
            current_entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * anneal_progress

        loss_info = None
        if (episode + 1) % update_interval == 0:
            update_start = time.time()
            loss_info = trainer.update(
                batch_size=cfg['batch_size'],
                epochs=cfg['ppo_epochs'],
                entropy_coef=current_entropy_coef,
            )
            time_tracker.network_update_time += (time.time() - update_start)

        current_hv = 0.0
        if pareto_front.size() > 0:
            current_hv = MetricsCalculator.hypervolume(pareto_front, normalize=True)
        hv_history.append(current_hv)

        if len(hv_history) >= best_ma_window:
            smoothed_hv = np.mean(hv_history[-best_ma_window:])
        else:
            smoothed_hv = np.mean(hv_history)

        ran_val = False
        avg_val_hv = None
        if save_best and val_instance_paths and (
            (episode + 1) % eval_every == 0 or episode == num_episodes - 1
        ):
            val_metrics = evaluate_on_validation_set(
                encoder, actor_critic, val_instance_paths,
                sequence=sequence, ranges=ranges, config=cfg,
                dynamic_config=dynamic_config, max_steps=max_steps,
                delay_reward_scale=delay_reward_scale, device=device,
            )
            avg_val_hv = val_metrics['avg_hv']
            ran_val = True

            if avg_val_hv > best_val_hv:
                best_val_hv = avg_val_hv
                best_hv = current_hv
                best_episode = episode
                save_checkpoint(
                    path=ckpt_paths['best'],
                    encoder=encoder, actor_critic=actor_critic,
                    optimizer=trainer.optimizer,
                    episode=episode, seed=seed, config=cfg,
                    sequence=sequence, feat_dim=feat_dim,
                    num_factories=num_factories,
                    objective_ranges=ranges,
                    dynamic_config=dynamic_config,
                    checkpoint_type='best',
                    best_val_metric=best_val_hv,
                    best_metric_name='avg_val_hv',
                    class_name=class_name,
                )

            if (episode >= late_stage_start_episode
                    and avg_val_hv > best_late_val_hv):
                best_late_val_hv = avg_val_hv
                best_late_episode = episode
                save_checkpoint(
                    path=ckpt_paths['best_late'],
                    encoder=encoder, actor_critic=actor_critic,
                    optimizer=trainer.optimizer,
                    episode=episode, seed=seed, config=cfg,
                    sequence=sequence, feat_dim=feat_dim,
                    num_factories=num_factories,
                    objective_ranges=ranges,
                    dynamic_config=dynamic_config,
                    checkpoint_type='best_late',
                    best_val_metric=best_late_val_hv,
                    best_metric_name='avg_val_hv',
                    class_name=class_name,
                )

        should_print = (episode < 20) or (episode % 5 == 0) or (best_episode == episode) or ran_val
        if should_print:
            loss_str = ""
            if loss_info:
                loss_str = (f" | A={loss_info['actor_loss']:.4f}"
                            f" V={loss_info['value_loss']:.4f}"
                            f" E={loss_info['entropy']:.4f}")
            val_str = ""
            if ran_val and avg_val_hv is not None:
                val_str = f" | valHV={avg_val_hv:.6f}"
            line = (f"[Ep {episode:>4d}/{num_episodes}] "
                    f"seed={seed} class={class_name or '-'} "
                    f"inst={current_inst_id} | "
                    f"steps={step_count:>3d} | "
                    f"MS={ms:.1f} EN={en:.1f} DL={dl:.1f} | "
                    f"sHV={smoothed_hv:.6f} Pareto={pareto_front.size()}"
                    f"{val_str}{loss_str}")
            print(line)
            log_lines.append(line)

    save_checkpoint(
        path=ckpt_paths['last'],
        encoder=encoder, actor_critic=actor_critic,
        optimizer=trainer.optimizer,
        episode=num_episodes - 1, seed=seed, config=cfg,
        sequence=sequence, feat_dim=feat_dim,
        num_factories=num_factories,
        objective_ranges=ranges,
        dynamic_config=dynamic_config,
        checkpoint_type='last',
        best_val_metric=best_val_hv,
        best_metric_name='avg_val_hv',
        class_name=class_name,
    )

    training_time = time.time() - training_start_time
    time_tracker.total_training_time = training_time

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    summary_lines = [
        f"Seed: {seed}",
        f"Class: {class_name if class_name else '(not set)'}",
        f"Episodes: {num_episodes}",
        f"Training time: {training_time:.2f}s",
        f"Best episode: {best_episode}",
        f"Best val HV (avg over val set): {best_val_hv:.6f}",
        f"Best late episode: {best_late_episode}",
        f"Best late val HV: {best_late_val_hv:.6f}",
        f"Final pareto size: {pareto_front.size()}",
        f"Num training instances: {len(train_instance_paths)}",
        f"Num val instances: {len(val_instance_paths)}",
        f"Instance sampling: {instance_sampling}",
        f"Checkpoints:",
        f"  best: {ckpt_paths['best']}",
        f"  best_late: {ckpt_paths['best_late']}",
        f"  last: {ckpt_paths['last']}",
    ]
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n  Training complete for seed {seed}.")
    print(f"  Best episode: {best_episode}, valHV: {best_val_hv:.6f}")
    print(f"  Training time: {training_time:.2f}s")

    return {
        'best_checkpoint_path': ckpt_paths['best'],
        'best_late_checkpoint_path': ckpt_paths['best_late'],
        'last_checkpoint_path': ckpt_paths['last'],
        'training_log_path': log_path,
        'training_summary_path': summary_path,
        'seed': seed,
        'best_val_metric': best_val_hv,
        'best_episode': best_episode,
    }
