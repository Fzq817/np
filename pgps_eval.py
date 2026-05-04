import os
import csv
import sys
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from checkpoint_utils import restore_models_from_checkpoint
from utils import load_instance_json
from evaluation import build_realization_from_json, export_front_csv, filter_nondominated
from data_splits import get_instance_id
from pgps import evaluate_with_pgps_from_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="PGPS Inference - Post-hoc reranking on test instances")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint .pth file")
    parser.add_argument("--instance", type=str, required=True,
                        help="Path to test instance JSON file")

    parser.add_argument("--pref", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                        help="Preference vector (3 values, e.g. --pref 0.4 0.3 0.3)")
    parser.add_argument("--e_ref", type=float, default=None,
                        help="Reference energy E_ref. "
                             "Reported experiments use a *manually specified* E_ref; "
                             "auto-estimation from a random rollout is a fallback only.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Reference-steering coefficient beta")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of candidate trajectories N")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for PGPS rollouts")

    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier (default: derived from checkpoint)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")

    return parser.parse_args()

def save_pgps_runtime_csv(
    csv_path: str,
    instance_id: str,
    run_id: str,
    num_samples: int,
    pref,
    e_ref,
    beta,
    pgps_result: dict,
    device: str,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    fieldnames = [
        'instance_id', 'run_id', 'num_samples',
        'pref_ms', 'pref_en', 'pref_dl',
        'e_ref', 'beta',
        'total_pgps_wall_time_sec',
        'rollout_wall_time_sec',
        'rerank_wall_time_sec',
        'mean_steps', 'std_steps', 'total_steps',
        'device',
    ]

    row = {
        'instance_id': instance_id,
        'run_id': run_id,
        'num_samples': num_samples,
        'pref_ms': f"{pref[0]:.4f}",
        'pref_en': f"{pref[1]:.4f}",
        'pref_dl': f"{pref[2]:.4f}",
        'e_ref': f"{e_ref:.4f}" if e_ref is not None else "",
        'beta': f"{beta:.4f}",
        'total_pgps_wall_time_sec': f"{pgps_result['total_pgps_wall_time_sec']:.6f}",
        'rollout_wall_time_sec': f"{pgps_result['rollout_wall_time_sec']:.6f}",
        'rerank_wall_time_sec': f"{pgps_result['rerank_wall_time_sec']:.6f}",
        'mean_steps': f"{pgps_result['mean_steps']:.4f}",
        'std_steps': f"{pgps_result['std_steps']:.4f}",
        'total_steps': int(pgps_result['total_steps']),
        'device': device,
    }

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"[PGPS] Runtime summary saved to: {csv_path}")

def main():
    args = parse_args()

    if args.e_ref is not None:
        e_ref_note = f"Using manually specified E_ref = {args.e_ref}"
    else:
        e_ref_note = "Using auto-estimated E_ref (fallback only)"

    print(f"\n{'='*80}")
    print(f"  PGPS Inference (Post-hoc Guided Policy Search)")
    print(f"{'='*80}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Instance:   {args.instance}")
    print(f"  Pref:       {args.pref}")
    print(f"  Beta:       {args.beta}")
    print(f"  Samples:    {args.num_samples}")
    print(f"  E_ref:      {e_ref_note}")
    print(f"  Device:     {args.device}")
    print(f"{'='*80}")

    instance_id = get_instance_id(args.instance)

    if args.run_id is None:
        ckpt_dir = os.path.basename(os.path.dirname(args.checkpoint))
        if ckpt_dir.startswith("seed_"):
            args.run_id = ckpt_dir.replace("_", "")
        else:
            args.run_id = "pgps_run"

    pref = np.array(args.pref)

    pgps_result = evaluate_with_pgps_from_checkpoint(
        checkpoint_path=args.checkpoint,
        instance_path=args.instance,
        pref=pref,
        e_ref=args.e_ref,
        beta=args.beta,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )

    e_ref_used = pgps_result.get('e_ref', args.e_ref)

    candidates_dir = os.path.join(args.output_dir, "pgps_candidates", instance_id)
    os.makedirs(candidates_dir, exist_ok=True)
    candidates_path = os.path.join(candidates_dir, f"{args.run_id}_candidates.csv")

    from pgps import save_pgps_candidates_csv
    save_pgps_candidates_csv(
        pgps_result['all_trajectory_results'],
        candidates_path,
        best_index=pgps_result['best_index'],
    )

    runtime_dir = os.path.join(args.output_dir, "pgps_runtime", instance_id)
    runtime_path = os.path.join(runtime_dir, f"{args.run_id}_runtime.csv")
    save_pgps_runtime_csv(
        csv_path=runtime_path,
        instance_id=instance_id,
        run_id=args.run_id,
        num_samples=args.num_samples,
        pref=args.pref,
        e_ref=e_ref_used,
        beta=args.beta,
        pgps_result=pgps_result,
        device=args.device,
    )

    all_objs = []
    for r in pgps_result['all_trajectory_results']:
        if r['makespan'] > 0 and r['energy'] > 0:
            all_objs.append([r['makespan'], r['energy'], r['delay']])

    if all_objs:
        nd_objs = filter_nondominated(np.array(all_objs))
        front_path = export_front_csv(
            nd_objs, instance_id, args.run_id,
            output_dir=os.path.join(args.output_dir, "fronts", "GAT_PPO_PGPS"),
            algorithm="GAT_PPO_PGPS",
        )
    else:
        print("  WARNING: No valid PGPS objectives collected.")
        front_path = None

    best = pgps_result['best_trajectory_result']
    print(f"\n{'='*80}")
    print(f"  PGPS Results")
    print(f"{'='*80}")
    print(f"  Best trajectory: MS={best['makespan']:.2f}, "
          f"EN={best['energy']:.2f}, DL={best['delay']:.2f}")
    print(f"  Best score: {best['pgps_score']:.6f}")
    print(f"  Total PGPS wall time: {pgps_result['total_pgps_wall_time_sec']:.4f} s  "
          f"(rollout {pgps_result['rollout_wall_time_sec']:.4f} s, "
          f"rerank {pgps_result['rerank_wall_time_sec']:.4f} s)")
    print(f"  Mean candidate steps: {pgps_result['mean_steps']:.2f} "
          f"(total {pgps_result['total_steps']})")
    print(f"  Candidates saved: {candidates_path}")
    print(f"  Runtime   saved: {runtime_path}")
    if front_path:
        print(f"  Front saved: {front_path}")
    print(f"{'='*80}")

    return pgps_result

if __name__ == "__main__":
    main()
