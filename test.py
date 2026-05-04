import os
import sys
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_splits import (
    load_split_file,
    load_split_for_class,
)
from evaluation import evaluate_checkpoint_on_testset

def parse_args():
    parser = argparse.ArgumentParser(
        description="GAT-PPO Test Evaluation - single-class, multi-seed")

    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to JSON split file with train/val/test paths")

    parser.add_argument("--test_dir", type=str, default=None,
                        help="Test root directory. With --class_name, "
                             "recursively scans {test_dir}/{class_name}/**/*.json")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Problem class to test on (e.g. h2_n20_m5). "
                             "REQUIRED when using --test_dir.")

    parser.add_argument("--checkpoint_dir", type=str,
                        default="results",
                        help="Base checkpoint directory. With --class_name, "
                             "the script looks under "
                             "{checkpoint_dir}/{class_name}/checkpoints/GAT_PPO/seed_<s>/. "
                             "Without --class_name, it looks under "
                             "{checkpoint_dir}/seed_<s>/ (legacy layout).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to evaluate (e.g. --seeds 42 43 44)")
    parser.add_argument("--checkpoint_type", type=str, default="best",
                        choices=["best", "best_late", "last"],
                        help="Which checkpoint to evaluate per seed")

    parser.add_argument("--output_dir", type=str, default="results/fronts",
                        help="Output directory for front CSVs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")

    parser.add_argument("--realization_seed", type=int, default=12345,
                        help="Seed for deterministic test realization")

    return parser.parse_args()

def _resolve_checkpoint_path(checkpoint_dir, class_name, seed, ckpt_name):
    candidates = []
    if class_name:
        candidates.append(os.path.join(
            checkpoint_dir, class_name,
            "checkpoints", "GAT_PPO", f"seed_{seed}", ckpt_name))
    candidates.append(os.path.join(
        checkpoint_dir, "checkpoints", "GAT_PPO", f"seed_{seed}", ckpt_name))
    candidates.append(os.path.join(
        checkpoint_dir, f"seed_{seed}", ckpt_name))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"  GAT-PPO Test Evaluation (single-class protocol)")
    print(f"{'='*80}")
    print(f"  Class:           {args.class_name}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Checkpoint type: {args.checkpoint_type}")
    print(f"  Output:          {args.output_dir}")

    if args.split_file:
        if not args.class_name:
            print("  [WARN] --split_file provided without --class_name; "
                  "user must ensure split corresponds to a single class.")
        splits = load_split_file(args.split_file)
        test_paths = splits['test']
    elif args.test_dir:
        if not args.class_name:
            print("\nERROR: --class_name is REQUIRED when using --test_dir.")
            sys.exit(1)
        splits = load_split_for_class(
            train_root=None, val_root=None,
            test_root=args.test_dir,
            class_name=args.class_name,
        )
        test_paths = splits['test']
    else:
        print("\nERROR: Must provide either --split_file or --test_dir + --class_name")
        sys.exit(1)

    if not test_paths:
        print("ERROR: No test instances found.")
        sys.exit(1)

    print(f"  Test instances: {len(test_paths)}")

    if args.class_name:
        out_dir = os.path.join(args.output_dir, args.class_name)
    else:
        out_dir = args.output_dir

    ckpt_name = {
        'best': 'best_model.pth',
        'best_late': 'best_late_model.pth',
        'last': 'last_model.pth',
    }[args.checkpoint_type]

    all_results = []
    for seed in args.seeds:
        ckpt_path = _resolve_checkpoint_path(
            args.checkpoint_dir, args.class_name, seed, ckpt_name)
        if ckpt_path is None:
            print(f"\n  WARNING: Checkpoint not found for seed {seed} "
                  f"(class={args.class_name}, type={args.checkpoint_type}). "
                  f"Skipping.")
            continue
        print(f"\n  [seed {seed}] using checkpoint: {ckpt_path}")

        run_id = f"seed{seed}"

        result = evaluate_checkpoint_on_testset(
            checkpoint_path=ckpt_path,
            test_instance_paths=test_paths,
            config={},
            seed=seed,
            run_id=run_id,
            output_dir=out_dir,
            realization_seed=args.realization_seed,
            device=args.device,
        )
        result['class_name'] = args.class_name
        all_results.append(result)

    print(f"\n{'='*80}")
    print(f"  Test Evaluation Summary  (class={args.class_name})")
    print(f"{'='*80}")
    for r in all_results:
        n_fronts = len(r['exported_paths'])
        print(f"  {r['run_id']}: {n_fronts} fronts exported")
        for inst_id, inst_r in r['per_instance'].items():
            nd_count = len(inst_r['nd_objectives'])
            print(f"    {inst_id}: {nd_count} ND solutions, "
                  f"MS={inst_r['avg_ms']:.2f}, "
                  f"EN={inst_r['avg_en']:.2f}, "
                  f"DL={inst_r['avg_dl']:.2f}")
    print(f"{'='*80}")
    print(f"\n  Front CSVs saved to: {out_dir}/GAT_PPO/")
    print(f"  Next step: python offline_eval.py --fronts_dir {out_dir}")

    return all_results

if __name__ == "__main__":
    main()
