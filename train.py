import os
import sys
import argparse
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import TRAIN_CONFIG
from data_splits import (
    load_split_file,
    load_split_from_dirs,
    load_split_for_class,
)
from trainer import train_one_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="GAT-PPO Training - single-class, multi-seed")

    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to JSON split file with train/val/test paths "
                             "(must correspond to a single class)")

    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training root. With --class_name, scans "
                             "{train_dir}/{class_name}/**/*.json")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation root. With --class_name, scans "
                             "{val_dir}/{class_name}/**/*.json")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Problem class to train on (e.g. f2_n20_m5). "
                             "REQUIRED when using --train_dir/--val_dir.")

    parser.add_argument("--instance_sampling", type=str, default="round_robin",
                        choices=["round_robin", "random"],
                        help="How to pick a training instance per episode")

    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds (e.g. --seeds 42 43 44)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")

    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--config_json", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"  GAT-PPO Training Script (single-class protocol)")
    print(f"{'='*80}")
    print(f"  Class:  {args.class_name}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {args.device}")

    if args.split_file:
        if not args.class_name:
            print("  [WARN] --split_file provided without --class_name. "
                  "It is the user's responsibility to ensure the split "
                  "corresponds to exactly ONE class.")
        splits = load_split_file(args.split_file)
    elif args.train_dir and args.val_dir:
        if not args.class_name:
            print("\nERROR: --class_name is REQUIRED when using --train_dir/--val_dir.\n"
                  "       Default behavior is one-class-per-run; mixing classes is\n"
                  "       not supported here. To train multiple classes, loop in a\n"
                  "       shell script.")
            sys.exit(1)
        splits = load_split_for_class(
            train_root=args.train_dir,
            val_root=args.val_dir,
            test_root=None,
            class_name=args.class_name,
        )
    else:
        print("\nERROR: Must provide either --split_file or --train_dir + --val_dir + --class_name")
        sys.exit(1)

    train_paths = splits['train']
    val_paths = splits['val']

    if not train_paths:
        print("ERROR: No training instances found.")
        sys.exit(1)
    if not val_paths:
        print("WARNING: No validation instances found. "
              "best_model.pth will not be saved (no val metric to track).")

    print(f"  Train instances: {len(train_paths)}")
    print(f"  Val instances:   {len(val_paths)}")

    custom_config = {}
    if args.config_json:
        with open(args.config_json, 'r') as f:
            custom_config.update(json.load(f))
    if args.num_episodes is not None:
        custom_config['num_episodes'] = args.num_episodes
    if args.max_steps is not None:
        custom_config['max_steps_per_episode'] = args.max_steps

    if args.class_name:
        seed_output_dir = os.path.join(args.output_dir, args.class_name)
    else:
        seed_output_dir = args.output_dir

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*80}")
        print(f"  Starting training: class={args.class_name}, seed={seed}")
        print(f"{'='*80}")

        result = train_one_seed(
            train_instance_paths=train_paths,
            val_instance_paths=val_paths,
            config=custom_config,
            seed=seed,
            output_dir=seed_output_dir,
            device=args.device,
            class_name=args.class_name,
            instance_sampling=args.instance_sampling,
        )
        result['class_name'] = args.class_name
        all_results.append(result)
        print(f"\n  Seed {seed} complete:")
        print(f"    Best checkpoint: {result['best_checkpoint_path']}")
        print(f"    Best episode:    {result['best_episode']}")
        print(f"    Best val metric: {result['best_val_metric']:.6f}")

    print(f"\n{'='*80}")
    print(f"  Training Summary  (class={args.class_name})")
    print(f"{'='*80}")
    for r in all_results:
        print(f"  Seed {r['seed']}: best_ep={r['best_episode']}, "
              f"val_hv={r['best_val_metric']:.6f}, "
              f"ckpt={r['best_checkpoint_path']}")
    print(f"{'='*80}")

    return all_results

if __name__ == "__main__":
    main()
