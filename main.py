import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def print_usage():
    print(f"""
{'='*80}
  GAT-PPO for Dynamic DHDNPFSP - Refactored Project Guide
{'='*80}

  The project has been restructured into separate stages:

  1. TRAINING:
     python train.py \\
         --split_file data/splits/split.json \\
         --seeds 42 43 44 \\
         --output_dir results \\
         --device cpu

  2. TESTING (standard evaluation):
     python test.py \\
         --split_file data/splits/split.json \\
         --checkpoint_dir results/checkpoints/GAT_PPO \\
         --seeds 42 43 44 \\
         --output_dir results/fronts \\
         --device cpu

  3. PGPS INFERENCE (post-hoc reranking):
     python pgps_eval.py \\
         --checkpoint results/checkpoints/GAT_PPO/seed_42/best_model.pth \\
         --instance data/test/instance.json \\
         --num_samples 32

  4. OFFLINE EVALUATION (unified HV / Grid-IGD):
     python offline_eval.py \\
         --fronts_dir results/fronts \\
         --output_dir results/metrics

  5. COLLECT FRONTS (merge all algorithm results):
     python collect_fronts.py \\
         --fronts_dir results/fronts \\
         --output_dir results/merged

{'='*80}
  Directory Structure After Running:
{'='*80}

  results/
    checkpoints/
      GAT_PPO/
        seed_42/
          best_model.pth
          best_late_model.pth
          last_model.pth
        seed_43/ ...
    logs/
      train_seed42.log
    summaries/
      train_seed42_summary.txt
    fronts/
      GAT_PPO/
        instance_A/
          seed42_final_front.csv
      GAT_PPO_PGPS/
        instance_A/
          seed42_final_front.csv
    pgps_candidates/
      instance_A/
        seed42_candidates.csv
    merged/
      all_final_fronts.csv
    metrics/
      hv_igd_summary.csv
      hv_igd_detail.csv

{'='*80}
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h', 'help'):
        print_usage()
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from train import main
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from test import main
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'pgps':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from pgps_eval import main
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from offline_eval import main
        main()
    else:
        print_usage()
        print("  Tip: Use 'python main.py train/test/pgps/eval [args]' or run scripts directly.")
