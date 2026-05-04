import subprocess
import sys
import os

CSV_PATH = r"D:\non小论文\算例集\flcmax_100_15.csv"

SEED_START = 201
SEED_END = 300

OUTPUT_DIR = r"D:\non小论文\操作代码\符合论文模型的代码\统一算例\test\f=5 n=100 m=15"

GENERATOR_SCRIPT = r"D:\non小论文\操作代码\符合论文模型的代码\统一算例\generate_eval_instances.py"

NUM_FACTORIES = 5

EXTRA_ARGS = []

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seeds = [str(s) for s in range(SEED_START, SEED_END + 1)]

    cmd = [
        sys.executable,
        GENERATOR_SCRIPT,
        "--csv", CSV_PATH,
        "--output_dir", OUTPUT_DIR,
        "--num_factories", str(NUM_FACTORIES),
        "--seeds", *seeds,
    ] + EXTRA_ARGS

    print("=" * 70)
    print(f"CSV       : {CSV_PATH}")
    print(f"Seeds     : {SEED_START} ~ {SEED_END}  (共 {len(seeds)} 个)")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Factories : {NUM_FACTORIES}")
    print("=" * 70)

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✅ 全部完成！共生成 {len(seeds)} 个 JSON 文件到：\n   {OUTPUT_DIR}")
    else:
        print(f"\n❌ 生成失败，退出码 {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
