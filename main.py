# EE 434 Programming Project main runner file
# Runs all questions sequentially, then saves all output figures to ./outputs/

import os
import sys

from src.random_generator import run_question_1
from src.distributions import run_question_2
from src.mm1_queue import run_question_3
from src.mek1_queue import run_question_4

# Ensure src is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))


OUTPUT_DIR = "outputs"

def main():
    print("=" * 60)
    print("  EE 434 Programming Project")
    print("  Running all questions sequentially")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_question_1(save_dir=OUTPUT_DIR)
    run_question_2(save_dir=OUTPUT_DIR)
    run_question_3(save_dir=OUTPUT_DIR)
    run_question_4(save_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f" Completed 4/4 \r\n Figures saved to: ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()