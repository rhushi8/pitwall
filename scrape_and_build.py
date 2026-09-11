#!/usr/bin/env python3
"""Scrape all historical data, then assemble the training dataset.

Run from the project root:
    python scrape_and_build.py

Options pass straight through to the scraper, so see --help for those.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    {
        "name": "Scrape historical races (2018–2024)",
        "cmd":  [sys.executable,
                 str(ROOT / "src/scraper/historical_scraper.py"),
                 "--years", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    },
    {
        "name": "Assemble training dataset",
        "cmd":  [sys.executable,
                 str(ROOT / "src/scraper/assemble_dataset.py")],
    },
]


def main() -> None:
    print("F1 Predictor: data pipeline")
    print("=" * 50)

    for i, step in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {step['name']} …\n")
        result = subprocess.run(step["cmd"])
        if result.returncode != 0:
            print(f"\n✗ Step {i} failed (exit code {result.returncode}). Aborting.")
            sys.exit(result.returncode)
        print(f"\n✓ Step {i} complete.")

    print("\n" + "=" * 50)
    print("Pipeline complete! Next step:")
    print("  python src/train.py --csv data/processed/historical_results.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
