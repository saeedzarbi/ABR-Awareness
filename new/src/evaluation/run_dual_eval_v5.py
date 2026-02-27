import os

from evaluate_all_models_v5 import run_eval


def main():
    configs = [
        ("raw", "0", "_raw"),
        ("safe", "1", "_safe"),
    ]

    for label, guard, suffix in configs:
        print("=" * 72)
        print(f"Running V5 evaluation ({label}, ABR_SAFETY_GUARD={guard})")
        print("=" * 72)
        os.environ["ABR_SAFETY_GUARD"] = guard
        run_eval(episodes_per_video=20, suffix=suffix)


if __name__ == "__main__":
    main()

