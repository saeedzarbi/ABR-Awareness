import os

from evaluate_all_models_v7 import run_eval


def main():
    configs = [
        ("raw", "0", "_v7_raw"),
        ("light-guard", "light", "_v7_light"),
        ("strong-guard", "2", "_v7_safe"),
    ]

    for label, guard, suffix in configs:
        print("=" * 72)
        print(f"Running V7 evaluation ({label}, ABR_SAFETY_GUARD={guard})")
        print("=" * 72)
        os.environ["ABR_SAFETY_GUARD"] = guard
        run_eval(episodes_per_video=20, suffix=suffix)


if __name__ == "__main__":
    main()

