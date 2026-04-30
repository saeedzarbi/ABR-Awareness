import os

from evaluate_all_models_v12 import run_eval


def main():
    configs = [
        ("policy", "_v12_policy"),
    ]

    # V12 focuses on method-level safety; no global system guard here.
    os.environ.setdefault("ABR_SAFETY_GUARD", "0")

    for label, suffix in configs:
        print("=" * 72)
        print(f"Running V12 evaluation ({label})")
        print("=" * 72)
        run_eval(episodes_per_video=20, suffix=suffix)


if __name__ == "__main__":
    main()

